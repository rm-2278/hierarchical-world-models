import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

# ==========================================
#  Helpers: SymLog & Two-Hot
# ==========================================
def symlog(x):
    """対称対数変換"""
    return torch.sign(x) * torch.log(1 + torch.abs(x))

def symexp(x):
    """対称指数変換"""
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)

def two_hot(x, cfg):
    """Symlog空間でのTwo-hot変換 (Clip -> Symlog -> Bin)"""
    if cfg.num_bins == 0: return x
    
    # 1. 生のスケールでクリップ
    x = x.clamp(cfg.vmin, cfg.vmax)
    # 2. Symlog変換
    x = symlog(x)
    
    # 3. Symlog空間でのビン範囲計算
    v_min_log = symlog(torch.tensor(cfg.vmin, device=x.device))
    v_max_log = symlog(torch.tensor(cfg.vmax, device=x.device))
    
    bin_size = (v_max_log - v_min_log) / (cfg.num_bins - 1)
    bin_idx = torch.floor((x - v_min_log) / bin_size).long()
    bin_offset = ((x - v_min_log) / bin_size - bin_idx)
    
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset.unsqueeze(1))
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1).clamp(max=cfg.num_bins-1), bin_offset.unsqueeze(1))
    return soft_two_hot

def two_hot_inv(x, cfg):
    """Two-hot -> 期待値 -> Symexp"""
    if cfg.num_bins == 0: return x
    v_min_log = symlog(torch.tensor(cfg.vmin, device=x.device))
    v_max_log = symlog(torch.tensor(cfg.vmax, device=x.device))
    dreg_bins = torch.linspace(v_min_log, v_max_log, cfg.num_bins, device=x.device)
    
    x = F.softmax(x, dim=-1)
    x_symlog = (x * dreg_bins).sum(dim=-1)
    return symexp(x_symlog)


# ==========================================
#  Multi-Task LoRA Layers
# ==========================================
class MultiTaskLoRALinear(nn.Module):
    """
    タスクIDに応じて LoRA パラメータ (A, B) を切り替える層
    """
    def __init__(self, in_features, out_features, num_tasks, rank=0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.num_tasks = num_tasks
        
        # Base Model (Shared across tasks)
        self.linear = nn.Linear(in_features, out_features)
        
        # Multi-Task LoRA weights
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(num_tasks, rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(num_tasks, out_features, rank))
            self.scale = 1.0 / rank
            
            # 初期化
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x, task_idx):
        # Base: Wx
        out = self.linear(x)
        
        if self.rank > 0:
            # 修正(3): task_idx が None の場合はデフォルト(0)を使用するフォールバック
            if task_idx is None:
                task_idx = 0

            # バッチ内のタスクID処理
            if isinstance(task_idx, int):
                # 全バッチで同一タスクの場合
                idx = task_idx % self.num_tasks
                A = self.lora_A[idx]
                B = self.lora_B[idx]
                lora_out = (x @ A.T) @ B.T
                
            elif isinstance(task_idx, torch.Tensor):
                # バッチごとにタスクが異なる場合
                # task_idx が device に載っていることを想定
                idx = task_idx.to(self.lora_A.device).long() % self.num_tasks
                # idx = task_idx.long() % self.num_tasks
                
                A = self.lora_A[idx]
                B = self.lora_B[idx]
                
                # (B, 1, in) @ (B, in, r) -> (B, 1, r)
                # (B, 1, r) @ (B, r, out) -> (B, 1, out)
                lora_out = torch.bmm(x.unsqueeze(1), A.transpose(1, 2))
                lora_out = torch.bmm(lora_out, B.transpose(1, 2)).squeeze(1)
            else:
                 # Fallback for unexpected types
                 idx = 0
                 A = self.lora_A[idx]
                 B = self.lora_B[idx]
                 lora_out = (x @ A.T) @ B.T

            out = out + lora_out * self.scale
            
        return self.ln(self.dropout(out))


class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
        self.tau = getattr(cfg, 'simnorm_temp', 0.1)

    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x / self.tau, dim=-1)
        return x.view(*shp)


class TaskAwareMLP(nn.Module):
    """
    nn.Sequential の代わりに、task_idx を伝播できる MLP コンテナ
    """
    def __init__(self, in_dim, mlp_dims, out_dim, cfg=None, act=None, dropout=0.):
        super().__init__()
        if isinstance(mlp_dims, int): mlp_dims = [mlp_dims]
        dims = [in_dim] + mlp_dims + [out_dim]
        
        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        
        rank = getattr(cfg, 'lora_rank', 0) if cfg else 0
        num_tasks = getattr(cfg, 'num_tasks', 1) if cfg else 1

        for i in range(len(dims) - 2):
            self.layers.append(
                MultiTaskLoRALinear(dims[i], dims[i+1], num_tasks, rank=rank, dropout=dropout)
            )
            self.acts.append(act if act else nn.Mish())
            
        # 出力層: 通常の Linear
        self.output_layer = nn.Linear(dims[-2], dims[-1])

    def forward(self, x, task_idx=None):
        for layer, act in zip(self.layers, self.acts):
            x = layer(x, task_idx)
            x = act(x)
        x = self.output_layer(x)
        return x

def mlp(in_dim, mlp_dims, out_dim, cfg=None, act=None, dropout=0.):
    return TaskAwareMLP(in_dim, mlp_dims, out_dim, cfg, act, dropout)


# ==========================================
#  World Model (Multi-Task LoRA)
# ==========================================
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Task Embedding
        if hasattr(cfg, 'num_tasks') and cfg.num_tasks > 1:
            if not hasattr(cfg, 'task_dim'): cfg.task_dim = 32
            self.task_emb = nn.Embedding(cfg.num_tasks, cfg.task_dim)
            self._use_task_emb = True
        else:
            self.task_emb = None
            self._use_task_emb = False
            cfg.task_dim = 0
            cfg.num_tasks = 1

        # 各コンポーネント
        self._encoder_mlp = mlp(cfg.obs_shape, [cfg.enc_dim], cfg.latent_dim, cfg)
        # self._encoder_simnorm = SimNorm(cfg)
        self._encoder_simnorm = nn.LayerNorm(cfg.latent_dim)
        
        self._dynamics_mlp = mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 
                                 [cfg.mlp_dim, cfg.mlp_dim], cfg.latent_dim, cfg)
        # self._dynamics_simnorm = SimNorm(cfg)
        self._dynamics_simnorm = nn.LayerNorm(cfg.latent_dim)
        
        self._reward = mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 
                           [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1), cfg)
        
        self._pi = mlp(cfg.latent_dim + cfg.task_dim, 
                       [cfg.mlp_dim, cfg.mlp_dim], 2 * cfg.action_dim, cfg)
        
        self._Qs = nn.ModuleList([
            mlp(cfg.latent_dim + cfg.action_dim + cfg.task_dim, 
                [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1), cfg) 
            for _ in range(cfg.num_q)
        ])
        self.init_target_q()

    def init_target_q(self):
        self._target_Qs = deepcopy(self._Qs)
        for p in self._target_Qs.parameters(): p.requires_grad = False

    def soft_update_target_q(self):
        with torch.no_grad():
            for p, p_targ in zip(self._Qs.parameters(), self._target_Qs.parameters()):
                p_targ.data.mul_(1 - self.cfg.tau)
                p_targ.data.add_(self.cfg.tau * p.data)

    def _get_emb(self, B, task_idx=None):
        if self._use_task_emb:
            if task_idx is None: task_idx = getattr(self.cfg, 'task_id', 0)
            
            # 修正(2): モデルのパラメータから正しいデバイスを取得
            device = next(self.parameters()).device

            if isinstance(task_idx, int):
                task_idx = torch.tensor([task_idx], device=device)
            elif isinstance(task_idx, torch.Tensor):
                task_idx = task_idx.to(device)
            
            if task_idx.dim() == 0: task_idx = task_idx.unsqueeze(0)
            
            e = self.task_emb(task_idx) 
            if e.shape[0] == 1 and B > 1:
                e = e.repeat(B, 1)
            return e
        return None

    def encode(self, obs, task_idx=None): 
        z = self._encoder_mlp(obs, task_idx)
        return self._encoder_simnorm(z)

    def next(self, z, a, task_idx=None):
        x = [z, a]
        if self._use_task_emb: x.append(self._get_emb(z.shape[0], task_idx))
        z_out = self._dynamics_mlp(torch.cat(x, dim=-1), task_idx)
        return self._dynamics_simnorm(z_out)

    def reward(self, z, a, task_idx=None):
        x = [z, a]
        if self._use_task_emb: x.append(self._get_emb(z.shape[0], task_idx))
        return self._reward(torch.cat(x, dim=-1), task_idx)
    
    def pi(self, z, task_idx=None):
        x = [z]
        if self._use_task_emb: x.append(self._get_emb(z.shape[0], task_idx))
        
        out = self._pi(torch.cat(x, dim=-1), task_idx)
        mu, log_std = out.chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        return mu, log_std

    def Q(self, z, a, task_idx=None, target=False):
        x = [z, a]
        if self._use_task_emb: x.append(self._get_emb(z.shape[0], task_idx))
        inp = torch.cat(x, dim=-1)
        
        qs = torch.stack([q(inp, task_idx) for q in (self._target_Qs if target else self._Qs)])
        return qs


# ==========================================
#  TD-MPC2 Agent
# ==========================================
class TDMPC2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        
        # MPPI state
        self.prev_mean = torch.zeros(cfg.horizon, cfg.action_dim, device=self.device)
        self.prev_std = 2.0 * torch.ones(cfg.horizon, cfg.action_dim, device=self.device)
        
    def act(self, obs, t0=False, eval_mode=False, task_idx=None):
        with torch.no_grad():
            obs = obs.to(self.device)

            if obs.ndim == 1:
                obs = obs.unsqueeze(0)
            
            z = self.model.encode(obs, task_idx)
            
            if eval_mode:
                a = self.model.pi(z, task_idx)[0]
            else:
                a = self.plan(z, t0=t0, eval_mode=eval_mode, task_idx=task_idx)
            
            if a.ndim == 2 and a.shape[0] == 1:
                a = a.squeeze(0)
                
            return a.cpu()

            # z = self.model.encode(obs.to(self.device), task_idx)
            # if eval_mode:
            #     a = self.model.pi(z, task_idx)[0]
            # else:
            #     a = self.plan(z, t0=t0, eval_mode=eval_mode, task_idx=task_idx)
            # return a.cpu()

    def plan(self, z, t0=False, eval_mode=False, task_idx=None):
        cfg = self.cfg
        
        # 1. Initialize
        if t0:
            mu, _ = self.model.pi(z, task_idx)
            self.prev_mean = mu.repeat(cfg.horizon, 1)
            self.prev_std = 2.0 * torch.ones_like(self.prev_mean)
        else:
            self.prev_mean = torch.roll(self.prev_mean, -1, dims=0)
            self.prev_mean[-1] = self.prev_mean[-2]
            self.prev_std = torch.roll(self.prev_std, -1, dims=0)
            self.prev_std[-1] = 2.0 

        mean = self.prev_mean.clone()
        std = self.prev_std.clone()
        
        z_expansion = z.repeat(cfg.num_samples, 1)

        for i in range(cfg.iterations):
            # 2. Sampling
            noise = torch.randn(cfg.num_samples, cfg.horizon, cfg.action_dim, device=self.device)
            actions = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * noise, -1, 1)
            
            # Policy Mixture
            if i == 0: 
                mixture_ratio = 0.05
                num_pi_trajs = int(cfg.num_samples * mixture_ratio)
                if num_pi_trajs > 0:
                    pi_mu, pi_log_std = self.model.pi(z, task_idx)
                    pi_std = pi_log_std.exp()
                    pi_noise = torch.randn(num_pi_trajs, cfg.horizon, cfg.action_dim, device=self.device)
                    
                    # 修正(1): 形状バグの修正 (B=1想定)
                    # pi_mu: (1, A) -> (1, 1, A) にしてブロードキャスト
                    pi_actions = torch.clamp(
                        pi_mu.squeeze(0).view(1, 1, -1) + 
                        pi_std.squeeze(0).view(1, 1, -1) * pi_noise, 
                        -1, 1
                    )
                    
                    rand_idxs = torch.randperm(cfg.num_samples, device=self.device)[:num_pi_trajs]
                    actions[rand_idxs] = pi_actions

            # 3. Rollout
            cumulative_reward = 0
            curr_z = z_expansion
            
            for t in range(cfg.horizon):
                a_t = actions[:, t]
                
                r_dist = self.model.reward(curr_z, a_t, task_idx)
                r_t = two_hot_inv(r_dist, cfg)
                cumulative_reward += r_t * (cfg.rho ** t)
                
                curr_z = self.model.next(curr_z, a_t, task_idx)
            
            # Terminal Value
            pi_a, _ = self.model.pi(curr_z, task_idx)
            qs = self.model.Q(curr_z, pi_a, task_idx, target=True)
            q_val = two_hot_inv(qs, cfg).min(0)[0]
            
            scores = cumulative_reward + (cfg.rho ** cfg.horizon) * q_val
            
            # 4. Update
            temperature = getattr(cfg, 'temperature', 0.5)
            weights = F.softmax(scores / temperature, dim=0)
            
            w_expanded = weights.view(-1, 1, 1)
            new_mean = (w_expanded * actions).sum(dim=0)
            var = (w_expanded * (actions - new_mean.unsqueeze(0))**2).sum(dim=0)
            new_std = torch.sqrt(var + 1e-6)
            
            mean = 0.1 * mean + 0.9 * new_mean
            std = 0.1 * std + 0.9 * new_std
            std = torch.clamp(std, cfg.min_std, cfg.max_std)

        self.prev_mean = mean.detach()
        self.prev_std = std.detach()
        
        action = mean[0]
        if not eval_mode:
            action = action + std[0] * torch.randn_like(action)
            
        return action.clamp(-1, 1)

    def update(self, buffer):
        batch = buffer.sample()
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        
        # 簡易対応: バッファに task_idx が保存されていない場合、ConfigのIDを使う
        # (本格的なマルチタスク学習にはbufferの改修が必要)
        task_idx = getattr(self.cfg, 'task_id', 0)
        
        z = self.model.encode(obs[0], task_idx)
        
        consistency_loss = 0
        reward_loss = 0
        value_loss = 0
        pi_loss = 0
        
        for t in range(self.cfg.horizon):
            z_next_pred = self.model.next(z, action[t], task_idx)
            reward_pred = self.model.reward(z, action[t], task_idx)
            qs_pred = self.model.Q(z, action[t], task_idx)
            
            with torch.no_grad():
                z_next_target = self.model.encode(obs[t+1], task_idx)
                reward_target = two_hot(reward[t], self.cfg)
                
                next_pi_action, _ = self.model.pi(z_next_target, task_idx)
                target_qs_all = self.model.Q(z_next_target, next_pi_action, task_idx, target=True)
                
                if self.cfg.num_q > 2:
                    indices = torch.randperm(self.cfg.num_q, device=self.device)[:2]
                    target_qs = target_qs_all[indices]
                else:
                    target_qs = target_qs_all
                
                target_q_val = two_hot_inv(target_qs, self.cfg).min(0)[0]
                target_val = reward[t] + self.cfg.rho * target_q_val
                target_val_dist = two_hot(target_val, self.cfg)

            rho_pow = (self.cfg.rho ** t)
            
            consistency_loss += F.mse_loss(z_next_pred, z_next_target) * rho_pow
            reward_loss += -(reward_target * F.log_softmax(reward_pred, -1)).sum(-1).mean() * rho_pow
            
            for q_pred in qs_pred:
                value_loss += -(target_val_dist * F.log_softmax(q_pred, -1)).sum(-1).mean() * rho_pow
            
            pi_action, pi_log_std = self.model.pi(z.detach(), task_idx)
            q_pi_all = self.model.Q(z.detach(), pi_action, task_idx, target=False)
            q_pi_val = two_hot_inv(q_pi_all, self.cfg).min(0)[0]
            
            entropy = 0.5 * pi_log_std.shape[1] * (1.0 + math.log(2 * math.pi)) + pi_log_std.sum(dim=-1)
            pi_loss += - (q_pi_val + self.cfg.entropy_coef * entropy).mean() * rho_pow

            z = z_next_pred

        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss +
            pi_loss
        )
        
        self.optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.model.soft_update_target_q()
        
        return {
            "loss": total_loss.item(),
            "pi_loss": pi_loss.item(),
            "rw_loss": reward_loss.item(),
            "v_loss": value_loss.item()
        }