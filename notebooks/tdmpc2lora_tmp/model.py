import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

# --- 1. Layers (LoRA対応) ---
class TaskAwareLinear(nn.Module):
    def __init__(self, in_features, out_features, rank=0, dropout=0.0):
        super().__init__()
        self.rank = rank
        self.linear = nn.Linear(in_features, out_features)
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.scale = 1.0 / rank
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.ln = nn.LayerNorm(out_features)

    def forward(self, x):
        out = self.linear(x)
        if self.rank > 0:
            lora_out = (x @ self.lora_A.T) @ self.lora_B.T
            out = out + lora_out * self.scale
        return self.ln(self.dropout(out))

class NormedLinear(nn.Linear):
    def __init__(self, in_features, out_features, act=None):
        super().__init__(in_features, out_features)
        self.ln = nn.LayerNorm(out_features)
        self.act = act if act else nn.Mish()
    def forward(self, x):
        return self.act(self.ln(super().forward(x)))

def mlp(in_dim, mlp_dims, out_dim, cfg=None, act=None, dropout=0.):
    if isinstance(mlp_dims, int): mlp_dims = [mlp_dims]
    dims = [in_dim] + mlp_dims + [out_dim]
    layers = []
    # Configからrankを取得。なければ0
    rank = getattr(cfg, 'lora_rank', 0) if cfg else 0

    for i in range(len(dims) - 2):
        # 中間層にLoRAを適用
        layers.append(TaskAwareLinear(dims[i], dims[i+1], rank=rank, dropout=dropout))
        layers.append(act if act else nn.Mish())
    
    # 出力層は通常のLinear (LoRAなし)
    layers.append(nn.Linear(dims[-2], dims[-1])) 
    return nn.Sequential(*layers)

class SimNorm(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.simnorm_dim
    def forward(self, x):
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

# --- 2. World Model ---
class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self._encoder = mlp(cfg.obs_shape, [cfg.enc_dim], cfg.latent_dim, cfg)
        self._dynamics = mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.latent_dim, cfg, act=SimNorm(cfg))
        self._reward = mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1), cfg)
        self._pi = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2*cfg.action_dim, cfg)
        
        # Q関数 (Ensemble)
        self._Qs = nn.ModuleList([
            mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1), cfg) 
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

    def encode(self, obs): return self._encoder(obs)
    def next(self, z, a): return self._dynamics(torch.cat([z, a], dim=-1))
    def reward(self, z, a): return self._reward(torch.cat([z, a], dim=-1))
    
    def pi(self, z):
        # Policy output: mean and log_std
        mu, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
        return mu, log_std

    def Q(self, z, a, target=False):
        x = torch.cat([z, a], dim=-1)
        qs = torch.stack([q(x) for q in (self._target_Qs if target else self._Qs)])
        return qs

# --- 3. TD-MPC2 Agent (MPC & Improved Loss) ---
def two_hot_inv(x, cfg):
    if cfg.num_bins == 0: return x
    dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
    x = F.softmax(x, dim=-1)
    return torch.sum(x * dreg_bins, dim=-1)

def two_hot(x, cfg):
    if cfg.num_bins == 0: return x
    x = x.view(-1).clamp(cfg.vmin, cfg.vmax)
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
    return soft_two_hot

class TDMPC2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = WorldModel(cfg).to(self.device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        
    def act(self, obs, eval_mode=False):
        """推論時の行動決定 (MPCを使用)"""
        with torch.no_grad():
            z = self.model.encode(obs.to(self.device))
            if self.cfg.mpc:
                a = self.plan(z, eval_mode=eval_mode)
            else:
                # MPCを使わない場合はPolicyの平均を出力
                a = self.model.pi(z)[0]
            return a.cpu()

    def plan(self, z, eval_mode=False):
        """MPC (MPPI / CEM-like) による計画"""
        # z: (1, latent_dim)
        cfg = self.cfg
        
        # 1. Initialize distribution (Policyの出力から開始)
        # 計画のために z を複製: (num_samples, latent_dim)
        z = z.repeat(cfg.num_samples, 1)
        
        mean = torch.zeros(cfg.horizon, cfg.action_dim, device=self.device)
        std = 2.0 * torch.ones(cfg.horizon, cfg.action_dim, device=self.device)
        
        # 初期解としてPolicyの出力を利用 (TD-MPCの特徴)
        # 実際の実装では、ここでPolicyを使って初期 mean をシフトさせたりします
        # 簡易版として、ノイズの中心を0とします
        
        for i in range(cfg.iterations):
            # 2. Sample actions
            # ノイズ生成
            noise = torch.randn(cfg.num_samples, cfg.horizon, cfg.action_dim, device=self.device)
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * noise
            
            # クランプ (Action Space)
            actions = torch.clamp(actions, -1, 1)

            # 3. Rollout (Latent Space)
            # 最初の数サンプルはPolicy由来のものにするなどの工夫もありますが、ここでは省略
            
            cumulative_reward = 0
            curr_z = z
            
            for t in range(cfg.horizon):
                a_t = actions[:, t]
                # 次の状態予測
                curr_z = self.model.next(curr_z, a_t)
                # 報酬予測 (Two-hot -> scalar)
                r_t_dist = self.model.reward(curr_z, a_t)
                r_t = two_hot_inv(r_t_dist, cfg)
                cumulative_reward += r_t * (cfg.rho ** t)
            
            # 終端価値 (Value Function)
            # Q(z_H, pi(z_H))
            pi_action, _ = self.model.pi(curr_z)
            q_vals = self.model.Q(curr_z, pi_action, target=True) # (num_q, num_samples, num_bins)
            # TD-MPC2: Min clipped Q (Double Q)
            q_val = two_hot_inv(q_vals, cfg).min(0)[0] # (num_samples,)
            
            total_score = cumulative_reward + cfg.rho ** cfg.horizon * q_val
            
            # 4. Update distribution (Elites Selection / MPPI Weighting)
            # 上位 num_elites を選択
            _, topk_idxs = torch.topk(total_score, cfg.num_elites)
            elites = actions[topk_idxs] # (num_elites, horizon, action_dim)
            
            # 新しい平均と分散
            new_mean = elites.mean(dim=0)
            new_std = elites.std(dim=0)
            
            # Update (Momentum)
            mean = 0.1 * mean + 0.9 * new_mean
            std = 0.1 * std + 0.9 * new_std
            std = torch.clamp(std, cfg.min_std, cfg.max_std)
        
        # 最終的な行動: 最良の平均、または最良サンプルの最初の行動
        return mean[0]

    def update(self, buffer):
        batch = buffer.sample()
        obs = batch['obs'].to(self.device)
        action = batch['action'].to(self.device)
        reward = batch['reward'].to(self.device)
        
        # --- Encoder ---
        z = self.model.encode(obs[0])
        
        consistency_loss, reward_loss, value_loss, pi_loss = 0, 0, 0, 0
        
        for t in range(self.cfg.horizon):
            # 予測
            z_next_pred = self.model.next(z, action[t])
            reward_pred = self.model.reward(z, action[t])
            qs_pred = self.model.Q(z, action[t]) # (num_q, batch, bins)
            
            # ターゲット作成
            with torch.no_grad():
                z_next_target = self.model.encode(obs[t+1])
                reward_target = two_hot(reward[t].view(-1), self.cfg)
                
                # TD Target
                next_pi_action, _ = self.model.pi(z_next_target)
                target_qs = self.model.Q(z_next_target, next_pi_action, target=True)
                target_q_val = two_hot_inv(target_qs, self.cfg).min(0)[0] # Min clipping
                target_val = reward[t].view(-1) + self.cfg.rho * target_q_val.view(-1)
                target_val_dist = two_hot(target_val, self.cfg)

            # --- Losses ---
            # 1. Consistency Loss (Latent state prediction)
            consistency_loss += F.mse_loss(z_next_pred, z_next_target) * (self.cfg.rho ** t)
            
            # 2. Reward Loss
            reward_loss += -(reward_target * F.log_softmax(reward_pred, -1)).sum(-1).mean() * (self.cfg.rho ** t)
            
            # 3. Value Loss (Critic)
            for q_pred in qs_pred:
                value_loss += -(target_val_dist * F.log_softmax(q_pred, -1)).sum(-1).mean() * (self.cfg.rho ** t)
            
            # 4. Policy Loss (Actor) - 追加！
            # Policy should maximize Q(z, a) + Entropy
            pi_action, pi_log_std = self.model.pi(z.detach()) # zは止めてPolicyだけ更新
            
            # Q-value estimate for policy
            q_pi = self.model.Q(z.detach(), pi_action, target=False)
            q_pi_val = two_hot_inv(q_pi, self.cfg).min(0)[0]
            
            # Entropy (Gaussian entropy)
            entropy = 0.5 * pi_log_std.shape[1] * (1.0 + math.log(2 * math.pi)) + pi_log_std.sum(dim=-1)
            
            # Maximize Q + Entropy  => Minimize -(Q + alpha * H)
            rho_t = (self.cfg.rho ** t)
            pi_loss += - (q_pi_val + self.cfg.entropy_coef * entropy).mean() * rho_t

            z = z_next_pred

        total_loss = (
            self.cfg.consistency_coef * consistency_loss +
            self.cfg.reward_coef * reward_loss +
            self.cfg.value_coef * value_loss +
            pi_loss # 係数は通常1.0または小さめ
        )
        
        self.optim.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
        self.optim.step()
        self.model.soft_update_target_q()
        
        return {
            "loss": total_loss.item(), 
            "rw_loss": reward_loss.item(),
            "pi_loss": pi_loss.item()
        }

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from copy import deepcopy
# import math

# # Models
# class NormedLinear(nn.Linear):
#     def __init__(self, in_features, out_features, act=None):
#         super().__init__(in_features, out_features)
#         self.ln = nn.LayerNorm(out_features)
#         self.act = act if act else nn.Mish()
#     def forward(self, x):
#         return self.act(self.ln(super().forward(x)))

# # Low rank adaptation (LoRA) layers
# class TaskAwareLinear(nn.Module):
#     def __init__(self, in_features, out_features, rank=0, dropout=0.0):
#         super().__init__()
#         self.rank = rank
        
#         # Base weight (W)
#         self.linear = nn.Linear(in_features, out_features)
        
#         # LoRA weights (B * A)
#         if rank > 0:
#             self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
#             self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
#             self.scale = 1.0 / rank # スケーリング係数
            
#             # 初期化: AはKaiming, Bは0 (学習開始時は影響ゼロにするため)
#             nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
#             nn.init.zeros_(self.lora_B)

#         self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
#         self.ln = nn.LayerNorm(out_features) # NormedLinearの機能を統合

#     def forward(self, x):
#         # Base output: Wx
#         out = self.linear(x)
        
#         # LoRA output: BAx
#         if self.rank > 0:
#             lora_out = (x @ self.lora_A.T) @ self.lora_B.T
#             out = out + lora_out * self.scale
            
#         return self.ln(out) 

# # def mlp(in_dim, mlp_dims, out_dim, act=None, dropout=0.):
# #     if isinstance(mlp_dims, int): mlp_dims = [mlp_dims]
# #     dims = [in_dim] + mlp_dims + [out_dim]
# #     layers = []
# #     for i in range(len(dims) - 2):
# #         layers.append(NormedLinear(dims[i], dims[i+1]))
# #         if dropout > 0: layers.append(nn.Dropout(dropout))
# #     layers.append(nn.Linear(dims[-2], dims[-1])) 
# #     return nn.Sequential(*layers)
# def mlp(in_dim, mlp_dims, out_dim, cfg=None, act=None, dropout=0.):
#     if isinstance(mlp_dims, int): mlp_dims = [mlp_dims]
#     dims = [in_dim] + mlp_dims + [out_dim]
#     layers = []
    
#     # rank情報の取得
#     rank = getattr(cfg, 'lora_rank', 0) if cfg else 0

#     for i in range(len(dims) - 2):
#         # NormedLinear の代わりに TaskAwareLinear を使用
#         layers.append(TaskAwareLinear(dims[i], dims[i+1], rank=rank, dropout=dropout))
#         if act: # actが指定されていなければMishを使うようにTaskAwareLinear内で制御しても良い
#             layers.append(act if act else nn.Mish())
            
#     # 最終層は通常Linearのままにするか、ここもLoRAにするかは設計次第
#     # 一般的には中間層にAdapterを入れます
#     layers.append(nn.Linear(dims[-2], dims[-1])) 
#     return nn.Sequential(*layers)

# class SimNorm(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.dim = cfg.simnorm_dim
#     def forward(self, x):
#         shp = x.shape
#         x = x.view(*shp[:-1], -1, self.dim)
#         x = F.softmax(x, dim=-1)
#         return x.view(*shp)

# class SimpleEnsemble(nn.Module):
#     def __init__(self, modules):
#         super().__init__()
#         self.models = nn.ModuleList(modules)
#     def forward(self, x):
#         return torch.stack([m(x) for m in self.models])

# class WorldModel(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self._encoder = mlp(cfg.obs_shape, [cfg.enc_dim], cfg.latent_dim)
#         self._dynamics = mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], cfg.latent_dim, act=SimNorm(cfg))
#         self._reward = mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1))
#         self._pi = mlp(cfg.latent_dim, [cfg.mlp_dim, cfg.mlp_dim], 2*cfg.action_dim)
#         self._Qs = SimpleEnsemble([
#             mlp(cfg.latent_dim + cfg.action_dim, [cfg.mlp_dim, cfg.mlp_dim], max(cfg.num_bins, 1)) 
#             for _ in range(cfg.num_q)
#         ])
#         self.init_target_q()

#     def init_target_q(self):
#         self._target_Qs = deepcopy(self._Qs)
#         for p in self._target_Qs.parameters(): p.requires_grad = False

#     def soft_update_target_q(self):
#         with torch.no_grad():
#             for p, p_targ in zip(self._Qs.parameters(), self._target_Qs.parameters()):
#                 p_targ.data.mul_(1 - self.cfg.tau)
#                 p_targ.data.add_(self.cfg.tau * p.data)

#     def encode(self, obs): return self._encoder(obs)
#     def next(self, z, a): return self._dynamics(torch.cat([z, a], dim=-1))
#     def reward(self, z, a): return self._reward(torch.cat([z, a], dim=-1))
#     def pi(self, z):
#         mu, log_std = self._pi(z).chunk(2, dim=-1)
#         log_std = torch.clamp(log_std, self.cfg.log_std_min, self.cfg.log_std_max)
#         return mu, log_std
#     def Q(self, z, a, target=False):
#         x = torch.cat([z, a], dim=-1)
#         return self._target_Qs(x) if target else self._Qs(x)
    


# # Agent and Training Utils
# def two_hot_inv(x, cfg):
#     if cfg.num_bins == 0: return x
#     dreg_bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=x.device)
#     x = F.softmax(x, dim=-1)
#     return torch.sum(x * dreg_bins, dim=-1)

# def two_hot(x, cfg):
#     if cfg.num_bins == 0: return x
#     x = x.view(-1).clamp(cfg.vmin, cfg.vmax)
#     bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()
#     bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx).unsqueeze(-1)
#     soft_two_hot = torch.zeros(x.shape[0], cfg.num_bins, device=x.device)
#     soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
#     soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)
#     return soft_two_hot

# class TDMPC2(nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         self.device = torch.device(cfg.device)
#         self.model = WorldModel(cfg).to(self.device)
#         self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
        
#     def act(self, obs, eval_mode=False):
#         with torch.no_grad():
#             z = self.model.encode(obs.to(self.device))
#             mu, _ = self.model.pi(z)
#             if eval_mode: return mu.cpu()
#             return (mu + 0.1 * torch.randn_like(mu)).cpu().clamp(-1, 1)

#     def update(self, buffer):
#         batch = buffer.sample()
#         obs = batch['obs'].to(self.device)
#         action = batch['action'].to(self.device)
#         reward = batch['reward'].to(self.device)
        
#         z = self.model.encode(obs[0])
#         consistency_loss, reward_loss, value_loss = 0, 0, 0
        
#         for t in range(self.cfg.horizon):
#             z_next_pred = self.model.next(z, action[t])
#             reward_pred = self.model.reward(z, action[t])
#             qs_pred = self.model.Q(z, action[t])
            
#             with torch.no_grad():
#                 z_next_target = self.model.encode(obs[t+1])
#                 reward_target = two_hot(reward[t].view(-1), self.cfg)
#                 next_pi_action, _ = self.model.pi(z_next_target)
#                 target_qs = self.model.Q(z_next_target, next_pi_action, target=True)
#                 target_q_val = two_hot_inv(target_qs, self.cfg).min(0)[0]
#                 target_val = reward[t].view(-1) + self.cfg.rho * target_q_val.view(-1)
#                 target_val_dist = two_hot(target_val, self.cfg)

#             consistency_loss += F.mse_loss(z_next_pred, z_next_target) * (self.cfg.rho ** t)
#             reward_loss += -(reward_target * F.log_softmax(reward_pred, -1)).sum(-1).mean() * (self.cfg.rho ** t)
#             for q_pred in qs_pred:
#                 value_loss += -(target_val_dist * F.log_softmax(q_pred, -1)).sum(-1).mean() * (self.cfg.rho ** t)
#             z = z_next_pred

#         total_loss = (
#             self.cfg.consistency_coef * consistency_loss +
#             self.cfg.reward_coef * reward_loss +
#             self.cfg.value_coef * value_loss
#         )
        
#         self.optim.zero_grad()
#         total_loss.backward()
#         nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip_norm)
#         self.optim.step()
#         self.model.soft_update_target_q()
#         return {"loss": total_loss.item(), "rw_loss": reward_loss.item()}