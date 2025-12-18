import torch
import numpy as np
from pathlib import Path
import os
from argparse import ArgumentParser

from tdmpc2lora_tmp.config import Config
from tdmpc2lora_tmp.train_env import make_env
from tdmpc2lora_tmp.model import TDMPC2
from tdmpc2lora_tmp.trainer import OnlineTrainer

# --- 実験設定 ---
NUM_TASKS = 4
LORA_RANK = 4   # ※ここを 0 にすると自動的に "LoRAなしフルFT" として動作します
# TOTAL_STEPS = 30000 

def run_training_step(task_id, prev_model_path=None, run_suffix="", freeze_base=False, total_steps=30000, device="cpu"):
    """
    Args:
        freeze_base (bool): Trueの場合、LoRA以外のパラメータを固定する。
                            ただし、LORA_RANK=0 の場合は強制的にFalseとして扱う。
    """
    task_name = f"Task{task_id}"
    print(f"\n{'='*10} Training {task_name} {run_suffix} {'='*10}")
    
    cfg = Config()
    cfg.task = "VariableReacher" 
    cfg.task_id = task_id
    cfg.num_tasks = NUM_TASKS
    cfg.lora_rank = LORA_RANK
    cfg.steps = total_steps
    cfg.device = device
    
    # ログ用の名前
    mode_str = "FreezeBase" if (freeze_base and LORA_RANK > 0) else "JointTrain"
    base_name = f"{cfg.task}_Task{task_id}_rank{LORA_RANK}_{mode_str}{run_suffix}"
    cfg.__class__.run_name = property(lambda self: base_name)
    
    # 環境・エージェント構築
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    
    # --- 1. モデルロード ---
    if prev_model_path:
        print(f"Loading weights from: {prev_model_path}")
        ckpt = torch.load(prev_model_path, map_location=cfg.device, weights_only=False)
        agent.model.load_state_dict(ckpt["model"], strict=False)
    else:
        print("No previous model loaded. Training from scratch.")

    # --- 2. フリーズ処理の分岐 ---
    # LoRAが無効(rank=0)なら、freeze_base指定があっても無視して全学習する
    if cfg.lora_rank == 0:
        if freeze_base:
            print("Warning: LORA_RANK=0, so 'freeze_base' is ignored. Running Full Fine-Tuning.")
        freeze_base = False 

    if freeze_base:
        print(">>> [Policy] Freezing Base parameters. Training LoRA ONLY.")
        frozen_cnt = 0
        active_cnt = 0
        for name, param in agent.model.named_parameters():
            # "lora_" (LoRA重み) と "task_emb" (タスクID埋め込み) は学習
            # ※ task_emb もタスク固有なので学習対象に含めるのが一般的です
            if "lora_" in name or "task_emb" in name:
                param.requires_grad = True
                active_cnt += 1
            else:
                param.requires_grad = False
                frozen_cnt += 1
        print(f" -> Frozen (Base): {frozen_cnt}, Active (LoRA): {active_cnt}")
        
        # 【重要】パラメータのrequires_gradを変更したら、オプティマイザを再作成する
        #  既存の agent.optim は全パラメータを持っているので破棄して作り直す
        agent.optim = torch.optim.Adam(
            filter(lambda p: p.requires_grad, agent.model.parameters()), 
            lr=cfg.lr
        )
        print(">>> Optimizer re-initialized for active parameters only.")
        
    else:
        print(">>> [Policy] Training ALL parameters (Base + LoRA/Full).")
        # デフォルトで全パラメータ requires_grad=True なので、そのままでOK
        # agent.optim も __init__ で作られたものがそのまま使える

    # 学習開始
    trainer = OnlineTrainer(cfg, env, agent)
    trainer.train()
    
    return cfg.get_model_dir() / "best.pth"

# (evaluate_task 関数は変更なしのため省略)
def evaluate_task(task_id, model_path):
    # ... (元のコードと同じ) ...
    print(f"\n>>> Evaluating Task {task_id} using model: {model_path}")
    cfg = Config()
    cfg.task = "VariableReacher"
    cfg.task_id = task_id
    cfg.num_tasks = NUM_TASKS
    cfg.lora_rank = LORA_RANK
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    env = make_env(cfg)
    agent = TDMPC2(cfg)
    ckpt = torch.load(model_path, map_location=cfg.device, weights_only=False)
    agent.model.load_state_dict(ckpt["model"], strict=False)
    agent.eval()
    rewards = []
    for _ in range(10):
        obs, _ = env.reset()
        done = False
        ep_reward = 0
        while not done:
            action = agent.act(obs, eval_mode=True, task_idx=task_id)
            obs, reward, done_tensor, info = env.step(action)
            done = bool(done_tensor.item())
            ep_reward += reward.item()
        rewards.append(ep_reward)
    avg_reward = np.mean(rewards)
    print(f"Result Task {task_id}: Average Reward = {avg_reward:.2f}")
    return avg_reward

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--lora_rank", type=int, default=4, help="LoRA rank (0 for no LoRA)")
    parser.add_argument("--total_steps", type=int, default=30000, help="Total training steps per task")
    parser.add_argument("--cuda", type=int, default=0, help="CUDA device ID (-1 for CPU)")
    device = "cpu" if parser.parse_args().cuda < 0 else f"cuda:{parser.parse_args().cuda}"
    torch.cuda.set_device(device)

    args = parser.parse_args()
    LORA_RANK = args.lora_rank
    TOTAL_STEPS = args.total_steps
    print(f"Starting Experiment (Rank={LORA_RANK}) on {device}")
    
    # === Phase 1: 共通表現の獲得 (Joint Training / Meta-Training) ===
    # 過去タスクと行き来しながら、BaseとLoRAを同時に育てる
    # freeze_base=False
    
    path_A = run_training_step(0, prev_model_path=None,   run_suffix="_init", freeze_base=False, total_steps=TOTAL_STEPS, device=device)
    path_B1 = run_training_step(1, prev_model_path=path_A, run_suffix="_1st",  freeze_base=False, total_steps=TOTAL_STEPS, device=device)
    path_C1 = run_training_step(2, prev_model_path=path_B1, run_suffix="_1st",  freeze_base=False, total_steps=TOTAL_STEPS, device=device)
    path_B2 = run_training_step(1, prev_model_path=path_C1, run_suffix="_2nd",  freeze_base=False, total_steps=TOTAL_STEPS, device=device)
    path_C2 = run_training_step(2, prev_model_path=path_B2, run_suffix="_2nd",  freeze_base=False, total_steps=TOTAL_STEPS, device=device)
    
    # === Phase 2: 新規タスクへの適応 (LoRA Adaptation) ===
    # 獲得した共通表現(Base)は固定し、Task D のLoRAのみを学習する
    # freeze_base=True (LoRAなしの場合は無視されてFull FTになる)
    
    path_D = run_training_step(3, prev_model_path=path_C2, run_suffix="_final", freeze_base=True, total_steps=int(TOTAL_STEPS/10), device=device)
    
    # === 評価 ===
    print("\n" + "="*30)
    print("ALL TRAINING FINISHED. STARTING EVALUATION.")
    print("="*30)
    
    score_A = evaluate_task(0, path_D)
    score_B = evaluate_task(1, path_D)
    score_C = evaluate_task(2, path_D)
    score_D = evaluate_task(3, path_D)
    
    print(f"\nFinal Check (Rank={LORA_RANK}):")
    print(f"Task A (Past/Preservation): {score_A:.1f}")
    print(f"Task B (Past/Preservation): {score_B:.1f}")
    print(f"Task C (Past/Preservation): {score_C:.1f}")
    print(f"Task D (New/Adaptation)   : {score_D:.1f}")