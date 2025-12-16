from pathlib import Path
import gymnasium as gym
import numpy as np
import torch

from gymnasium import spaces

class SimpleReacherEnv(gym.Env):
    """Numpyだけで動く超軽量な2リンク・ロボットアーム環境"""
    def __init__(self, task_id=0):
        super().__init__()
        self.task_id = task_id
        self.link_lengths = [1.0, 1.0]
        self.dt = 0.05
        # ゴール位置定義 (タスクごとに変更)
        self.goals = [
            np.array([1.0, 1.0]),   # Task 0: 右上
            np.array([-1.0, -1.0]), # Task 1: 左下
            np.array([-1.0, 1.0]),  # Task 2: 左上
            np.array([1.0, -1.0])   # Task 3: 右下
        ]
        self.target = self.goals[task_id % len(self.goals)]
        
        # 観測: [cos, sin, vel, target] (8次元) に修正
        # ※GymWrapperでBoxのshapeを自動取得させるため、ここでは厳密な定義が必要
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.angles = np.random.uniform(-np.pi, np.pi, size=(2,))
        self.velocities = np.zeros(2)
        self._steps = 0
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action, -1, 1)
        accel = action * 2.0 - 0.1 * self.velocities 
        self.velocities += accel * self.dt
        self.angles += self.velocities * self.dt
        self.angles = (self.angles + np.pi) % (2 * np.pi) - np.pi
        self._steps += 1
        
        # Forward Kinematics
        tip_x = self.link_lengths[0] * np.cos(self.angles[0]) + \
                self.link_lengths[1] * np.cos(self.angles[0] + self.angles[1])
        tip_y = self.link_lengths[0] * np.sin(self.angles[0]) + \
                self.link_lengths[1] * np.sin(self.angles[0] + self.angles[1])
        tip_pos = np.array([tip_x, tip_y])
        
        dist = np.linalg.norm(tip_pos - self.target)
        reward = -dist
        truncated = self._steps >= 100
        terminated = False
        return self._get_obs(), reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([
            np.cos(self.angles), np.sin(self.angles),
            self.velocities, 
            self.target
        ]).astype(np.float32)

class GymWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=env.observation_space.shape, 
            dtype=np.float32
        )
    
    def _to_tensor(self, x):
        return torch.from_numpy(x).float()

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._to_tensor(obs), info

    def step(self, action):
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = action * 2.0
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        info['success'] = False
        return (
            self._to_tensor(obs), 
            torch.tensor(reward, dtype=torch.float32), 
            torch.tensor(float(done), dtype=torch.float32), 
            info
        )

    def rand_act(self):
        return torch.from_numpy(self.action_space.sample().astype(np.float32)) / 2.0

# def make_env(cfg):
#     env = gym.make(cfg.task)
#     env = GymWrapper(env)
#     cfg.action_dim = env.action_space.shape[0]
#     cfg.obs_shape = env.observation_space.shape[0]
#     cfg.episode_length = 200 
#     cfg.seed_steps = 1000
#     return env
def make_env(cfg):
    """Configのタスク名を見て環境を切り替える"""
    if cfg.task == "SimpleReacher":
        # Configに task_id がなければ 0 を使う
        task_id = getattr(cfg, 'task_id', 0)
        env = SimpleReacherEnv(task_id=task_id)
        print(f"[Env] Created SimpleReacherEnv for Task {task_id} (Target: {env.target})")
    else:
        env = gym.make(cfg.task)
    
    env = GymWrapper(env)
    
    cfg.action_dim = env.action_space.shape[0]
    cfg.obs_shape = env.observation_space.shape[0]
    cfg.episode_length = 200 # Reacherの場合は100でも良いがConfig優先でもOK
    cfg.seed_steps = 1000
    return env