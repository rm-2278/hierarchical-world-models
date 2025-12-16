from collections import defaultdict
import numpy as np
import torch
from tdmpc2lora_tmp.utils import Logger, Buffer

class OnlineTrainer:
    def __init__(self, cfg, env, agent):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.buffer = Buffer(cfg)
        self.logger = Logger(cfg)
        self.best_reward = -float('inf')
    
    def eval(self):
        rewards = []
        for _ in range(self.cfg.eval_episodes):
            obs, _ = self.env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = self.agent.act(obs, eval_mode=True)
                obs, reward, done_tensor, _ = self.env.step(action)
                done = bool(done_tensor.item())
                ep_reward += reward.item()
            rewards.append(ep_reward)
        return np.mean(rewards)

    def train(self):
        print(f"Start training on {self.cfg.task} (Mac Local / CPU)")
        print(f"Check results in: {self.cfg.root_dir}")
        
        step = 0
        while step < self.cfg.steps:
            obs, _ = self.env.reset()
            done = False
            episode_data = defaultdict(list)
            
            while not done:
                if step < self.cfg.seed_steps: action = self.env.rand_act()
                else: action = self.agent.act(obs)
                
                next_obs, reward, done_tensor, info = self.env.step(action)
                done = bool(done_tensor.item())
                
                episode_data['obs'].append(obs)
                episode_data['action'].append(action)
                episode_data['reward'].append(reward)
                
                obs = next_obs
                step += 1
                
                if step >= self.cfg.seed_steps and len(self.buffer.storage) > 0:
                    train_info = self.agent.update(self.buffer)
                    if step % 1000 == 0:
                        print(f"Step: {step}, Loss: {train_info['loss']:.3f}")
                        self.logger.log({"step": step, "loss": train_info['loss']}, category="train")

                if step % self.cfg.eval_freq == 0:
                    avg_reward = self.eval()
                    print(f"Eval at step {step}: Reward {avg_reward:.1f}")
                    self.logger.log({"step": step, "episode_reward": avg_reward}, category="eval")
                    
                    if avg_reward > self.best_reward:
                        self.best_reward = avg_reward
                        self.logger.save_agent(self.agent, step, is_best=True)
                    else:
                        self.logger.save_agent(self.agent, step, is_best=False)

            td = {k: torch.stack(v) for k, v in episode_data.items()}
            self.buffer.add(td)