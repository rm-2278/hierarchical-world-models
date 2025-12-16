from pathlib import Path

class Config:
    def __init__(self):
        # Task
        self.task = "Pendulum-v1"
        self.task_title = "Pendulum (Classic Control)"
        self.episodic = True
        self.seed = 1

        # Hyperparams
        self.steps = 5000
        # self.steps = 20000        
        self.batch_size = 256
        self.lr = 1e-3            
        self.grad_clip_norm = 10
        self.tau = 0.01
        self.rho = 0.5
        self.consistency_coef = 1
        self.reward_coef = 0.1
        self.value_coef = 0.1
        self.termination_coef = 0 
        self.entropy_coef = 1e-4

        # Architecture
        self.latent_dim = 32      
        self.mlp_dim = 64
        self.enc_dim = 64
        self.num_q = 2
        self.dropout = 0.0
        self.simnorm_dim = 4
        self.lora_rank = 0 

        # MPC
        self.mpc = True
        self.horizon = 10         
        self.iterations = 4
        self.num_samples = 256
        self.num_elites = 32
        self.num_pi_trajs = 16
        self.min_std = 0.1
        self.max_std = 1.0
        self.temperature = 0.5

        # Value
        self.log_std_min = -5
        self.log_std_max = 2
        self.num_bins = 51        
        self.vmin = -100
        self.vmax = 0             
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)

        # Logging & Saving
        self.eval_episodes = 5
        self.eval_freq = 1000
        self.buffer_size = 100000
        self.device = "cpu"
        
        # 保存先ディレクトリ
        self.root_dir = Path("result") 

        # Placeholders
        self.action_dim = None
        self.episode_length = None
        self.seed_steps = None
        self.obs_shape = None

    @property
    def run_name(self):
        lora_str = f"lora{self.lora_rank}" if self.lora_rank > 0 else "no_lora"
        return f"{self.task}_{lora_str}_seed{self.seed}"

    def get_log_dir(self):
        return self.root_dir / "logs" / self.run_name

    def get_model_dir(self):
        return self.root_dir / "models" / self.run_name
