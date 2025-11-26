from dataclasses import dataclass

@dataclass
class ImpalaConfig:
    unroll_length: int = 80
    batch_size: int = 4 #how many rollouts per learner step
    discount: float = .99 #v-trace value function discount 
    clip_rho_threshold: float = 1.0
    clip_c_threshold: float = 1.0
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    grad_clip: float = 40.0
    actor_queue_size: int = 64
    num_actors: int = 8
    refresh_interval: int = 50_000  #env steps before we weight learner vs actor 
    learner_device: str = "cuda"
    actor_device: str = "cpu"
    checkpoint_interval: int = 10_000


    
