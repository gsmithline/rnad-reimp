import threading
from queue import Queue, Full
import torch as th
from trainers.impala.storage import ActorRollout
from trainers.impala.config import ImpalaConfig

class ActorWorker(threading.Thread):
    def __init__(self, actor_id, env_fn, policy_fn, param_server, 
                 queue: Queue, config: ImpalaConfig):
        super().__init__(daemon=True)
        self.actor_id = actor_id
        self.env = env_fn()
        self.policy = policy_fn().to(config.actor_device)
        self.param_server = param_server
        self.queue = queue
        self.config = config
        self.shutdown = threading.Event()
        self.steps_since_refresh = 0
        self.env_steps = 0
    
    def run(self):
        obs = self.env.reset()
        obs = th.as_tensor(obs, device=self.config.actor_device).unsqueeze(0)

        while not self.shutdown.is_set():
            # if self.steps_since_refresh >= self.config.refresh_interval:
            version, weights = self.param_server.get()
            self.policy.load_state_dict(weights, strict=False)
            self.steps_since_refresh = 0
            rollout = self.collect_rollout(obs)
            try:
                self.queue.put(rollout, timeout=1.0)
            except Full:
                continue
            obs = rollout.observations[-1]
            self.steps_since_refresh += rollout.env_steps
            self.env_steps += rollout.env_steps
    
    def collect_rollout(self, obs0):
        obs_list = [obs0]
        actions, rewards, dones, beh_logps = [], [], [], []
        extras = {}

        for _ in range(self.config.unroll_length):
            with th.no_grad():
                action_dist, value, extra = self.policy(obs_list[-1])

            action = action_dist.sample()
            beh_logp = action_dist.log_prob(action)
            if beh_logp.ndim > 0:
                beh_logp = beh_logp.sum(dim=-1)
            
            next_obs, reward, done, info = self.env.step(action.cpu().numpy())
            next_obs = th.as_tensor(next_obs, device=self.config.actor_device).unsqueeze(0)
            obs_list.append(next_obs)
            actions.append(action)
            rewards.append(th.tensor(reward, device=self.config.actor_device))
            dones.append(th.tensor(done, device=self.config.actor_device))
            beh_logps.append(beh_logp)
            for k, v in extra.items():
                extras.setdefault(k, []).append(v)
            if done:
                next_obs = th.as_tensor(self.env.reset(), device=self.config.actor_device).unsqueeze(0)
                obs_list[-1] = next_obs
        to_tensor = lambda xs: th.stack(xs, dim=0)
        extras = {k: th.stack(v, dim=0) for k, v in extras.items()}
        return ActorRollout(
            observations=th.stack(obs_list, dim=0),
            actions=to_tensor(actions),
            rewards=to_tensor(rewards),
            dones=to_tensor(dones),
            behavior_log_probs=to_tensor(beh_logps),
            actor_id=self.actor_id,
            env_steps=self.config.unroll_length,
            extras=extras,
        )
    


            














                
        










