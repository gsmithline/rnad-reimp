import argparse
import os
from queue import Queue
import torch as th

from trainers.impala.actor import ActorWorker
from trainers.impala.config import ImpalaConfig
from trainers.impala.learner import Learner
from trainers.impala.parameter_server import ParameterServer
import gymnasium as gym
from trainers.impala.simple_policy import SimpleCategoricalMLP


def build_env_fn(env_id: str):
    def _env_fn():
        return gym.make(env_id)
    return _env_fn
def policy_fn(obs_dim, num_actions):
    return SimpleCategoricalMLP(obs_dim=obs_dim, num_actions=num_actions)

def build_policy_fn():
    """
    have some rnad builder here
    """
    raise NotImplementedError("Policy factory is constructed in main() from env spaces.")


class PrintLogger:
    def log(self, data):
        print(data)


def main():
    parser = argparse.ArgumentParser(description="IMPALA runner")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--learner-steps", type=int, default=10_000)
    parser.add_argument("--learner-device", type=str, default=None, help='e.g. "cpu", "cuda:0", or "auto"')
    parser.add_argument("--actor-device", type=str, default=None, help='e.g. "cpu" (recommended)')
    parser.add_argument("--num-actors", type=int, default=None)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    config = ImpalaConfig()
    if args.learner_device is not None:
        config.learner_device = args.learner_device
        config.__post_init__()
    if args.actor_device is not None:
        config.actor_device = args.actor_device
    if args.num_actors is not None:
        config.num_actors = args.num_actors

    if th.cuda.device_count() > 1 and str(config.learner_device).startswith("cuda"):
        print(
            f"[impala_runner] Detected {th.cuda.device_count()} CUDA devices. "
            "This runner currently uses a single learner process/device. "
            'You can pin with --learner-device "cuda:N". True multi-GPU (DDP) is not implemented here.'
        )

    param_server = ParameterServer()
    queue = Queue(maxsize=config.actor_queue_size)

    env_fn = build_env_fn(args.env_id)
    
    probe_env = env_fn()
    obs_space = probe_env.observation_space
    act_space = probe_env.action_space
    probe_env.close()
     



    obs_dim = int(obs_space.shape[0])
    num_actions = int(act_space.n)



    learner_policy = policy_fn(obs_dim=obs_dim, num_actions=num_actions)
    param_server.update(learner_policy.state_dict())

    logger = PrintLogger()
    learner = Learner(learner_policy, config, param_server, queue, logger)

    actors = [
        ActorWorker(
            actor_id=i,
            env_fn=env_fn,
            policy_fn=policy_fn,
            param_server=param_server,
            queue=queue,
            config=config,
        )
        for i in range(config.num_actors)
    ]
    for actor in actors:
        actor.start()

    try:
        for _ in range(args.learner_steps):
            learner.step()
            if args.checkpoint_path and learner.learner_steps % config.checkpoint_interval == 0:
                os.makedirs(args.checkpoint_path, exist_ok=True)
                ckpt_file = os.path.join(args.checkpoint_path, f"impala_{learner.learner_steps}.pt")
                learner_state = {
                    "step": learner.learner_steps,
                    "model": learner.policy.state_dict(),
                    "optimizer": learner.optimizer.state_dict(),
                }
                th.save(learner_state, ckpt_file)
    except KeyboardInterrupt:
        pass
    finally:
        for actor in actors:
            actor.shutdown.set()
        for actor in actors:
            actor.join(timeout=1.0)


if __name__ == "__main__":
    main()
