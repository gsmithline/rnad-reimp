import argparse
import os
from queue import Queue
import torch as th

from trainers.impala.actor import ActorWorker
from trainers.impala.config import ImpalaConfig
from trainers.impala.learner import Learner
from trainers.impala.parameter_server import ParameterServer


def build_env_fn(env_id: str):
    def _env_fn():
        import gym

        return gym.make(env_id)

    return _env_fn


def build_policy_fn():
    """
    TODO: replace this factory with your RNAD policy constructor.
    It must return an nn.Module implementing:
      - __call__(obs) -> (action_dist, value, extra)
      - forward_batch(observations)
      - log_prob(actions, logits)
      - value(observations)
      - entropy(logits)
    """
    raise NotImplementedError("Swap in your RNAD policy factory here.")


class PrintLogger:
    def log(self, data):
        print(data)


def main():
    parser = argparse.ArgumentParser(description="IMPALA runner")
    parser.add_argument("--env-id", type=str, default="CartPole-v1")
    parser.add_argument("--learner-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    config = ImpalaConfig()
    param_server = ParameterServer()
    queue = Queue(maxsize=config.actor_queue_size)

    env_fn = build_env_fn(args.env_id)
    policy_fn = build_policy_fn

    # Seed parameter server with initial learner weights
    learner_policy = policy_fn()
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
