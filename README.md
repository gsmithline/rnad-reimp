# rnad-reimp

Minimal IMPALA skeleton (actor threads + learner + parameter server + V-trace).

## Quick smoke test (CartPole)

Install deps (example):

```bash
pip install torch gymnasium
```

Run:

```bash
python impala_runner.py --env-id CartPole-v1 --learner-steps 2000
```

Notes:
- The runner uses a tiny discrete-action MLP policy in `trainers/impala/simple_policy.py` for testing.
- Swap `policy_fn()` in `impala_runner.py` with your RNAD policy once wired up.
