An open-source reimplementation of the Regularised Nash Dynamics (R-NaD) training procedure that powered DeepMind's *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning*.

## Why R-NaD?
- **Game-theoretic foundation**: R-NaD augments regret-minimisation dynamics with entropy regularisation, letting large policy networks converge toward Nash equilibria in imperfect-information games without search.
- **Scalable value learning**: The agent predicts distributions over returns rather than point estimates, enabling stable learning across the enormous Stratego game tree.
- **Research reproducibility**: Our goal is to turn the high-level algorithmic description from the paper into reproducible, well-tested PyTorch building blocks.

## Current Scope
- `utils/loss.py`: Histogram Loss with Gaussian targets (`HLGaussLoss`), the classification-based value loss used in R-NaD.
- Additional components (policy parameterisation, fictitious self-play loop, Stratego environment harness) are under active development.

## Getting Started
### Prerequisites
- Python 3.12+
- [UV](https://github.com/astral-sh/uv) or `pip`

### Installation
```bash
uv sync  # or: pip install -e .
```

### Quick Example
```python
import torch
from utils.loss import HLGaussLoss

loss_fn = HLGaussLoss(min_value=-1.0, max_value=1.0, num_bins=51)
logits = torch.randn(8, 51)          # predicted value distribution
targets = torch.rand(8) * 2 - 1      # sampled returns in [-1, 1]
loss = loss_fn(logits, targets)
loss.backward()
```

## Repository Layout
- `main.py`: CLI entry point (placeholder while the training harness is incubated).
- `utils/`: General-purpose utilities, loss functions, and exploratory notebooks.
- `pyproject.toml`: Project metadata and dependency list.

## Roadmap
- Expand value-head coverage tests and add Torch unit tests.
- Implement the R-NaD policy update (regularised Nash dynamics with entropy terms).
- Integrate Stratego/Barrage environment bindings and self-play orchestration.
- Reproduce key metrics from the paper (exploitability, Gravon ELO) and publish evaluation scripts.

## References
- Julien Pérolat et al., *Mastering the Game of Stratego with Model-Free Multiagent Reinforcement Learning*, 2022. [arXiv:2206.15378](https://arxiv.org/abs/2206.15378)

