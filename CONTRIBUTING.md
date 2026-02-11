# contributing

## setup

```bash
git clone https://github.com/YOUR_USERNAME/digital_twin_lab_project.git
cd digital_twin_lab_project
git remote add upstream https://github.com/sheydHD/digital_twin_lab_project.git
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
make test
```

## workflow

```bash
# sync with upstream
git checkout main
git pull upstream main

# create branch
git checkout -b feature/your-feature

# make changes, then:
make format
make lint
make test

# commit
git add .
git commit -m "feat: short description"

# push and open PR
git push origin feature/your-feature
```

### branch naming

- `feature/*` — new features
- `bugfix/*` — bug fixes
- `docs/*` — documentation updates

### commit messages

Follow [conventional commits](https://www.conventionalcommits.org/):

```
feat(scope): add something
fix(scope): correct something
docs: update readme
test: add edge case tests
refactor: simplify calibration logic
```

## coding standards

- follow PEP 8, line length 100
- format with `black`, sort imports with `isort`
- type hints on all public functions
- Google-style docstrings

```python
def compute_deflection(
    x: float,
    force: float,
    length: float,
    elastic_modulus: float,
) -> float:
    """Compute beam deflection at position x.

    Args:
        x: position along beam [m]
        force: applied tip load [N]
        length: beam length [m]
        elastic_modulus: Young's modulus [Pa]

    Returns:
        vertical deflection [m]
    """
```

### naming

| type | convention | example |
|------|-----------|---------|
| variables | snake_case | `elastic_modulus` |
| functions | snake_case | `compute_deflection()` |
| classes | PascalCase | `TimoshenkoBeam` |
| constants | UPPER_SNAKE_CASE | `MAX_ITERATIONS` |
| private | _leading_underscore | `_validate_input()` |

## testing

All public functions need unit tests. Use pytest.

```bash
make test          # all tests
make test-cov      # with coverage
pytest tests/test_beam_models.py -v  # specific file
```

```python
class TestEulerBernoulliBeam:
    @pytest.fixture
    def beam(self):
        return EulerBernoulliBeam(length=1.0, height=0.1, width=0.05,
                                  elastic_modulus=210e9, poisson_ratio=0.3)

    def test_tip_deflection(self, beam):
        actual = beam.deflection(x=1.0, P=1000.0)
        expected = 0.000196
        np.testing.assert_allclose(actual, expected, rtol=1e-3)
```

## pull requests

Before submitting:
1. all tests pass (`make test`)
2. code is formatted (`make format`)
3. no lint errors (`make lint`)
4. documentation updated if needed

PR title format: `[Feature] add beam theory` or `[Bugfix] fix convergence check`.

## reporting issues

Search existing issues first. Include:
- steps to reproduce
- expected vs actual behavior
- Python version and OS
- minimal reproducible example

## license

Contributions are licensed under the same MIT license as the project.
