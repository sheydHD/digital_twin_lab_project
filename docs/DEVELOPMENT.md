# Development Guide

## Quick Start for Developers

This guide will get you up and running with the development environment in under 5 minutes.

### Prerequisites

```bash
python --version  # Must be ≥ 3.10
git --version
```

### Setup

```bash
# Clone and navigate
git clone https://github.com/sheydHD/digital_twin_lab_project.git
cd digital_twin_lab_project

# Create environment and install
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

# Verify setup
make test
```

---

## Development Environment

### IDE Setup

#### VS Code (Recommended)

Install extensions:
```bash
code --install-extension ms-python.python
code --install-extension ms-python.vscode-pylance
code --install-extension ms-python.black-formatter
```

Workspace settings (`.vscode/settings.json`):
```json
{
    "python.defaultInterpreterPath": ".venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.formatting.provider": "black",
    "editor.formatOnSave": true,
    "editor.rulers": [100],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```

#### PyCharm

1. Open project
2. File → Settings → Project → Python Interpreter
3. Add interpreter → Virtual Environment → Existing environment
4. Select `.venv/bin/python`
5. Enable pytest: Settings → Tools → Python Integrated Tools → Testing → pytest

### Environment Variables

Create `.env` file (optional):
```bash
# Development settings
DEBUG=1
LOG_LEVEL=DEBUG

# Computational settings
PYMC_NUM_CHAINS=2
PYMC_CORES=2

# Output settings
OUTPUT_DIR=outputs
SAVE_TRACES=1
```

Load in code:
```python
from dotenv import load_dotenv
load_dotenv()
```

---

## Common Development Tasks

### Running the Pipeline

```bash
# Full pipeline with all stages
make run

# Individual stages
python main.py --stage data          # Data generation only
python main.py --stage calibration   # Bayesian calibration only
python main.py --stage analysis      # Model selection only
python main.py --stage report        # Reporting only

# Specific aspect ratios
python main.py -a 5 -a 8 -a 10 --stage all

# Verbose output
python main.py --verbose --debug
```

### Testing

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_beam_models.py -v

# Specific test function
pytest tests/test_beam_models.py::test_deflection -v

# With debugging
pytest tests/test_beam_models.py -v -s --pdb

# Parallel execution
pytest -n auto

# Watch mode (re-run on changes)
ptw
```

### Code Quality

```bash
# Format code
make format
# Or: black apps/ tests/

# Check formatting (no changes)
black --check apps/ tests/

# Lint code
make lint
# Or: pylint apps/ tests/

# Type checking
mypy apps/

# Check import order
isort --check-only apps/ tests/

# Fix import order
isort apps/ tests/
```

### Documentation

```bash
# Generate API docs
cd docs
make html

# View docs locally
python -m http.server 8000 --directory docs/_build/html

# Check for broken links
make linkcheck
```

### Profiling

```bash
# Profile entire pipeline
python -m cProfile -o profile.stats main.py --stage all
python -c "import pstats; p = pstats.Stats('profile.stats'); p.sort_stats('cumtime'); p.print_stats(20)"

# Profile with line-by-line detail
kernprof -l -v main.py

# Memory profiling
python -m memory_profiler main.py
```

---

## Project Structure Deep Dive

### Module Organization

```
apps/
├── models/              # Beam theory implementations
│   ├── base_beam.py    # Abstract base class (interface)
│   ├── euler_bernoulli.py
│   └── timoshenko.py
│
├── fem/                # Finite element solvers
│   ├── beam_fem.py     # 1D Timoshenko FEM (active)
│   └── cantilever_fem.py  # 2D FEM (legacy)
│
├── data/               # Data generation pipeline
│   └── synthetic_generator.py
│
├── bayesian/           # Probabilistic inference
│   ├── calibration.py  # PyMC model definitions
│   └── model_selection.py  # Model comparison
│
├── analysis/           # Post-processing
│   ├── visualization.py
│   └── reporter.py
│
├── pipeline/           # Orchestration
│   └── orchestrator.py
│
└── utils/              # Shared utilities
    ├── config.py
    └── logging_setup.py
```

### Key Design Patterns

#### 1. Strategy Pattern (Beam Models)

```python
# Abstract base
class BeamModel(ABC):
    @abstractmethod
    def deflection(self, x, P, L, E, I):
        pass

# Concrete implementations
class EulerBernoulliBeam(BeamModel):
    def deflection(self, x, P, L, E, I):
        return (P * x**2 / (6*E*I)) * (3*L - x)

# Usage - interchangeable
model: BeamModel = EulerBernoulliBeam(...)  # or TimoshenkoBeam(...)
result = model.deflection(x, P, L, E, I)
```

#### 2. Builder Pattern (Pipeline Orchestrator)

```python
orchestrator = (
    PipelineOrchestrator(config)
    .with_data_generation()
    .with_calibration()
    .with_model_selection()
    .with_reporting()
    .build()
)
orchestrator.run()
```

#### 3. Factory Pattern (FEM Solvers)

```python
def create_fem_solver(solver_type: str, params: dict):
    if solver_type == "timoshenko_1d":
        return TimoshenkoBeamFEM(**params)
    elif solver_type == "euler_bernoulli_1d":
        return EulerBernoulliFEM(**params)
    else:
        raise ValueError(f"Unknown solver: {solver_type}")
```

---

## Debugging Guide

### Common Issues

#### 1. PyMC Sampling Fails

**Symptoms**:
```
SamplingError: Initial evaluation of model at starting point failed!
```

**Solutions**:
```python
# Check test value evaluation
with pm.Model() as model:
    # Define model...
    
    # Test likelihood before sampling
    test_point = model.initial_point()
    logp = model.logp(test_point)
    print(f"Log probability at test point: {logp}")
    
    # If -inf, check your likelihood function
```

#### 2. Memory Issues

**Symptoms**:
```
MemoryError: Unable to allocate array
```

**Solutions**:
```python
# Reduce memory footprint
config["bayesian"]["n_samples"] = 500  # Instead of 2000
config["bayesian"]["n_chains"] = 2     # Instead of 4

# Process aspect ratios one at a time
for aspect_ratio in aspect_ratios:
    run_pipeline(aspect_ratio)
    gc.collect()  # Force garbage collection
```

#### 3. Numerical Instability

**Symptoms**:
```
RuntimeWarning: divide by zero encountered
RuntimeWarning: invalid value encountered in double_scalars
```

**Solutions**:
```python
# Add numerical guards
def safe_divide(num, denom, eps=1e-10):
    return num / (denom + eps)

# Normalize inputs
E_normalized = (E - E_mean) / E_std
```

#### 4. Slow MCMC Convergence

**Symptoms**:
- R-hat > 1.01
- Low effective sample size
- High divergences

**Solutions**:
```python
# Increase tuning steps
n_tune = 2000  # Instead of 500

# Adjust target acceptance
target_accept = 0.99  # Instead of 0.95

# Use better initialization
with pm.Model() as model:
    # Define priors...
    
    # Initialize near mode
    start = pm.find_MAP()
    trace = pm.sample(start=start)
```

### Debugging Tools

#### Interactive Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use built-in (Python 3.7+)
breakpoint()

# IPython debugger (better UI)
from IPython import embed; embed()
```

#### Logging

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# In your code
logger.debug(f"Input parameters: {params}")
logger.info("Starting calibration")
logger.warning("High divergence rate detected")
logger.error("Sampling failed", exc_info=True)
```

---

## Performance Optimization

### Profiling Workflow

1. **Identify bottleneck**
   ```bash
   python -m cProfile -o profile.stats main.py
   python -m pstats profile.stats
   >>> sort cumtime
   >>> stats 20
   ```

2. **Line-level profiling**
   ```python
   # Add decorator to function
   from line_profiler import profile
   
   @profile
   def expensive_function():
       # Your code
   ```
   
   ```bash
   kernprof -l -v script.py
   ```

3. **Memory profiling**
   ```bash
   python -m memory_profiler script.py
   ```

### Optimization Strategies

#### 1. Vectorization

```python
# Slow: Python loop
results = []
for x in x_values:
    results.append(compute_deflection(x))

# Fast: NumPy vectorization
results = np.vectorize(compute_deflection)(x_values)

# Faster: Direct array operations
results = (P * x_values**2 / (6*E*I)) * (3*L - x_values)
```

#### 2. Caching

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_computation(param):
    # Cached based on param value
    return result
```

#### 3. Parallel Processing

```python
from multiprocessing import Pool

def process_aspect_ratio(Lh):
    # Process one aspect ratio
    return results

# Parallel execution
with Pool(processes=4) as pool:
    results = pool.map(process_aspect_ratio, aspect_ratios)
```

#### 4. PyMC Optimization

```python
with pm.Model() as model:
    # Use JAX backend for faster autodiff
    import pymc as pm
    pm.set_backend("jax")
    
    # Precompile likelihood
    @pm.as_op(itypes=[pt.dvector], otypes=[pt.dscalar])
    def likelihood(params):
        return compute_log_likelihood(params)
```

---

## Testing Best Practices

### Test Structure

```python
# Use descriptive test names
def test_euler_bernoulli_tip_deflection_matches_analytical_solution():
    """
    Test that EB beam tip deflection matches the classical solution
    δ = PL³/(3EI) for a cantilever with point load.
    """
    # Arrange
    beam = create_test_beam()
    P, L, E, I = 1000.0, 1.0, 210e9, 8.33e-6
    
    # Act
    actual = beam.deflection(x=L, P=P)
    
    # Assert
    expected = (P * L**3) / (3 * E * I)
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
```

### Test Categories

```python
# Unit tests: Fast, isolated
@pytest.mark.unit
def test_single_function():
    pass

# Integration tests: Multiple components
@pytest.mark.integration
def test_full_pipeline_stage():
    pass

# Slow tests: Expensive computations
@pytest.mark.slow
def test_mcmc_convergence():
    pass

# Run specific categories
# pytest -m "unit"
# pytest -m "not slow"
```

### Fixtures

```python
@pytest.fixture(scope="module")
def test_beam():
    """Create beam once per test module."""
    return EulerBernoulliBeam(L=1.0, h=0.1, b=0.05, E=210e9, nu=0.3)

@pytest.fixture
def mock_data():
    """Generate synthetic data for testing."""
    return np.random.randn(100)

def test_with_fixtures(test_beam, mock_data):
    result = test_beam.fit(mock_data)
    assert result.converged
```

### Parametric Tests

```python
@pytest.mark.parametrize("Lh,expected_model", [
    (5.0, "Timoshenko"),
    (8.0, "Timoshenko"),
    (10.0, "Timoshenko"),
    (20.0, "Euler-Bernoulli"),
    (50.0, "Euler-Bernoulli"),
])
def test_model_selection_by_aspect_ratio(Lh, expected_model):
    result = run_model_selection(Lh)
    assert result.recommended_model == expected_model
```

---

## Release Process

### Version Numbering

We use [Semantic Versioning](https://semver.org/):

- **Major** (X.0.0): Breaking API changes
- **Minor** (0.X.0): New features, backward compatible
- **Patch** (0.0.X): Bug fixes

### Release Checklist

- [ ] All tests pass (`make test`)
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
- [ ] Version bumped in `pyproject.toml`
- [ ] Git tag created (`git tag v1.2.3`)
- [ ] Tagged commit pushed (`git push --tags`)
- [ ] GitHub release created
- [ ] Package published to PyPI (if applicable)

### Creating a Release

```bash
# Update version
bumpversion minor  # or major, patch

# Update CHANGELOG.md
# ... manually edit ...

# Commit and tag
git add .
git commit -m "Bump version to 1.2.0"
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin main --tags

# Create GitHub release
gh release create v1.2.0 --notes "Release notes here"
```

---

## CI/CD Pipeline

### GitHub Actions Workflow

```yaml
# .github/workflows/ci.yml
name: CI

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ['3.10', '3.11', '3.12']
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -e ".[dev]"
    
    - name: Run tests
      run: |
        make test-cov
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
```

---

## Additional Resources

### Learning Resources

- **Bayesian Inference**: [Statistical Rethinking](https://xcelab.net/rm/statistical-rethinking/) by Richard McElreath
- **PyMC**: [Official Tutorials](https://www.pymc.io/projects/examples/en/latest/gallery.html)
- **FEM**: [Introduction to FEM](https://onlinelibrary.wiley.com/doi/book/10.1002/9781119426462) by J.N. Reddy
- **Python Best Practices**: [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)

### Useful Commands

```bash
# Find TODOs in code
grep -r "TODO" apps/

# Count lines of code
cloc apps/ tests/

# Find large files
find . -type f -size +1M

# Clean up cache files
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete

# Check security vulnerabilities
pip install safety
safety check

# Update dependencies
pip list --outdated
pip install --upgrade package_name
```

---

## Getting Help

- **Documentation**: Start with README.md and ARCHITECTURE.md
- **Examples**: Check `examples/` directory
- **Issues**: Search existing issues on GitHub
- **Discussions**: GitHub Discussions for questions
- **Code**: Read the tests for usage examples

---

## Tips for New Contributors

1. **Start small**: Begin with documentation or tests
2. **Ask questions**: No question is too basic
3. **Read existing code**: Learn patterns from the codebase
4. **Run tests frequently**: Catch issues early
5. **Commit often**: Small, logical commits
6. **Write tests first**: TDD helps design better APIs
7. **Profile before optimizing**: Don't guess, measure

Happy coding! 🚀
