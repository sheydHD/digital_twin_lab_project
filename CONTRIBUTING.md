# Contributing to Digital Twin Bayesian Model Selection

Thank you for your interest in contributing! This document provides guidelines and best practices for contributing to this project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing Requirements](#testing-requirements)
- [Pull Request Process](#pull-request-process)
- [Documentation](#documentation)
- [Issue Reporting](#issue-reporting)

---

## Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inclusive experience for everyone. We expect all contributors to:

- Be respectful and considerate in communication
- Accept constructive criticism gracefully
- Focus on what is best for the community
- Show empathy towards other community members

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Publishing others' private information without permission
- Trolling, insulting, or derogatory comments
- Any other conduct that would be inappropriate in a professional setting

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of Bayesian inference and finite element methods

### Setting Up Your Development Environment

1. **Fork the repository**
   ```bash
   # Fork on GitHub, then clone your fork
   git clone https://github.com/YOUR_USERNAME/digital_twin_lab_project.git
   cd digital_twin_lab_project
   ```

2. **Add upstream remote**
   ```bash
   git remote add upstream https://github.com/sheydHD/digital_twin_lab_project.git
   ```

3. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

4. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

5. **Verify installation**
   ```bash
   make test
   ```

---

## Development Workflow

### Branch Strategy

We follow a simplified Git Flow model:

- `main`: Production-ready code, always stable
- `develop`: Integration branch for features (if exists)
- `feature/*`: New features or enhancements
- `bugfix/*`: Bug fixes
- `hotfix/*`: Urgent production fixes
- `docs/*`: Documentation updates

### Creating a Feature Branch

```bash
# Sync with upstream
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name
```

### Making Changes

1. **Write your code**
   - Follow coding standards (see below)
   - Add tests for new functionality
   - Update documentation as needed

2. **Commit your changes**
   ```bash
   git add .
   git commit -m "feat: add new beam theory implementation
   
   - Implemented Reddy third-order shear deformation theory
   - Added unit tests for deflection calculations
   - Updated documentation with usage examples
   
   Closes #42"
   ```

3. **Keep your branch updated**
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

4. **Push to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

### Commit Message Format

We follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <subject>

<body>

<footer>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only changes
- `style`: Code style changes (formatting, missing semicolons, etc.)
- `refactor`: Code refactoring without changing functionality
- `perf`: Performance improvements
- `test`: Adding or updating tests
- `chore`: Maintenance tasks, dependency updates

**Examples**:
```bash
feat(fem): add 3D beam element support

Implemented 3D Timoshenko beam elements with full 6-DOF nodes.
Includes torsional and axial stiffness contributions.

Closes #123

---

fix(calibration): correct WAIC computation for multivariate models

The previous implementation didn't account for correlation between
parameters. Now using proper multivariate normal likelihood.

Fixes #145

---

docs(readme): update installation instructions for Python 3.12

Added notes about compatibility issues with PyMC on Python 3.12
```

---

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 100 characters (not 79)
- **Quotes**: Prefer double quotes for strings
- **Imports**: Organized using `isort`
- **Formatting**: Automated with `black`
- **Type hints**: Required for all public functions

### Code Formatting

**Before committing**, run:

```bash
# Auto-format code
make format

# Check for issues
make lint
```

### Example: Well-Formatted Function

```python
from typing import Optional, Tuple

import numpy as np


def compute_deflection(
    x: float,
    force: float,
    length: float,
    elastic_modulus: float,
    moment_of_inertia: float,
    shear_modulus: Optional[float] = None,
) -> Tuple[float, float]:
    """Compute beam deflection and rotation at a given position.
    
    Uses Euler-Bernoulli theory if shear_modulus is None, otherwise
    uses Timoshenko beam theory with shear correction factor κ = 5/6.
    
    Args:
        x: Position along beam length [m]. Must be 0 ≤ x ≤ length.
        force: Applied tip load [N]. Positive downward.
        length: Total beam length [m]. Must be positive.
        elastic_modulus: Young's modulus [Pa]. Must be positive.
        moment_of_inertia: Second moment of area [m⁴]. Must be positive.
        shear_modulus: Shear modulus [Pa]. If None, uses Euler-Bernoulli.
    
    Returns:
        deflection: Vertical displacement [m]. Positive downward.
        rotation: Cross-section rotation [rad]. Positive clockwise.
    
    Raises:
        ValueError: If input parameters are outside valid ranges.
    
    Examples:
        >>> compute_deflection(1.0, 1000.0, 1.0, 210e9, 8.33e-6)
        (0.0002, 0.0003)
    
    References:
        Timoshenko, S.P. (1921). "On the correction for shear of the 
        differential equation for transverse vibrations of prismatic bars."
    """
    # Input validation
    if not (0 <= x <= length):
        raise ValueError(f"Position x={x} must be in [0, {length}]")
    
    if force <= 0:
        raise ValueError(f"Force must be positive, got {force}")
    
    # Bending deflection (always present)
    deflection_bending = (force * x**2) / (6 * elastic_modulus * moment_of_inertia) * (
        3 * length - x
    )
    
    # Shear deflection (Timoshenko only)
    if shear_modulus is not None:
        area = moment_of_inertia / (length**2 / 12)  # Simplified for rectangular section
        kappa = 5 / 6  # Shear correction factor
        deflection_shear = (force * x) / (kappa * shear_modulus * area)
        deflection = deflection_bending + deflection_shear
    else:
        deflection = deflection_bending
    
    # Rotation (slope of deflection curve)
    rotation = (force * x) / (2 * elastic_modulus * moment_of_inertia) * (2 * length - x)
    
    return deflection, rotation
```

### Naming Conventions

| Type | Convention | Example |
|------|-----------|---------|
| Variables | `snake_case` | `elastic_modulus` |
| Functions | `snake_case` | `compute_deflection()` |
| Classes | `PascalCase` | `TimoshenkoBeam` |
| Constants | `UPPER_SNAKE_CASE` | `MAX_ITERATIONS` |
| Private | `_leading_underscore` | `_validate_input()` |
| Modules | `snake_case` | `beam_fem.py` |

### Docstring Format

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings):

```python
def function_name(arg1: Type1, arg2: Type2) -> ReturnType:
    """Short one-line summary.
    
    Longer description explaining the function's purpose,
    algorithm, and any important implementation details.
    
    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.
    
    Returns:
        Description of return value.
    
    Raises:
        ValueError: When input is invalid.
        RuntimeError: When computation fails.
    
    Examples:
        >>> function_name(value1, value2)
        expected_result
    """
```

---

## Testing Requirements

### Test Coverage

- **Minimum coverage**: 80% for new code
- **Required tests**: All public functions must have unit tests
- **Edge cases**: Test boundary conditions and error handling

### Running Tests

```bash
# Run all tests
make test

# Run with coverage report
make test-cov

# Run specific test file
pytest tests/test_beam_models.py -v

# Run specific test function
pytest tests/test_beam_models.py::test_euler_bernoulli_deflection -v
```

### Writing Tests

Use `pytest` for all tests:

```python
import pytest
import numpy as np
from apps.models.euler_bernoulli import EulerBernoulliBeam


class TestEulerBernoulliBeam:
    """Test suite for Euler-Bernoulli beam model."""
    
    @pytest.fixture
    def beam(self):
        """Create a standard test beam."""
        return EulerBernoulliBeam(
            length=1.0,
            height=0.1,
            width=0.05,
            elastic_modulus=210e9,
            poisson_ratio=0.3,
        )
    
    def test_tip_deflection_analytical(self, beam):
        """Verify tip deflection against analytical solution."""
        force = 1000.0
        expected_deflection = 0.000196  # From handbook
        
        actual_deflection = beam.deflection(x=1.0, P=force)
        
        np.testing.assert_allclose(
            actual_deflection, expected_deflection, rtol=1e-3
        )
    
    def test_zero_force_gives_zero_deflection(self, beam):
        """Edge case: Zero force should produce zero deflection."""
        deflection = beam.deflection(x=0.5, P=0.0)
        assert deflection == 0.0
    
    @pytest.mark.parametrize("x", [-0.1, 1.5, 2.0])
    def test_invalid_position_raises_error(self, beam, x):
        """Test that invalid positions raise ValueError."""
        with pytest.raises(ValueError, match="Position.*out of range"):
            beam.deflection(x=x, P=1000.0)
```

### Test Organization

```
tests/
├── __init__.py
├── test_beam_models.py       # Analytical beam theory tests
├── test_fem.py                # FEM solver tests
├── test_calibration.py        # Bayesian inference tests
├── test_model_selection.py    # Model comparison tests
├── test_integration.py        # End-to-end tests
└── fixtures/                  # Shared test data
    ├── sample_data.csv
    └── reference_solutions.json
```

---

## Pull Request Process

### Before Submitting

1. ✅ **All tests pass**: `make test`
2. ✅ **Code is formatted**: `make format`
3. ✅ **No linting errors**: `make lint`
4. ✅ **Documentation updated**: README, docstrings, etc.
5. ✅ **CHANGELOG updated** (if applicable)
6. ✅ **Branch is up to date with main**

### Submitting a Pull Request

1. **Push your branch to your fork**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create PR on GitHub**
   - Go to the original repository
   - Click "New Pull Request"
   - Select your branch
   - Fill out the PR template

3. **PR Title Format**
   ```
   [Type] Brief description
   
   Examples:
   [Feature] Add Reddy beam theory support
   [Bugfix] Fix WAIC computation for edge cases
   [Docs] Improve installation instructions
   ```

4. **PR Description Template**
   ```markdown
   ## Description
   Brief summary of changes and motivation.
   
   ## Changes Made
   - Bullet list of specific changes
   - Include file names and key modifications
   
   ## Testing
   - [ ] Unit tests added/updated
   - [ ] Integration tests pass
   - [ ] Manual testing performed
   
   ## Documentation
   - [ ] Docstrings updated
   - [ ] README updated (if needed)
   - [ ] CHANGELOG updated
   
   ## Related Issues
   Closes #123
   Related to #456
   
   ## Screenshots (if applicable)
   [Include plots or output screenshots]
   ```

### Code Review Process

1. **Automated checks**: CI/CD runs tests, linting, coverage
2. **Peer review**: At least one maintainer review required
3. **Address feedback**: Make requested changes
4. **Approval**: Once approved, PR will be merged

### Review Criteria

Reviewers will check:
- ✅ Code follows style guide
- ✅ Tests are comprehensive
- ✅ Documentation is clear
- ✅ No unnecessary complexity
- ✅ Backward compatibility maintained
- ✅ Performance impact acceptable

---

## Documentation

### Types of Documentation

1. **Code documentation**: Docstrings, inline comments
2. **User documentation**: README, tutorials, examples
3. **API documentation**: Auto-generated from docstrings
4. **Architecture documentation**: ARCHITECTURE.md

### Updating Documentation

When making changes, update:

- **README.md**: If public API changes
- **ARCHITECTURE.md**: If design changes
- **Docstrings**: Always for new functions
- **Examples**: If usage patterns change
- **CHANGELOG.md**: For all user-facing changes

### Writing Good Documentation

**Do**:
- ✅ Use clear, concise language
- ✅ Include code examples
- ✅ Explain *why*, not just *what*
- ✅ Keep examples up to date
- ✅ Use diagrams for complex concepts

**Don't**:
- ❌ Assume prior knowledge
- ❌ Use jargon without explanation
- ❌ Leave outdated information
- ❌ Omit edge cases

---

## Issue Reporting

### Before Creating an Issue

1. **Search existing issues**: Your issue may already be reported
2. **Check documentation**: Issue might be a misunderstanding
3. **Verify reproducibility**: Can you reproduce it consistently?

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug.

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What should happen?

## Actual Behavior
What actually happens?

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python version: [e.g., 3.10.8]
- Package versions: [run `pip freeze`]

## Additional Context
- Error messages
- Screenshots
- Relevant configuration

## Minimal Reproducible Example
```python
# Minimal code to reproduce the issue
```
```

### Feature Request Template

```markdown
## Feature Description
Clear description of the proposed feature.

## Motivation
Why is this feature needed? What problem does it solve?

## Proposed Solution
How should this feature work?

## Alternatives Considered
What other approaches did you consider?

## Additional Context
- Use cases
- Example code
- References
```

---

## Development Tips

### Debugging PyMC Models

```python
# Use test values to debug before sampling
with pm.Model() as model:
    E = pm.Normal("E", mu=210e9, sigma=10e9)
    
    # Set test value
    E.tag.test_value = 210e9
    
    # Check likelihood evaluation
    logp = model.logp()
    print(f"Log probability: {logp}")
```

### Profiling Performance

```python
# Profile function execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumtime")
stats.print_stats(10)  # Top 10 slowest functions
```

### Common Pitfalls

1. **MCMC not converging**: Increase tuning steps, check priors
2. **Memory issues**: Reduce sample count, process in batches
3. **Numerical instability**: Normalize inputs, check condition numbers
4. **Test flakiness**: Use fixed random seeds

---

## Community

### Getting Help

- **GitHub Issues**: Technical questions, bugs
- **Discussions**: General questions, ideas
- **Email**: [maintainer@example.com] for private matters

### Recognition

Contributors will be:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Credited in published papers (for significant contributions)

---

## License

By contributing, you agree that your contributions will be licensed under the same MIT License that covers this project.

---

## Questions?

If you have questions about contributing, feel free to:
- Open a discussion on GitHub
- Contact the maintainers
- Check existing documentation

Thank you for contributing to making this project better! 🎉
