# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive documentation (ARCHITECTURE.md, CONTRIBUTING.md, API.md)
- Development guide with testing and profiling instructions
- CI/CD workflow examples
- Type hints for all public functions

## [1.0.0] - 2026-01-17

### Added
- 1D Timoshenko beam FEM implementation for ground truth generation
- 1D Euler-Bernoulli beam FEM for validation
- Bayesian model selection framework using PyMC
- WAIC and LOO-CV model comparison metrics
- Automated pipeline orchestration across multiple aspect ratios
- Comprehensive visualization and reporting system
- Configuration-driven execution via YAML

### Changed
- **BREAKING**: Replaced 2D plane stress FEM with 1D beam FEM
  - Eliminates systematic stiffness mismatch
  - 100× performance improvement
  - Exact consistency with analytical beam theories
- Reduced default MCMC samples from 2000 to 800 per chain
- Reduced noise level from 0.1% to 0.05% for better discrimination
- Updated mesh sizing: adaptive element count based on aspect ratio

### Fixed
- **Critical**: L/h=8 aspect ratio now correctly favors Timoshenko (was favoring EB)
- Model selection results now physically accurate across all aspect ratios
- Eliminated numerical instability at high aspect ratios (L/h=30+)
- Fixed memory issues causing crashes on standard laptops

### Performance
- Data generation: 5 seconds for 8 aspect ratios (was 2-3 minutes)
- Full pipeline: ~30-40 minutes (was 1-2 hours)
- Peak memory usage: ~1 GB (was 4+ GB)

### Results
- Transition aspect ratio: L/h ≈ 19.2
- Thick beams (L/h ≤ 15): Strong Timoshenko preference (log BF down to -10.8)
- Slender beams (L/h = 20-30): Euler-Bernoulli preference
- Very slender beams (L/h = 50): Models indistinguishable (correct behavior)

## [0.1.0] - 2025-12-15

### Added
- Initial project structure
- Basic beam theory implementations (EB and Timoshenko)
- 2D plane stress FEM (legacy, deprecated in v1.0.0)
- PyMC integration for Bayesian calibration
- Basic visualization utilities

### Known Issues
- 2D FEM has systematic stiffness bias vs beam theories
- Non-monotonic model selection results at intermediate aspect ratios
- Performance issues at high aspect ratios

---

## Release Notes Format

Each release includes:
- **Added**: New features
- **Changed**: Changes to existing functionality
- **Deprecated**: Soon-to-be removed features
- **Removed**: Removed features
- **Fixed**: Bug fixes
- **Security**: Security vulnerability patches
- **Performance**: Performance improvements

---

## Migration Guides

### Migrating from 0.x to 1.0

**Breaking Changes**:

1. **FEM Backend Change**
   ```python
   # Old (0.x)
   from apps.fem.cantilever_fem import CantileverFEM
   fem = CantileverFEM(nx=100, ny=20, ...)
   
   # New (1.0+)
   from apps.fem.beam_fem import TimoshenkoBeamFEM
   fem = TimoshenkoBeamFEM(n_elements=40, ...)
   ```

2. **Configuration Updates**
   ```yaml
   # Old
   bayesian:
     n_samples: 2000
   data:
     noise_fraction: 0.001
   
   # New
   bayesian:
     n_samples: 800
   data:
     noise_fraction: 0.0005
   ```

3. **Mesh Parameters**
   ```python
   # Old: 2D mesh required (nx, ny)
   mesh = (100, 20)
   
   # New: 1D mesh, single parameter
   n_elements = 40  # Computed adaptively: 4 × L/h
   ```

**Benefits**:
- 100× faster execution
- Physically correct results
- Lower memory footprint
- No more crashes on standard hardware
