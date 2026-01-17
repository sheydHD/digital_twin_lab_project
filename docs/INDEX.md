# Documentation Index

Welcome to the Digital Twin Bayesian Model Selection framework documentation! This index will help you find the right documentation for your needs.

## 🎯 I Want To...

### Get Started
- **Run the project quickly** → [README Quick Start](../README.md#quick-start)
- **Understand what this project does** → [EXPLANATION.md](../EXPLANATION.md)
- **See the results** → [README Key Results](../README.md#key-results)

### Understand the System
- **Learn the architecture** → [ARCHITECTURE.md](../ARCHITECTURE.md)
- **Understand design decisions** → [ARCHITECTURE.md - Design Decisions](../ARCHITECTURE.md#design-decisions--trade-offs)
- **See data flow** → [ARCHITECTURE.md - Data Flow](../ARCHITECTURE.md#data-flow-diagram)
- **Understand beam theories** → [README Key Concepts](../README.md#key-concepts)

### Use the API
- **API reference** → [API.md](API.md)
- **Code examples** → [API.md - Usage Examples](API.md)
- **Configuration options** → [ARCHITECTURE.md - Configuration Schema](../ARCHITECTURE.md#configuration-schema)

### Develop
- **Set up dev environment** → [DEVELOPMENT.md](DEVELOPMENT.md)
- **Run tests** → [DEVELOPMENT.md - Testing](DEVELOPMENT.md#testing)
- **Debug issues** → [DEVELOPMENT.md - Debugging](DEVELOPMENT.md#debugging-guide)
- **Optimize performance** → [DEVELOPMENT.md - Performance](DEVELOPMENT.md#performance-optimization)

### Contribute
- **Contributing guidelines** → [CONTRIBUTING.md](../CONTRIBUTING.md)
- **Code standards** → [CONTRIBUTING.md - Coding Standards](../CONTRIBUTING.md#coding-standards)
- **Pull request process** → [CONTRIBUTING.md - PR Process](../CONTRIBUTING.md#pull-request-process)
- **Report bugs** → [CONTRIBUTING.md - Issue Reporting](../CONTRIBUTING.md#issue-reporting)

### Stay Informed
- **What's new** → [CHANGELOG.md](../CHANGELOG.md)
- **Security policy** → [SECURITY.md](../SECURITY.md)
- **License** → [LICENSE](../LICENSE)

---

## 📚 Documentation Structure

```
digital_twin_lab_project/
├── README.md                    # Project overview and quick start
├── EXPLANATION.md               # Plain-English explanation
├── ARCHITECTURE.md              # System design and architecture
├── CONTRIBUTING.md              # Contribution guidelines
├── CHANGELOG.md                 # Version history
├── LICENSE                      # MIT License
├── SECURITY.md                  # Security policy
│
├── docs/
│   ├── INDEX.md                 # This file
│   ├── API.md                   # Complete API reference
│   └── DEVELOPMENT.md           # Development guide
│
└── configs/
    └── default_config.yaml      # Configuration reference
```

---

## 📖 Reading Paths by Persona

### For **End Users**

1. [README.md](../README.md) - Overview and installation
2. [EXPLANATION.md](../EXPLANATION.md) - What it does in simple terms
3. [README.md - Usage](../README.md#usage) - How to run it
4. [README.md - Expected Results](../README.md#expected-results) - What to expect

### For **Researchers**

1. [README.md - Key Results](../README.md#key-results) - Research findings
2. [ARCHITECTURE.md](../ARCHITECTURE.md) - Technical methodology
3. [README.md - Key Concepts](../README.md#key-concepts) - Theoretical background
4. [API.md](API.md) - Implementation details
5. [README.md - References](../README.md#references) - Citations

### For **Developers**

1. [CONTRIBUTING.md](../CONTRIBUTING.md) - Start here
2. [DEVELOPMENT.md](DEVELOPMENT.md) - Dev environment setup
3. [API.md](API.md) - Code reference
4. [ARCHITECTURE.md](../ARCHITECTURE.md) - System design
5. [CHANGELOG.md](../CHANGELOG.md) - Version history

### For **Contributors**

1. [CONTRIBUTING.md](../CONTRIBUTING.md) - Contribution workflow
2. [DEVELOPMENT.md](DEVELOPMENT.md) - Development practices
3. [ARCHITECTURE.md - Extensibility](../ARCHITECTURE.md#extensibility-points) - How to extend
4. [API.md](API.md) - API to follow
5. [SECURITY.md](../SECURITY.md) - Security guidelines

---

## 🔍 Find by Topic

### Beam Theory
- **Euler-Bernoulli**: [README - Key Concepts](../README.md#key-concepts), [API.md - Euler-Bernoulli](API.md#euler-bernoulli-beam)
- **Timoshenko**: [README - Key Concepts](../README.md#key-concepts), [API.md - Timoshenko](API.md#timoshenko-beam)
- **Comparison**: [ARCHITECTURE.md - Design Decisions](../ARCHITECTURE.md#design-decisions--trade-offs)

### Finite Element Method
- **1D Beam FEM**: [API.md - 1D Timoshenko FEM](API.md#1d-timoshenko-beam-fem)
- **Why 1D vs 2D**: [ARCHITECTURE.md - Design Decisions](../ARCHITECTURE.md#1-why-1d-fem-instead-of-2d-fem)
- **Element formulation**: [API.md - Element Formulation](API.md#1d-timoshenko-beam-fem)

### Bayesian Inference
- **Calibration**: [API.md - Bayesian Calibrator](API.md#bayesian-calibrator)
- **Model selection**: [API.md - Model Selector](API.md#model-selector)
- **WAIC/LOO**: [ARCHITECTURE.md - Model Selection](../ARCHITECTURE.md#4-bayesian-inference-layer-appsbayesian)
- **Bayes factors**: [README - Key Concepts](../README.md#bayesian-model-selection)

### Data Generation
- **Synthetic data**: [API.md - Synthetic Data Generator](API.md#synthetic-data-generator)
- **Noise models**: [ARCHITECTURE.md - Data Generation](../ARCHITECTURE.md#1-data-generation-layer-appsdata)
- **Sensor placement**: [API.md - Configuration](API.md#configuration)

### Configuration
- **Config schema**: [ARCHITECTURE.md - Configuration Schema](../ARCHITECTURE.md#configuration-schema)
- **Examples**: [README - Configuration](../README.md#configuration)
- **Validation**: [API.md - Config Loader](API.md#config-loader)

### Testing
- **Running tests**: [DEVELOPMENT.md - Testing](DEVELOPMENT.md#testing)
- **Writing tests**: [CONTRIBUTING.md - Testing Requirements](../CONTRIBUTING.md#testing-requirements)
- **Test structure**: [DEVELOPMENT.md - Test Structure](DEVELOPMENT.md#test-structure)

### Performance
- **Optimization**: [DEVELOPMENT.md - Performance](DEVELOPMENT.md#performance-optimization)
- **Profiling**: [DEVELOPMENT.md - Profiling](DEVELOPMENT.md#profiling-workflow)
- **Benchmarks**: [ARCHITECTURE.md - Performance](../ARCHITECTURE.md#performance-characteristics)

### Troubleshooting
- **Common issues**: [DEVELOPMENT.md - Debugging](DEVELOPMENT.md#debugging-guide)
- **Error messages**: [API.md - Error Handling](API.md#error-handling)
- **FAQ**: [EXPLANATION.md](../EXPLANATION.md)

---

## 🆕 Recently Updated

- **2026-01-17**: Complete documentation overhaul
  - Added ARCHITECTURE.md
  - Added CONTRIBUTING.md
  - Added API.md
  - Added DEVELOPMENT.md
  - Enhanced README.md

---

## 🔗 External Resources

### PyMC & Bayesian Inference
- [PyMC Documentation](https://www.pymc.io/)
- [PyMC Examples Gallery](https://www.pymc.io/projects/examples/en/latest/gallery.html)
- [ArviZ Documentation](https://python.arviz.org/)
- [Bayesian Data Analysis Book](http://www.stat.columbia.edu/~gelman/book/)

### Beam Theory
- [Timoshenko Beam Theory (Wikipedia)](https://en.wikipedia.org/wiki/Timoshenko%E2%80%93Ehrenfest_beam_theory)
- [Euler-Bernoulli Beam Theory (Wikipedia)](https://en.wikipedia.org/wiki/Euler%E2%80%93Bernoulli_beam_theory)

### Finite Element Method
- [Introduction to FEM](https://www.colorado.edu/engineering/CAS/courses.d/IFEM.d/)
- [FEniCS Project](https://fenicsproject.org/) (advanced FEM)

### Python Best Practices
- [The Hitchhiker's Guide to Python](https://docs.python-guide.org/)
- [Python Packaging Guide](https://packaging.python.org/)
- [Real Python Tutorials](https://realpython.com/)

---

## 📝 Document Conventions

### Code Examples
- ✅ All code examples are tested and working
- ✅ Examples include expected output where relevant
- ✅ Type hints included for clarity

### Diagrams
- 📊 ASCII diagrams for simple flows
- 📈 Mermaid diagrams (future: convert to images)
- 🎨 Architecture diagrams in ARCHITECTURE.md

### Navigation
- 🔗 All documents cross-reference each other
- ⬆️ "Back to top" links in long documents
- 📋 Table of contents in all major documents

---

## 🤝 Contributing to Documentation

Documentation is code! To contribute:

1. **Found a typo?** → Open a PR with the fix
2. **Unclear explanation?** → Open an issue with questions
3. **Missing documentation?** → See [CONTRIBUTING.md](../CONTRIBUTING.md#documentation)
4. **Want to add examples?** → PRs welcome!

### Documentation Style Guide

- **Be concise**: Get to the point quickly
- **Use examples**: Show, don't just tell
- **Stay current**: Update docs with code changes
- **Link liberally**: Cross-reference related docs
- **Format consistently**: Follow existing patterns

---

## 📧 Need Help?

- **Questions**: [GitHub Discussions](https://github.com/sheydHD/digital_twin_lab_project/discussions)
- **Bugs**: [GitHub Issues](https://github.com/sheydHD/digital_twin_lab_project/issues)
- **Security**: See [SECURITY.md](../SECURITY.md)

---

<div align="center">

**Happy Reading!** 📖

[⬆ Back to Top](#documentation-index)

</div>
