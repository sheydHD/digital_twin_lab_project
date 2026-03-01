# documentation

Documentation for the Bayesian beam theory model selection framework.

## contents

| document | description |
|----------|-------------|
| [getting started](getting-started.md) | installation and first run |
| [architecture](architecture.md) | system design, components, data flow |
| [technical spec](technical-spec.md) | full design rationale, formulas, and sequence diagrams |
| [API reference](api-reference.md) | class and function documentation |
| [development guide](development-guide.md) | dev environment, testing, debugging |
| [parameters](parameters.md) | parameter tables, priors, sign conventions |
| [bayesian glossary](bayesian-glossary.md) | glossary of all statistical methods used |
| [presentation guide](PRESENTATION_GUIDE.md) | slide-by-slide presentation notes |

## root-level docs

| document | description |
|----------|-------------|
| [README](../README.md) | project overview and quick start |
| [contributing](../CONTRIBUTING.md) | contribution guidelines |
| [changelog](../CHANGELOG.md) | version history |
| [authors](../AUTHORS.md) | developers and acknowledgments |
| [license](../LICENSE) | MIT license |

## Structure

```
docs/
  README.md               this file
  getting-started.md      installation, first run, troubleshooting
  architecture.md         system design, Mermaid diagrams, design decisions
  technical-spec.md       full design rationale, LaTeX formulas, sequence diagrams
  api-reference.md        every public function and class
  parameters.md           parameter tables, priors, sign conventions, study grid
  bayesian-glossary.md    all 22 statistical concepts with code locations
  development-guide.md    testing, linting, debugging, extension patterns
  PRESENTATION_GUIDE.md   slide-by-slide presentation notes
  internal/               internal development notes
```
