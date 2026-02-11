# security policy

## supported versions

| version | supported |
|---------|-----------|
| 1.0.x   | yes       |
| < 1.0   | no        |

## reporting a vulnerability

Do not open a public issue. Send an email to the maintainers directly.

Include:
- description of the vulnerability
- steps to reproduce
- potential impact
- suggested fix (if any)

### response timeline

- initial response: within 48 hours
- assessment: within 1 week
- fix timeline: depends on severity (critical: 1-7 days, high: 1-2 weeks, medium: 2-4 weeks)

## security considerations

### arbitrary code execution

YAML loader uses `safe_load()` — no code execution from config files. Configuration schema is validated before use.

### resource exhaustion

Default limits on MCMC sample counts prevent excessive memory use. Configuration validation rejects unreasonable values.

### file system access

All output paths are validated. No access to files outside the project directory.

### dependencies

Run `pip-audit` or `safety check` periodically to scan for known vulnerabilities in third-party packages.

## secure configuration

```yaml
output:
  base_dir: "outputs"        # relative paths only

bayesian:
  n_samples: 800             # reasonable limits
  n_chains: 2

beam_parameters:
  length: 1.0                # must be positive
  aspect_ratios: [5, 8, 10]  # must be > 0
```

## security tools

```bash
pip install bandit && bandit -r apps/
pip install pip-audit && pip-audit
```

## contact

- email: maintainers (see AUTHORS.md)
- GitHub: open a private security advisory
