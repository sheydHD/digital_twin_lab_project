# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security seriously. If you discover a security vulnerability, please follow these steps:

### 1. **Do Not** Open a Public Issue

Security vulnerabilities should not be disclosed publicly until a fix is available.

### 2. Report Privately

Send an email to: **security@example.com** (or contact maintainers directly)

Include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

### 3. Response Timeline

- **Initial Response**: Within 48 hours
- **Assessment**: Within 1 week
- **Fix Timeline**: Depends on severity
  - Critical: 1-7 days
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

### 4. Disclosure Process

Once a fix is available:
1. We'll prepare a security advisory
2. Release a patched version
3. Publish the advisory with credit to reporter
4. Notify users through GitHub Security Advisory

## Security Best Practices

### For Users

1. **Keep Dependencies Updated**
   ```bash
   pip install --upgrade digital-twin-lab
   ```

2. **Validate Configuration Files**
   - Don't use configuration from untrusted sources
   - Validate YAML files before loading

3. **Sanitize Input Data**
   - Verify CSV files are from trusted sources
   - Check for NaN or inf values

4. **Use Virtual Environments**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

### For Contributors

1. **No Hardcoded Secrets**
   - Never commit API keys, passwords, or tokens
   - Use environment variables or config files (in `.gitignore`)

2. **Input Validation**
   ```python
   def validate_input(value: float, min_val: float, max_val: float):
       if not (min_val <= value <= max_val):
           raise ValueError(f"Value {value} out of range [{min_val}, {max_val}]")
   ```

3. **Safe File Operations**
   ```python
   from pathlib import Path
   
   # Prevent directory traversal
   def safe_path(base_dir: Path, filename: str) -> Path:
       path = (base_dir / filename).resolve()
       if not path.is_relative_to(base_dir):
           raise ValueError("Path traversal attempt detected")
       return path
   ```

4. **Dependency Scanning**
   ```bash
   pip install safety
   safety check
   ```

## Known Security Considerations

### 1. Arbitrary Code Execution

**Risk**: Loading untrusted configuration files could execute arbitrary code

**Mitigation**: 
- YAML loader uses `safe_load()` (no code execution)
- Configuration schema validation enforced

### 2. Resource Exhaustion

**Risk**: Large MCMC sample counts could exhaust memory

**Mitigation**:
- Default limits on sample counts
- Configuration validation prevents unreasonable values
- Pipeline monitors memory usage

### 3. File System Access

**Risk**: Writing to arbitrary locations

**Mitigation**:
- All output paths validated
- No access to files outside project directory
- Path sanitization implemented

### 4. Dependency Vulnerabilities

**Risk**: Third-party packages may have vulnerabilities

**Mitigation**:
- Regular `pip audit` checks
- Dependabot alerts enabled
- Pin major versions, allow minor updates

## Secure Configuration Example

```yaml
# configs/secure_config.yaml

# Safe: All paths are relative to project root
output:
  base_dir: "outputs"  # Not "/tmp" or absolute paths
  
# Safe: Reasonable computational limits
bayesian:
  n_samples: 800      # Not 1000000
  n_chains: 2         # Not 100
  
# Safe: Validated numeric ranges
beam_parameters:
  length: 1.0         # Must be positive
  aspect_ratios: [5, 8, 10]  # Must be > 0
```

## Audit Log

| Date       | Issue | Severity | Status |
|------------|-------|----------|--------|
| 2026-01-17 | N/A   | N/A      | No vulnerabilities reported |

## Security Tools

We recommend using:

- **Bandit**: Python security linter
  ```bash
  pip install bandit
  bandit -r apps/
  ```

- **Safety**: Dependency vulnerability scanner
  ```bash
  pip install safety
  safety check
  ```

- **pip-audit**: Python package auditor
  ```bash
  pip install pip-audit
  pip-audit
  ```

## Contact

For security concerns, contact:
- **Email**: security@example.com
- **GitHub**: Open a private security advisory

## Acknowledgments

We thank the following security researchers:
- (None yet - be the first!)

---

Last updated: 2026-01-17
