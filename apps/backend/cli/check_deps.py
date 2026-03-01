#!/usr/bin/env python3
"""Dependency doctor — verifies every runtime package can be imported.

Called by ``make check-deps``.  Exit-code 0 means all good; 1 means at
least one package is missing, and the output tells you exactly which one
and how to install it.
"""

from __future__ import annotations

import importlib
import sys

# Mapping:  import-name  →  pip-install-name
PACKAGES: dict[str, str] = {
    "numpy": "numpy",
    "scipy": "scipy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "pymc": "pymc",
    "arviz": "arviz",
    "pytensor": "pytensor",
    "yaml": "pyyaml",
    "rich": "rich",
    "click": "click",
    "h5py": "h5py",
    "optuna": "optuna",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
}


def main() -> int:
    ok = True
    for mod, pip_name in PACKAGES.items():
        try:
            m = importlib.import_module(mod)
            version = getattr(m, "__version__", "?")
            print(f"  \033[32m✓\033[0m {pip_name:20s} {version}")
        except ImportError:
            print(f"  \033[31m✗\033[0m {pip_name:20s} MISSING  →  pip install {pip_name}")
            ok = False

    print()
    if not ok:
        print("\033[33mFix: run  make install  to install everything at once.\033[0m")
        return 1

    print("\033[32mAll dependencies present ✓\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
