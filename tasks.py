"""Invoke tasks module."""

import sys
from invoke import Context, task


def run(c: Context, cmd: str) -> None:
    """Run command with platform-aware PTY (Linux/macOS only)."""
    use_pty = sys.platform != "win32"
    c.run(cmd, pty=use_pty)


@task
def check(c: Context) -> None:
    """Format code, then run static checks: ruff, black, mypy."""
    print("Formatting code with black...")
    run(c, "black src tests")
    print("Formatting imports and lint fixes with ruff...")
    run(c, "ruff check src tests --fix")
    print("Linting code with ruff...")
    run(c, "ruff check src tests")
    print("Checking types with mypy...")
    run(c, "mypy src tests")
    print("Checking code quality with pylint...")
    run(c, "pylint src tests")
    print("Checking logical errors and doc styling with flake8...")
    run(c, "flake8 --ignore=E501 src tests")
    print("Checking doc styling based on PEP 257 with pydocstyle...")
    run(c, "pydocstyle src tests")
    print("All checks passed!")


@task
def test(c: Context) -> None:
    """Run tests with pytest."""
    print("Running tests...")
    run(c, "pytest tests")


@task
def lint_fix(c: Context) -> None:
    """Apply automatic formatting and fixes."""
    print("Applying automatic formatting with black...")
    run(c, "black src tests")
    print("Applying automatic import sorting and fixes with ruff...")
    run(c, "ruff check src tests --fix")
    print("Formatting applied successfully!")
