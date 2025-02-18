import os
import pathlib
import shutil
from itertools import chain

import nox

PROJECT = "bmi-example-pynetlogo"
PACKAGE = "heat"
HERE = pathlib.Path(__file__)
ROOT = HERE.parent
PATHS = [PACKAGE, "examples", "tests", HERE.name]
PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13"]


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Run the tests."""
    session.install(".[testing]")

    args = [
        "--cov",
        PACKAGE,
        "-vvv",
    ] + session.posargs

    if "CI" in os.environ:
        args.append(f"--cov-report=xml:{ROOT.absolute()!s}/coverage.xml")
    session.run("pytest", *args)

    if "CI" not in os.environ:
        session.run("coverage", "report", "--ignore-errors", "--show-missing")


@nox.session(name="test-bmi", python=PYTHON_VERSIONS)
def test_bmi(session: nox.Session) -> None:
    """Test the Basic Model Interface."""
    session.install("bmi-tester")
    session.install(".")
    session.run(
        "bmi-test",
        "heat:BmiHeatDiffusion",
        "--config-file",
        f"{ROOT}/examples/config.yaml",
        "--root-dir",
        "examples",
        "-vvv",
    )


@nox.session(name="run-examples", python=PYTHON_VERSIONS)
def run_examples(session: nox.Session):
    """Run Python script examples."""
    session.install(".[examples]")
    session.cd("examples")
    session.run("python", "run-bmi-model.py")


@nox.session
def format(session: nox.Session) -> None:
    """Clean lint and assert style."""
    session.install(".[dev]")

    if session.posargs:
        black_args = session.posargs
    else:
        black_args = []

    session.run("black", *black_args, *PATHS)
    session.run("isort", *PATHS)
    session.run("ruff", "check", "--fix", *PATHS)


@nox.session
def release(session):
    """Tag and build a new version."""
    session.install(".[build]")
    session.run("fullrelease")


@nox.session(python=False)
def clean(session):
    """Remove virtual environments, build files, and caches."""
    shutil.rmtree("build", ignore_errors=True)
    shutil.rmtree("dist", ignore_errors=True)
    shutil.rmtree("docs/build", ignore_errors=True)
    shutil.rmtree(f"{PACKAGE}.egg-info", ignore_errors=True)
    shutil.rmtree(".pytest_cache", ignore_errors=True)
    shutil.rmtree(".venv", ignore_errors=True)
    if os.path.exists(".coverage"):
        os.remove(".coverage")
    for p in chain(ROOT.rglob("*.py[co]"), ROOT.rglob("__pycache__")):
        if p.is_dir():
            p.rmdir()
        else:
            p.unlink()


@nox.session(python=False)
def nuke(session):
    """Clean and also remove the .nox directory."""
    clean(session)
    shutil.rmtree(".nox", ignore_errors=True)
