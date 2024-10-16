import nox


@nox.session
def lint(session):
    """
    Run flake8 to lint the codebase.

    Args:
        session: The Nox session object.
    """
    session.install("flake8")
    session.run("flake8", "src/", "tests/")


@nox.session
def format(session):
    """
    Run black to format the codebase.

    Args:
        session: The Nox session object.
    """
    session.install("black")
    session.run("black", "src/", "tests/")


@nox.session(python=["3.8", "3.9", "3.10", "3.11", "3.12"])
def tests(session):
    """
    Run the test suite using pytest across multiple Python versions.

    Args:
        session: The Nox session object.
    """
    session.install("pytest", "pytest-cov")
    session.run("poetry", "install", external=True)
    session.run("pytest", "--cov", "src", "--cov-report", "term-missing")
