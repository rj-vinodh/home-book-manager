# AGENTS.md

This repository must be validated using Docker-based test commands.

## Required test runner

Use the test image defined in `Dockerfile.test`.

## Build the test image

```bash
docker build -f Dockerfile.test -t book-extract-only-test .
```

## Run the test suite

```bash
docker run --rm book-extract-only-test
```

## Run a specific test file

```bash
docker run --rm book-extract-only-test pytest -q test_app.py
```

## Rules

- Never use the system Python to run tests or app code for this project.
- Always use Docker for validation in this repo.
- If you need to inspect or modify the working tree, prefer the repository files mounted inside the container context rather than the host Python environment.
- Keep the app and tests executing in the same Dockerized environment used by the project.
