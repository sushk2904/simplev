## 1. Introduction
Thank you for your interest in contributing to SimpleV. We are building a local-first, zero-configuration vector database, and we welcome contributions from developers of all experience levels. Whether you are fixing a typo, adding a new feature, or optimizing our array operations, your help is deeply appreciated.

## 2. Getting Started
To contribute to the codebase, you will need to set up a local development environment.

1. Fork the repository on GitHub and clone your fork locally.
2. Ensure you have Python 3.10 or newer installed.
3. Create and activate a virtual environment:
   `python -m venv venv`
   `source venv/bin/activate` (or `venv\Scripts\activate` on Windows)
4. Install the project in editable mode along with the development dependencies:
   `pip install -e ".[dev]"`

## 3. Development Workflow
We want to ensure that your time is well spent. For minor bug fixes and documentation updates, feel free to open a Pull Request directly. 

For larger features, architectural changes, or significant performance optimizations, please open an Issue first to discuss your proposed approach. This ensures your work aligns with the project roadmap and saves you from rewriting code.

## 4. Coding Standards
To maintain a clean and educational codebase, we enforce the following standards:
* **Formatting and Linting:** We use `black` for code formatting and `ruff` for linting. Please run `ruff check .` and `black .` before committing your changes.
* **Type Hints:** All function signatures must include Python type hints.
* **Documentation:** Public classes and methods should include Google-style docstrings. If your change affects the API or architecture, update the relevant Markdown documentation (like `API_SPEC.md` or `ARCHITECTURE.md`).
* **Architecture Decision Records:** If you are introducing a major technical change, you will be asked to fill out an ADR template to document the reasoning behind the decision.

## 5. Testing
We rely on `pytest` to maintain database integrity. 
* All Pull Requests must pass the existing test suite.
* If you add a new feature, you must include corresponding unit or integration tests.
* Run the test suite locally using `pytest` before submitting your PR.

## 6. Pull Request Process
Once your code is ready, push your branch to your fork and open a Pull Request against the main SimpleV repository. A maintainer will review your code, provide feedback if necessary, and merge it once it meets our quality and testing standards.