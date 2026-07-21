## 1. Overview
To maintain a high-quality, educational, and accessible codebase, SimpleV strictly adheres to standard Python community conventions. Consistency is critical for an open-source database to ensure that developers can easily read, review, and contribute to the internal mechanics without friction.

## 2. Python Guidelines
* **Version:** The project targets Python 3.10 and above to leverage modern language features and type hinting syntax.
* **Formatting:** We use `black` as our uncompromising code formatter, adhering to its standard line length of 88 characters.
* **Linting:** We rely on `ruff` for lightning-fast linting. All code must pass `ruff` checks before merging to catch unused imports, manage cyclomatic complexity, and enforce best practices.
* **Type Hints:** Static typing is mandatory. Every function and method signature must include accurate Python type hints to assist with IDE autocompletion and static analysis.

## 3. Naming Conventions
* **Classes:** Use `PascalCase` (e.g., `StorageEngine`, `VectorIndex`).
* **Methods and Variables:** Use `snake_case` (e.g., `insert_document`, `query_vector`).
* **Private Members:** Any internal function, method, or variable not intended for the public API must begin with a single leading underscore (e.g., `_flush_to_disk`).

## 4. Documentation Requirements
* **Docstrings:** All public classes, methods, and functions must be documented using Google-style docstrings. This ensures consistent, readable explanations of arguments and return types.
* **Inline Comments:** While code should be as self-documenting as possible, complex mathematical operations within the indexing engine (such as exact distance calculations or graph traversals) must include inline comments explaining the reasoning behind the specific vector arithmetic.