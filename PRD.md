# Product Requirements Document: SimpleV

## 1. Introduction and Vision
SimpleV is an open-source, local-first vector database. Our vision is to build the "SQLite for Vector Search." We want to provide developers, researchers, and students with a tool that is extremely simple to use on the surface, while maintaining a robust and educational internal architecture. We are not aiming to compete with enterprise-scale distributed databases; rather, we hope to perfect the local developer experience.

## 2. Problem Statement
Currently, many vector databases require significant operational overhead. Developers often have to configure Docker containers, manage cloud accounts, or handle complex embedding pipelines just to get started. When someone simply wants to build a local Retrieval-Augmented Generation (RAG) prototype or a personal knowledge base, this infrastructure becomes a barrier. SimpleV aims to remove this friction by offering sensible defaults and a lightweight, zero-configuration setup.

## 3. Target Audience
This project is built with the following users in mind:
* AI and ML Engineers looking for quick local prototyping.
* Students and Researchers studying database internals or vector algorithms.
* Developers integrating RAG systems into local applications.
* Local LLM users building personal knowledge tools.
* Open-source contributors seeking an approachable, systems-level project.

## 4. Core Philosophy
* Local-first: Data remains firmly on the user's machine.
* Zero configuration: Ready to use immediately after a simple installation.
* Python-first: An intuitive and native developer experience.
* Educational architecture: Clean, well-documented code that demonstrates real database concepts.
* Powerful internals: Relying on strong engineering principles (like HNSW indexing and custom file formats) abstracted behind a simple API.

## 5. Functional Requirements (Iterative Roadmap)

### Phase v0.1: Core Foundation
* Implement in-memory vector storage.
* Provide basic Create, Read, Update, Delete (CRUD) operations via `insert`, `search`, and `delete` methods.
* Support exact nearest neighbor search (flat indexing).

### Phase v0.2: Data Persistence
* Introduce the `.sv` custom binary file format for local storage.
* Enable disk persistence to save and load state.
* Add metadata storage alongside vectors, including basic pre-filtering capabilities.

### Phase v0.3: Document Ingestion
* Build automatic document ingestion and parsing for common file types (PDF, DOCX, TXT, Markdown, code).
* Implement sensible text chunking utilities to prepare long documents for embedding.

### Phase v0.4: Performance and Tooling
* Integrate Approximate Nearest Neighbor (ANN) search via HNSW to handle larger datasets efficiently.
* Develop a Command Line Interface (CLI) for simple database management outside of Python scripts.
* Establish a standardized benchmarking suite to track and ensure performance goals.

## 6. Non-Functional Requirements
* Usability: The system must operate completely locally without external network dependencies for its core functionality (aside from the initial download of embedding models).
* Architecture Layering: The codebase must be strictly separated into functional layers (Storage Engine, Indexing Engine, Query Engine, Python SDK) to allow for future integration of performance-critical modules (e.g., Rust/PyO3).
* Educational Value: The codebase must prioritize readability, with thorough documentation and Architecture Decision Records (ADRs) explaining the "why" behind technical choices.

## 7. Success Criteria
The core project will be considered successful when a user can execute `pip install simplev`, initialize a local database, insert unstructured text documents, and successfully perform a semantic search—all within a single Python script, without configuring Docker, cloud services, or manual embedding functions.