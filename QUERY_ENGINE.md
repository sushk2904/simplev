## 1. Overview
The Query Engine functions as the central orchestrator of the SimpleV database. It does not store data, generate embeddings, or calculate mathematical distances itself. Instead, it manages the complex pipeline required to turn a user's natural language query into a formatted list of relevant document results. By keeping this coordination logic isolated, the system remains modular and easy to debug.

## 2. The Search Pipeline
When a user issues a search request, the Query Engine oversees a strict, sequential pipeline:

1. **Validation:** The engine first inspects the incoming request, ensuring the query string is valid and that any provided constraints (like `top_k` limits or metadata filters) are properly formatted.
2. **Embedding:** The natural language query is passed to the Embeddings Module, which converts the text into a dense vector representation.
3. **Pre-filtering:** If the user provided metadata filters, the engine queries the Storage Engine's metadata cache to identify which documents match the criteria. It generates a boolean bitmask (an array of true/false values) representing eligible document IDs.
4. **Execution:** The generated query vector and the pre-filter bitmask are handed off to the Indexing Engine. The Indexing Engine performs the similarity search, considering only the documents allowed by the bitmask.
5. **Hydration:** The Indexing Engine returns raw document IDs and their similarity scores. The Query Engine then requests the original text and metadata for these specific IDs from the Storage Engine.
6. **Response Formulation:** The final step involves packing the hydrated text, metadata, and similarity scores into a clean, standardized list of Python dictionaries to return to the user.


## 3. Filtering Strategy
Handling metadata filters in a vector database can be challenging. SimpleV employs a "pre-filtering" strategy. By evaluating metadata constraints *before* executing the mathematical vector search, we ensure that the search algorithm only spends compute cycles on documents that are actually eligible to be returned. This is handled efficiently in memory using bitwise operations, preventing the search logic from becoming bottlenecked by dictionary lookups.

    ## 4. Error Handling and Resilience
The Query Engine acts as the primary defensive layer for the application. It gracefully catches issues such as empty queries, unsupported filter syntax, or requested `top_k` values that exceed the total database size. By normalizing these edge cases, it guarantees that the underlying Indexing and Storage engines only ever receive clean, actionable instructions.