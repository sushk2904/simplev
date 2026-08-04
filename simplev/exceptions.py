"""
Custom exceptions for SimpleV.

Keeping these separate so every module can import them
without circular dependency headaches.
"""


class SimpleVError(Exception):
    """Base exception for all SimpleV errors."""
    pass


class EmbeddingError(SimpleVError):
    """Raised when something goes wrong during embedding generation.
    
    Could be a model loading failure, bad input text,
    or the model returning something unexpected.
    """
    pass


class StorageError(SimpleVError):
    """Raised for disk I/O or serialization problems."""
    pass


class QueryError(SimpleVError):
    """Raised when search validation or execution fails."""
    pass