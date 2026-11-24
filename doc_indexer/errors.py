"""Domain exceptions.

Messages raised from here must be fixed, user-safe text. Never embed the string
form of a third-party exception or any connection detail / secret in them.
"""


class DocIndexerError(Exception):
    """Base class for all expected, user-facing errors."""

    exit_code = 1


class ConfigError(DocIndexerError):
    """Missing or invalid configuration."""

    exit_code = 2


class DocumentError(DocIndexerError):
    """Problem with the input document."""

    exit_code = 3


class DocumentNotFoundError(DocumentError):
    """The given file path does not exist."""


class UnsupportedFileTypeError(DocumentError):
    """Only PDF and DOCX are supported."""


class EmptyDocumentError(DocumentError):
    """The document contains no extractable text (e.g. a scanned PDF)."""


class EmbeddingError(DocIndexerError):
    """Embedding generation failed."""

    exit_code = 4


class DatabaseError(DocIndexerError):
    """Database connection or query failed."""

    exit_code = 5
