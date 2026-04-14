"""Document loaders for raw source files.

Supports: PDF, plain text, Markdown, HTML, CSV.
Each loaded document gets `source` and `file_name` metadata.
"""
import logging
from pathlib import Path

from langchain_core.documents import Document

logger = logging.getLogger("hybrid_rag.ingestion.loader")

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".html", ".htm", ".csv"}


def load_file(file_path: str | Path) -> list[Document]:
    """Load a single file into a list of LangChain Documents."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()

    if ext == ".pdf":
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))
    elif ext in (".txt", ".md"):
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(path), encoding="utf-8")
    elif ext in (".html", ".htm"):
        from langchain_community.document_loaders import UnstructuredHTMLLoader
        loader = UnstructuredHTMLLoader(str(path))
    elif ext == ".csv":
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(str(path))
    else:
        raise ValueError(
            f"Unsupported file type: {ext!r}. Supported: {SUPPORTED_EXTENSIONS}"
        )

    docs = loader.load()
    for doc in docs:
        doc.metadata.setdefault("source", str(path))
        doc.metadata.setdefault("file_name", path.name)

    logger.info(f"Loaded {len(docs)} page(s) from '{path.name}'")
    return docs


def load_directory(dir_path: str | Path, recursive: bool = True) -> list[Document]:
    """Recursively load all supported files from a directory."""
    path = Path(dir_path)
    if not path.is_dir():
        raise NotADirectoryError(f"Not a directory: {path}")

    pattern = "**/*" if recursive else "*"
    files = [
        f for f in sorted(path.glob(pattern))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        logger.warning(f"No supported files found in '{path}'")
        return []

    logger.info(f"Found {len(files)} file(s) in '{path}'")
    all_docs: list[Document] = []

    for file in files:
        try:
            docs = load_file(file)
            all_docs.extend(docs)
        except Exception as exc:
            logger.error(f"Failed to load '{file.name}': {exc}")

    logger.info(f"Total pages/records loaded: {len(all_docs)}")
    return all_docs
