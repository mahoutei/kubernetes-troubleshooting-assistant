import os
import tempfile
from pathlib import Path

# Keep test collection hermetic: rag.api builds its module-level app on import.
os.environ["RAG_INDEX_DIR"] = str(Path(tempfile.mkdtemp(prefix="rag-test-")) / "missing")
