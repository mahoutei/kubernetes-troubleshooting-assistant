import os
import tempfile
from pathlib import Path

# Defensive isolation even if another test starts the module-level app.
_test_dir = tempfile.TemporaryDirectory(prefix="rag-test-")
os.environ["RAG_INDEX_DIR"] = str(Path(_test_dir.name) / "missing")
