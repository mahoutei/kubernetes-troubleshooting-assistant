import ast
import hashlib
import json
from pathlib import Path

from scripts.build_colab import NOTEBOOK, ROOT, SOURCES, build_notebook, render


def test_notebook_is_current_and_deterministic():
    assert (ROOT / NOTEBOOK).read_text() == render()
    assert render() == render()


def test_notebook_cells_compile_and_have_cleared_outputs():
    notebook = build_notebook()
    assert notebook["nbformat"] == 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None
            ast.parse("".join(cell["source"]))


def test_embedded_files_reconstruct_exact_sources_without_repo_access(tmp_path):
    namespace = {"FILES": {}}
    for cell in build_notebook()["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] == "code" and source.startswith("FILES["):
            assignment = ast.parse(source).body[0]
            target = ast.literal_eval(assignment.targets[0].slice)
            namespace["FILES"][target] = ast.literal_eval(assignment.value)
    assert set(namespace["FILES"]) == set(SOURCES)
    for target, original in SOURCES.items():
        text = namespace["FILES"][target]
        assert text == (ROOT / original).read_text()
        if target.endswith(".py"):
            ast.parse(text)
        destination = tmp_path / target
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(text)
        assert hashlib.sha256(destination.read_bytes()).hexdigest() == build_notebook()["metadata"]["embedded_sources"][target]


def test_notebook_configuration_and_commands_are_standalone():
    notebook = build_notebook()
    text = "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code" and not "".join(c["source"]).startswith("FILES["))
    assert 'AUTO_FULL = True' in text
    assert 'USE_DRIVE = False' in text
    assert 'RESUME_FULL = False' in text
    assert 'git clone' not in text and 'REPO_URL' not in text
    assert '"sync", "--project"' in text and '"--frozen"' in text
    assert text.index('gpu_run("gpu_check.py")') < text.index('gpu_run("training/colab_runner.py", "prepare")')
    assert text.index('"smoke")') < text.index('"full")')
    json.dumps(notebook)
    assert Path(NOTEBOOK).suffix == ".ipynb"
