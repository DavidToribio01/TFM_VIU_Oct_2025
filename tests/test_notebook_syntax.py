import json
import ast
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
NBS = sorted((ROOT/'notebooks').glob('colab/*.ipynb')) + sorted((ROOT/'notebooks').glob('visual_studio/*.ipynb'))

@pytest.mark.parametrize('path', NBS, ids=[x.name for x in NBS])
def test_notebook_cells_parse(path):
    from IPython.core.interactiveshell import InteractiveShell
    transformer = InteractiveShell.instance().input_transformer_manager
    doc = json.loads(path.read_text(encoding='utf-8'))
    for i, c in enumerate(doc['cells']):
        if c['cell_type'] == 'code':
            src = ''.join(c['source'])
            transformed = transformer.transform_cell(src)
            compile(transformed, f'{path.name}:celda {i}', 'exec')
            assert not ('drive.mount(' in src or '/content/drive/' in src)


def test_no_outdated_repo_bootstrap():
    for path in NBS:
        src = path.read_text(encoding='utf-8')
        assert 'tfm_unificado_v2' not in src
