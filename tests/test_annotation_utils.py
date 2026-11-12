from types import SimpleNamespace
import pytest
from bias_detector.annotation_utils import load_prompt_templates, format_annotation_prompt, validate_annotation_turns


def test_load_and_format_prompt(tmp_path):
    file = tmp_path / 'a.txt'
    file.write_text('Conversación: {{CONVERSATION}}', encoding='utf-8')
    prompts = load_prompt_templates({'confirmation_bias': file})
    assert format_annotation_prompt(prompts, 'confirmation_bias', 'Hola') == 'Conversación: Hola'
    with pytest.raises(ValueError, match='Sesgo desconocido'):
        format_annotation_prompt(prompts, 'otro', 'Hola')


def test_template_requires_placeholder(tmp_path):
    file = tmp_path / 'a.txt'
    file.write_text('Sin marcador', encoding='utf-8')
    with pytest.raises(ValueError):
        load_prompt_templates({'confirmation_bias': file})


def test_validate_turns():
    ann = SimpleNamespace(bias='confirmation_bias', trigger_turns=[1], evidence_turns=[2])
    assert validate_annotation_turns(ann, 'confirmation_bias', 2)
    with pytest.raises(ValueError):
        validate_annotation_turns(ann, 'confirmation_bias', 1)
