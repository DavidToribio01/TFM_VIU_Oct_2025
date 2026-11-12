from bias_detector.alert_generator import BIASES, generate_alerts, MissingExplanationError
import pytest

def test_no_predictions_no_alerts():
    result = generate_alerts(conversation_id="sample", scores={bias:0.0 for bias in BIASES}, thresholds={bias:0.5 for bias in BIASES}, shap_fragments={})
    assert result == []

def test_positive_requires_shap():
    scores={bias:0.0 for bias in BIASES}; scores[BIASES[0]]=0.8
    with pytest.raises(MissingExplanationError):
        generate_alerts(conversation_id="sample", scores=scores, thresholds={bias:0.5 for bias in BIASES}, shap_fragments={})
