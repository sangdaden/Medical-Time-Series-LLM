import numpy as np
import src.modalities as M
from src import framework, config


def test_registry_has_three_modality_types():
    for name in ["ecg", "rr", "clinical"]:
        assert name in framework.REGISTRY
    specs = framework.get_specs(["ecg", "rr", "clinical"])
    assert [s.feat_dim for s in specs] == [config.WINDOW, config.RR_CONTEXT, M.CLINICAL_DIM]


def test_get_specs_rejects_unknown():
    try:
        framework.get_specs(["ecg", "mri"])
        assert False, "expected KeyError"
    except KeyError:
        pass


def test_parse_clinical_real_header_line():
    # MIT-BIH header comment format: "<age> <sex> ..." then a medications line
    feat = M.parse_clinical(["69 M 1085 1629 x1", "Aldomet, Inderal"])
    assert feat.shape == (3,)
    assert abs(feat[0] - 0.69) < 1e-6     # age 69 / 100
    assert feat[1] == 1.0                  # M
    assert abs(feat[2] - 2 / 5) < 1e-6     # 2 medications / 5


def test_parse_clinical_female_and_prose_ignored():
    feat = M.parse_clinical(["76 F 2777 3655 x2", "The rhythm is compatible with sick sinus."])
    assert feat[1] == 0.0                   # F
    assert feat[2] == 0.0                   # prose line is not counted as medications


def test_parse_clinical_defaults_on_empty():
    feat = M.parse_clinical(None)
    assert feat.shape == (3,) and feat[1] == 0.5
