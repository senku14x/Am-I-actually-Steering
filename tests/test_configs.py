"""Config smoke tests: every model_*.yaml loads on base.yaml and declares sane geometry fields.

Cheap guard so a malformed or half-filled model config fails CI instead of a GPU run. Parametrized
over whatever model configs exist, so adding a model is automatically covered.
"""
from pathlib import Path

import pytest

from strat_geom.config import load_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
BASE = CONFIGS / "base.yaml"
MODEL_CONFIGS = sorted(CONFIGS.glob("model_*.yaml"))


def test_model_configs_present():
    assert MODEL_CONFIGS, "no configs/model_*.yaml found"


@pytest.mark.parametrize("model_cfg", MODEL_CONFIGS, ids=lambda p: p.stem)
def test_model_config_loads_and_is_sane(model_cfg):
    cfg = load_config(str(BASE), str(model_cfg))
    assert cfg.model_name, f"{model_cfg.name}: empty model_name"
    assert cfg.model_short, f"{model_cfg.name}: empty model_short"
    assert cfg.n_layers > 0, f"{model_cfg.name}: n_layers must be > 0"
    assert cfg.hidden_dim > 0, f"{model_cfg.name}: hidden_dim must be > 0"
    assert cfg.max_new_tokens > 0, f"{model_cfg.name}: max_new_tokens must be > 0"
    # headline layer must be a real interior layer (drives the geometry/causal analysis)
    assert 0 < cfg.headline_layer < cfg.n_layers, (
        f"{model_cfg.name}: headline_layer {cfg.headline_layer} not in (0, {cfg.n_layers})"
    )
