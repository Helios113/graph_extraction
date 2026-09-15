"""Tests for gmanifold/onnx_transformer.py against configs/simple_stories.toml
(SimpleStories-V2-11M, already exported to onnx_models/simple_stories/).
Requires a CUDA-capable node (see construct.construct's CUDAExecutionProvider
sessions) -- run via srun on mauao/ruapehu, not the login node.
"""
import torch
import pytest

from config import load_config
from gmanifold import onnx_transformer as tr

CONFIG_PATH = "configs/simple_stories.toml"


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_PATH)


def test_vocab_states_shape(cfg):
    X, ids = tr.vocab_states(cfg)
    assert X.shape == (cfg.vocab_size, cfg.d_model)
    assert ids == list(range(cfg.vocab_size))


def test_collect_states_matches_vocab_states_at_position_zero(cfg):
    """embedding is tokenwise and position-independent, so collect_states at
    position 0 of an all-zero-padded sequence must agree with vocab_states
    (which reads the same location the same way)."""
    X, ids = tr.vocab_states(cfg)
    sample_ids = torch.tensor(ids[:37])
    input_ids = torch.zeros(len(sample_ids), cfg.seq, dtype=torch.long)
    input_ids[:, 0] = sample_ids
    positions = torch.zeros(len(sample_ids), dtype=torch.long)
    st = tr.collect_states(cfg, input_ids, positions, ["embedding"])
    assert torch.allclose(st["embedding"], X[:37], atol=1e-5)


def test_make_map_reproduces_real_ffn_output(cfg):
    """add_6 -> add_8 is SimpleStories-V2-11M block 0's FFN sublayer
    (declared in configs/simple_stories.toml) -- a real forward pass through
    it (collect_states, reading the graph's own intermediate tensors) must
    match make_map's extracted-subgraph forward pass exactly."""
    X, ids = tr.vocab_states(cfg)
    sample_ids = torch.tensor(ids[:64])
    input_ids = torch.zeros(len(sample_ids), cfg.seq, dtype=torch.long)
    input_ids[:, 0] = sample_ids
    positions = torch.zeros(len(sample_ids), dtype=torch.long)
    st = tr.collect_states(cfg, input_ids, positions, ["add_6", "add_8"])

    f = tr.make_map(cfg, "add_6", "add_8")
    assert torch.allclose(f(st["add_6"]), st["add_8"], atol=1e-4)


def test_make_map_rejects_undeclared_sublayer(cfg):
    with pytest.raises(ValueError, match="no sublayer"):
        tr.make_map(cfg, "embedding", "add_100")
