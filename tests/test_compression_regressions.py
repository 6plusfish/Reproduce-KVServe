"""Configuration, LC byte counts, and CUDA stream regression tests."""
import copy
import os
import json
from pathlib import Path

import pytest
import torch

from kvserve_v1.compression.manager import KVCompressionAdapter
from kvserve_v1.compression.quantizer.tilelang_quantizer import TileLangFusedQuantizer


def test_fused_normalization_preserves_seed_and_request_pipeline():
    spec = {"pipeline": ["transformer", "quantizer"],
            "transformer_config": {"transform_type": "hadamard", "seed": 1234},
            "quantizer_config": {"impl": "tilelang_fused"}}
    before = copy.deepcopy(spec)
    adapter = KVCompressionAdapter(spec, num_kv_heads=4, head_size=128)
    assert spec == before
    assert adapter._config.pipeline == ["quantizer"]
    assert adapter._manager.config.pipeline == adapter._config.pipeline
    assert adapter._config.quantizer_config["base_seed"] == 1234
    adapter._manager.update_config(adapter._config)
    assert adapter._manager.quantizer._cfg["base_seed"] == 1234


@pytest.mark.parametrize("patch", [
    {"axis_key": "token"}, {"axis_value": "tensor"},
    {"quant_type": "absmax"}, {"split_type": "invalid"},
])
def test_fused_rejects_unsupported_config(patch):
    with pytest.raises(ValueError):
        TileLangFusedQuantizer(**patch)
    q = TileLangFusedQuantizer()
    before = dict(q._cfg)
    with pytest.raises(ValueError):
        q.update_params(**patch)
    assert q._cfg == before


def test_fused_rejects_conflicting_seed():
    with pytest.raises(ValueError, match="Conflicting"):
        KVCompressionAdapter._normalize_config({
            "pipeline": ["transformer", "quantizer"], "impl": "tilelang_fused",
            "transformer_config": {"seed": 1}, "quantizer_config": {"base_seed": 2}})


def test_nvcomp_updates_constructor_options(monkeypatch):
    from kvserve_v1.compression.codec import kvserve_codec
    from kvserve_v1.compression.codec import nvcomp_func
    created = []
    def factory(**kwargs):
        created.append(kwargs)
        return object()
    monkeypatch.setattr(nvcomp_func, "nvCOMPCodec", factory)
    codec = kvserve_codec.KVServeCodec(data_type="|u1")
    codec.update_params(data_type="|u1")
    assert len(created) == 1
    codec.update_params(data_type="<i4")
    assert len(created) == 2
    assert created[-1]["data_type"] == "<i4"


@pytest.fixture
def lc():
    if not torch.cuda.is_available() or not os.environ.get("KVSERVE_LC_META_PATH"):
        pytest.skip("CUDA and a built LC runtime are required")
    from kvserve_v1.compression.codec.lc_codec import LCCodec
    return LCCodec()


@pytest.mark.parametrize("dtype", [torch.uint8, torch.float16, torch.bfloat16, torch.float32])
def test_lc_roundtrip_dtypes(lc, dtype):
    x = torch.randint(0, 12, (2, 3, 257), device="cuda").to(dtype)
    y = lc.encode(x)
    z = lc.decode(y, str(dtype), list(x.shape), str(x.device))
    assert z.dtype == dtype and z.shape == x.shape
    assert torch.equal(x, z)


def test_lc_nondefault_stream(lc):
    x = torch.randint(0, 12, (1 << 20,), dtype=torch.uint8, device="cuda")
    y = lc.encode(x)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        lc.decode(y, "uint8", list(x.shape), str(x.device))
    torch.cuda.synchronize()
    # Delay the default stream to make a missing dependency reproducible.
    with torch.cuda.stream(torch.cuda.default_stream()):
        torch.cuda._sleep(800_000_000)
    with torch.cuda.stream(stream):
        z = lc.decode(y, "uint8", list(x.shape), str(x.device)).clone()
        ok = torch.equal(x, z)
    torch.cuda.synchronize()
    assert ok


def test_lc_wrong_shape_rejected_before_decode(lc):
    x = torch.arange(1024, device="cuda", dtype=torch.float32)
    y = lc.encode(x)
    with pytest.raises(RuntimeError, match="code -7"):
        lc.decode(y, "float32", [1023], str(x.device))
    assert torch.equal(x, lc.decode(y, "float32", [1024], str(x.device)))


def test_lc_other_current_device(lc):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two CUDA devices are required")
    with torch.cuda.device(1):
        x = torch.randn(1024, device="cuda:0")
        y = lc.encode(x)
        z = lc.decode(y, "float32", [1024], "cuda:0")
        assert torch.cuda.current_device() == 1
        assert torch.equal(x, z)


def test_lc_codec_only_pipeline(lc):
    adapter = KVCompressionAdapter({"pipeline": ["codec"], "codec_config": {"codec_type": "lc"}},
                                   num_kv_heads=2, head_size=128)
    x = torch.randn(2, 2, 3, 16, 2, 128, dtype=torch.bfloat16, device="cuda")
    compressed = adapter.compress(x, "codec-only")
    assert compressed is not None and compressed.metadata["codec_applied"]
    assert torch.equal(x, adapter.decompress(compressed))


@pytest.mark.parametrize("shape", [[], [0, 2]])
def test_lc_scalar_and_empty(lc, shape):
    x = torch.full(shape, 1.5, dtype=torch.float32, device="cuda")
    compressed = lc.encode(x)
    assert torch.equal(x, lc.decode(compressed, "float32", shape, str(x.device)))


def test_wire_scalar_metadata():
    from kvserve_v1.compression.wire import _coalesce_aux_tensors, _expand_aux_tensors
    values = [torch.tensor(1.5), torch.tensor([1, 2], dtype=torch.int64)]
    meta = {}
    restored = _expand_aux_tensors(_coalesce_aux_tensors(meta, values), meta["aux_layout"])
    assert all(torch.equal(a, b) for a, b in zip(values, restored))


@pytest.mark.parametrize("profile", ["original_top_lc", "original_top_nvcomp",
                                    "fused_top_lc", "fused_top_nvcomp"])
@pytest.mark.parametrize("tp_size,tp_rank", [(1, 0), (2, 0), (2, 1)])
def test_pipeline_wire_roundtrip(lc, profile, tp_size, tp_rank):
    from kvserve_v1.compression.wire import build_wire, restore_from_wire
    config_path = Path(__file__).resolve().parents[1] / "configs/compression" / (profile + ".json")
    config = json.loads(config_path.read_text())
    heads = 8 // tp_size
    adapter = KVCompressionAdapter(config, heads, 128, tp_rank=tp_rank, tp_size=tp_size)
    x = torch.randn(2, 2, 3, 16, heads, 128, device="cuda", dtype=torch.bfloat16)
    compressed = adapter.compress(x, "wire-regression")
    assert compressed is not None
    assert compressed.metadata["codec_applied"]
    restored = adapter.decompress(restore_from_wire(build_wire(compressed, max_chunk_bytes=4096)))
    assert restored is not None and restored.shape == x.shape
    assert torch.isfinite(restored).all()
    # Lossy pipelines need an accuracy check, not merely matching shapes.
    cosine = torch.nn.functional.cosine_similarity(x.float().flatten(), restored.float().flatten(), dim=0)
    assert cosine > 0.9
