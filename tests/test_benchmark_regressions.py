"""Benchmark wall time must include measured inference and transport work."""
from types import SimpleNamespace
import importlib
import json
from pathlib import Path
import sys

import pytest

from scripts.run_remote_pressure import replica_kv_ports


@pytest.fixture
def remote_benchmark(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parent))
    return importlib.import_module("test_kvserve_remote")


@pytest.mark.parametrize("role", ["prefill", "decode"])
def test_remote_submits_after_barrier_and_inside_timer(monkeypatch, tmp_path, role, remote_benchmark):
    benchmark = remote_benchmark
    clock = [0.0]
    ready = [False]
    sent = []
    captured = {}
    shutdown = []
    output = SimpleNamespace(prompt="hello", prompt_token_ids=[1, 2],
                             outputs=[SimpleNamespace(text="ok", token_ids=[3])])

    class LLM:
        def generate(self, prompts, sampling_params, **kwargs):
            if "warmup" in sampling_params:
                clock[0] += 1000
            else:
                assert ready[0], "Measured work started before the barrier"
                if role == "decode":
                    assert b"GO\n" in sent
                clock[0] += 5
            return [output for _ in prompts]

    class Barrier:
        def sendall(self, data):
            sent.append(data)
        def close(self):
            pass

    def build(*args):
        clock[0] += 500
        return LLM()

    def connect(*args):
        clock[0] += 2000
        ready[0] = True
        return Barrier()

    def accept(*args):
        return Barrier(), connect()

    def done(*args):
        clock[0] += 1
        return "DONE 5.000000000"

    monkeypatch.setattr(benchmark, "time", SimpleNamespace(perf_counter=lambda: clock[0]))
    monkeypatch.setattr(benchmark, "_build_llm", build)
    monkeypatch.setattr(benchmark, "_make_params", lambda prompts, tokens, prefix, ignore_eos=False: prefix)
    monkeypatch.setattr(benchmark, "_connect_measurement_barrier", connect)
    monkeypatch.setattr(benchmark, "_accept_measurement_barrier", accept)
    monkeypatch.setattr(benchmark, "_recv_barrier_line", done)
    monkeypatch.setattr(benchmark, "_read_ib_bytes", lambda *args: None)
    monkeypatch.setattr(benchmark, "_load_measured_compression_ratios", lambda *args: [])
    monkeypatch.setattr(benchmark, "_write_benchmark_summary", lambda **kwargs: captured.update(kwargs))
    monkeypatch.setattr(benchmark, "_shutdown_vllm", lambda llm: shutdown.append(True))
    monkeypatch.setattr(benchmark, "print_summary", lambda *args, **kwargs: None)
    monkeypatch.setattr(benchmark, "save_csv", lambda *args: None)
    args = SimpleNamespace(gpus="0", transport_stats_path=str(tmp_path / "stats.jsonl"),
                           async_send=True, max_inflight_gib=1, compression_stats_path=None,
                           kv_ip="127.0.0.1", sync_port=1000, sync_timeout_s=10,
                           transfer_prefix="test", ib_device=None, ib_port=1,
                           max_tokens=1, ignore_eos=True, print_outputs=False, run_label="test",
                           lmeval_task=None, data_path=None, output_dir=str(tmp_path))
    run = benchmark.run_prefill if role == "prefill" else benchmark.run_decode
    run(args, ["hello"], ["warmup"], None, 0)
    assert captured["engine_init_s"] == 500
    assert captured["warmup_s"] == 1000
    assert captured["measured_s"] == (6 if role == "prefill" else 5)
    assert shutdown == [True]


def test_pressure_tp_channels_are_disjoint():
    assert replica_kv_ports(25000, ["0", "1", "2"]) == [25000, 25001, 25002]
    assert replica_kv_ports(25000, ["0,1", "2,3", "4"]) == [25000, 25002, 25004]


def test_remote_fixed_output_budget(remote_benchmark):
    params = remote_benchmark._make_params(["hello"], 32, "measured", ignore_eos=True)
    assert params[0].ignore_eos
    assert params[0].max_tokens == 32
    assert params[0].extra_args["kv_transfer_params"]["transfer_id"] == "measured-0"


def test_hotpot_cached_context_preserves_question(tmp_path, remote_benchmark):
    import test_kvserve
    question = "Which city hosted the event?"
    path = tmp_path / "hotpotqa.jsonl"
    path.write_text(json.dumps({"context": "Answer the question based on passages.\n" + "context "*1000,
                                "question": question}) + "\n")
    prompt = test_kvserve.load_jsonl_prompts(str(path), 1, max_prompt_chars=1000)[0]
    assert len(prompt) == 1000
    assert prompt.startswith("Answer the question")
    assert prompt.endswith(f"Question: {question}\nAnswer:")


def test_token_truncation_keeps_both_ends(monkeypatch, remote_benchmark):
    import test_kvserve
    tokenizer = SimpleNamespace(encode=lambda text, **kwargs: list(text),
                                decode=lambda tokens, **kwargs: "".join(tokens))
    fake = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=lambda *args, **kwargs: tokenizer))
    monkeypatch.setitem(sys.modules, "transformers", fake)
    prompt = "INSTRUCTION\n" + "context "*100 + "\nQUESTION?"
    result = test_kvserve._truncate_prompts_by_tokens([prompt], max_prompt_tokens=40)[0]
    assert len(result) == 40
    assert result.startswith("INSTRUCTION\n")
    assert result.endswith("\nQUESTION?")
