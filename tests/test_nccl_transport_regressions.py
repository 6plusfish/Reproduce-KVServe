"""Opt-in two-GPU NCCL data integrity and receiver-credit integration tests."""
import json
import multiprocessing as mp
import os
import time

import pytest
import torch


def _transport_worker(sender, port, channel, stats, results, cross_step=False):
    try:
        from kvserve_v1.transport.nccl_transport import NcclTransport
        torch.cuda.set_device(0 if sender else 1)
        os.environ["KVSERVE_TRANSPORT_STATS_PATH"] = stats
        transport = NcclTransport(sender, "127.0.0.1", port,
                                  local_rank=0 if sender else 1,
                                  channel_rank=channel, recv_buffer_size=262144)
        plan = [1, 2, 0] if cross_step else list(range(6))
        if sender:
            stream = torch.cuda.Stream(device=0)
            with torch.cuda.stream(stream):
                for i in plan:
                    size = 524288 if i == 2 and not cross_step else 262144
                    body = torch.full((size,), i, dtype=torch.uint8, device=0)
                    if i % 2:
                        transport.send_bundle(str(i), ["__compressed__"], {"i": i},
                                              [body], [torch.empty(0, dtype=torch.uint8, device=0)])
                    else:
                        transport.send(str(i), ["layer"], body)
            wait_s = transport.wait_for_below(262144)
            transport.wait_for_sent()
            assert transport.pending_bytes() == 0
            results.put({"sender": True, "wait_s": wait_s})
        else:
            seen = []
            buffered = {}
            if cross_step:
                transport.set_expected_requests(["0"])
            deadline = time.monotonic() + 10
            while len(seen) < len(plan):
                assert time.monotonic() < deadline, "receive timeout"
                batch = transport.drain_received()
                buffered.update(batch)
                batch = {rid: items for rid, items in buffered.items()
                         if not cross_step or "0" in seen or rid == "0"}
                for rid, items in batch.items():
                    for names, payload in items:
                        body = payload["body_chunks"][0] if isinstance(payload, dict) else payload
                        assert bool(torch.all(body == int(rid)))
                        time.sleep(0.03)  # Force producer and receiver backpressure.
                        transport.release_received(rid, transport.payload_nbytes(payload))
                        seen.append(rid)
                    buffered.pop(rid)
                if cross_step and "0" in seen:
                    transport.set_expected_requests(["1", "2"])
                time.sleep(0.002)
            assert sorted(seen) == sorted(str(i) for i in plan)
            assert transport._recv_reserved_bytes == 0
            results.put({"sender": False, "seen": seen})
    except BaseException as exc:
        results.put({"error": repr(exc), "sender": sender})
        raise


@pytest.mark.skipif(os.environ.get("KVSERVE_RUN_NCCL_TESTS") != "1",
                    reason="Set KVSERVE_RUN_NCCL_TESTS=1 on a two-GPU host")
@pytest.mark.parametrize("channel,cross_step", [(0, False), (1, False), (2, True)])
def test_real_nccl_credit_and_bundle_transfer(tmp_path, channel, cross_step):
    assert torch.cuda.device_count() >= 2
    context = mp.get_context("spawn")
    queue = context.Queue()
    stats = str(tmp_path / "transport.jsonl")
    processes = [context.Process(target=_transport_worker,
                                args=(sender, 29310, channel, stats, queue, cross_step))
                 for sender in [False, True]]
    try:
        for process in processes:
            process.start()
        outcomes = [queue.get(timeout=60) for _ in processes]
        for process in processes:
            process.join(10)
        assert all(process.exitcode == 0 for process in processes), outcomes
        assert not any("error" in outcome for outcome in outcomes), outcomes
        if not cross_step:
            assert next(outcome for outcome in outcomes if outcome["sender"])["wait_s"] > 0.02
        rows = [json.loads(line) for line in open(stats)]
        assert sum(r["direction"] == "recv" for r in rows) == (3 if cross_step else 6)
        if not cross_step:
            assert any(r["direction"] == "recv_credit" and r["oversize"] for r in rows)
        assert any(r["direction"] == "credit_retry" for r in rows)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(10)
