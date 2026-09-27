"""Decode scheduling must consume available payloads before waiting for credit."""
from collections import defaultdict, deque
from types import SimpleNamespace
import threading
import time

import torch
import pytest

import kvserve_v1.connector.compressed_kv_connector as connector_module
from kvserve_v1.connector.compressed_kv_connector import (
    CompressedKVConnector, CompressedKVConnectorMetadata, ReqMeta,
)


def test_load_reordered_payloads_releases_credit(monkeypatch):
    class Transport:
        device = torch.device("cpu")
        def __init__(self):
            self.ready = {"B": [([], torch.ones(1))]}
            self.rows = []
        def drain_received(self):
            ready, self.ready = self.ready, {}
            return ready
        def payload_nbytes(self, payload):
            return payload.numel() * payload.element_size()
        def release_received(self, rid, nbytes):
            if rid == "B":
                self.ready["A"] = [([], torch.ones(1))]
        def record_stat(self, row):
            self.rows.append(row)
        def set_expected_requests(self, rids):
            pass
    obj = CompressedKVConnector.__new__(CompressedKVConnector)
    obj.is_producer = False
    obj._transport = Transport()
    obj._worker_received_kv = defaultdict(deque)
    meta = CompressedKVConnectorMetadata(requests=[
        ReqMeta(rid, rid, [], [], torch.empty(0, dtype=torch.long)) for rid in ["A", "B"]])
    obj._get_connector_metadata = lambda: meta
    monkeypatch.setattr(connector_module, "_LOAD_TIMEOUT_S", 0.1)
    obj.start_load_kv(SimpleNamespace(no_compile_layers={}, virtual_engine=0))
    assert not any(row["kind"] == "timeout" for row in obj._transport.rows)
    assert {row["request_id"] for row in obj._transport.rows} == {"A", "B"}
    assert not obj._worker_received_kv


def test_one_scheduled_overflow_unblocks_next_decode_step():
    from kvserve_v1.transport.nccl_transport import NcclTransport
    transport = NcclTransport.__new__(NcclTransport)
    transport.is_sender = False
    transport._recv_budget_cv = threading.Condition()
    transport._recv_capacity_bytes = 100
    transport._recv_reserved_bytes = 0
    transport._expected_requests = set()
    transport._priority_request = None
    transport._record_stat = lambda row: None
    transport._reserve_received("future", 100)
    transport.set_expected_requests(["current-1", "current-2"])
    transport._reserve_received("current-1", 100)
    assert transport._recv_reserved_bytes == 200
    done = threading.Event()
    thread = threading.Thread(target=lambda: (transport._reserve_received("current-2", 100), done.set()))
    thread.start()
    assert not done.wait(0.05)
    transport.release_received("current-1", 100)
    assert done.wait(1)
    thread.join()
    assert transport._recv_reserved_bytes == 200
    transport.release_received("current-2", 100)
    transport.release_received("future", 100)
    assert transport._recv_reserved_bytes == 0
    assert transport._priority_request is None
