"""In-process GPU LC codec backed by liblc_runtime.so.

Built by scripts/build_lc_runtime.py from LC-framework. Reads algorithm→chain
mappings from lc_runtime_meta.json and calls:

  lc_encode_device(chain, input, insize, output, capacity, host_outsize, stream)
  lc_decode_device(chain, input, insize, output, capacity, host_outsize, stream)

Surface matches KVServeCodec.encode/decode so it plugs into CompressionManager.
"""

from __future__ import annotations

import ctypes
import json
import math
import os
from pathlib import Path
from typing import Any, Optional

import torch

from kvserve_v1.compression.components import Codec


def _default_meta_path() -> str:
    return os.environ.get(
        "KVSERVE_LC_META_PATH",
        str(Path(__file__).resolve().parents[3] / "build" / "lc_runtime" / "lc_runtime_meta.json"),
    )


class LCCodec(Codec):
    """Lossless LC codec parallel to nvCOMP."""

    def __init__(self, **kwargs: Any) -> None:
        self.codec_type = "lc"
        self.lc_meta_path = str(
            kwargs.get("lc_meta_path")
            or os.environ.get("KVSERVE_LC_META_PATH")
            or _default_meta_path()
        )
        self.lc_algorithm = str(
            kwargs.get("lc_algorithm")
            or os.environ.get("KVSERVE_LC_ALGORITHM")
            or "TUPL8_1 BIT_8 RZE_2"
        )
        self._meta = self._load_meta(self.lc_meta_path)
        self._lib = self._load_lib(self._meta["library"])
        self._chain = self._resolve_chain(self.lc_algorithm)
        # Persistent encode staging buffer (device → tensor), avoids alloc+clone.
        self._encode_buf: Optional[torch.Tensor] = None
        self._encode_buf_cap: int = 0

    @staticmethod
    def _load_meta(path: str) -> dict[str, Any]:
        meta_path = Path(path)
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"LC runtime metadata not found: {meta_path}. "
                "Build it with scripts/build_lc_runtime.py"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        for key in ("library", "algorithms"):
            if key not in meta:
                raise ValueError(f"Invalid LC meta (missing {key}): {meta_path}")
        return meta

    @staticmethod
    def _load_lib(library: str) -> ctypes.CDLL:
        lib_path = Path(library)
        if not lib_path.is_file():
            raise FileNotFoundError(f"LC runtime library not found: {lib_path}")
        lib = ctypes.CDLL(str(lib_path))
        if not hasattr(lib, "lc_runtime_abi_version"):
            raise RuntimeError("LC runtime ABI is outdated; rebuild with scripts/build_lc_runtime.py")
        lib.lc_runtime_abi_version.restype = ctypes.c_int
        lib.lc_runtime_abi_version.argtypes = []
        if lib.lc_runtime_abi_version() != 2:
            raise RuntimeError("Unsupported LC runtime ABI; rebuild with scripts/build_lc_runtime.py")
        lib.lc_max_compressed_size.restype = ctypes.c_longlong
        lib.lc_max_compressed_size.argtypes = [ctypes.c_longlong]
        lib.lc_encode_device.restype = ctypes.c_int
        lib.lc_encode_device.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_void_p,
        ]
        lib.lc_decode_device.restype = ctypes.c_int
        lib.lc_decode_device.argtypes = [
            ctypes.c_ulonglong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.c_void_p,
            ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_longlong),
            ctypes.c_void_p,
        ]
        return lib

    def _resolve_chain(self, algorithm: str) -> int:
        algorithms = self._meta["algorithms"]
        if algorithm not in algorithms:
            known = ", ".join(sorted(algorithms))
            raise KeyError(
                f"Unknown LC algorithm {algorithm!r}. Built algorithms: {known}"
            )
        return int(algorithms[algorithm])

    def update_params(self, **kwargs: Any) -> None:
        if "lc_meta_path" in kwargs and kwargs["lc_meta_path"]:
            path = str(kwargs["lc_meta_path"])
            if path != self.lc_meta_path:
                self.lc_meta_path = path
                self._meta = self._load_meta(path)
                self._lib = self._load_lib(self._meta["library"])
        if "lc_algorithm" in kwargs and kwargs["lc_algorithm"]:
            self.lc_algorithm = str(kwargs["lc_algorithm"])
            self._chain = self._resolve_chain(self.lc_algorithm)

    def _acquire_encode_buf(self, device: torch.device, capacity: int) -> torch.Tensor:
        if (
            self._encode_buf is not None
            and self._encode_buf.device == device
            and self._encode_buf_cap >= capacity
        ):
            return self._encode_buf
        self._encode_buf = torch.empty(capacity, dtype=torch.uint8, device=device)
        self._encode_buf_cap = capacity
        return self._encode_buf

    def encode(self, tensor: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        """Match nvCOMPCodec.encode(tensor, **kwargs) for KVServeCodec routing."""
        if "lc_algorithm" in kwargs or "lc_meta_path" in kwargs:
            self.update_params(**kwargs)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"LCCodec.encode expects a torch.Tensor, got {type(tensor)}")
        if not tensor.is_cuda:
            raise ValueError("LCCodec.encode requires a CUDA tensor")

        d_in = tensor if tensor.is_contiguous() else tensor.contiguous()
        d_in = d_in.reshape(-1).view(torch.uint8)
        insize = int(d_in.numel())
        if insize == 0:
            return torch.empty(0, dtype=torch.uint8, device=d_in.device)

        maxsize = int(self._lib.lc_max_compressed_size(ctypes.c_longlong(insize)))
        # Reuse capacity across calls when input size is stable (common in PD).
        d_out = self._acquire_encode_buf(d_in.device, maxsize)
        host_outsize = ctypes.c_longlong(0)
        with torch.cuda.device(d_in.device):
            rc = self._lib.lc_encode_device(
                ctypes.c_ulonglong(self._chain),
                ctypes.c_void_p(d_in.data_ptr()),
                ctypes.c_longlong(insize),
                ctypes.c_void_p(d_out.data_ptr()),
                ctypes.c_longlong(maxsize),
                ctypes.byref(host_outsize),
                ctypes.c_void_p(torch.cuda.current_stream(d_in.device).cuda_stream),
            )
        if rc != 0:
            raise RuntimeError(f"lc_encode_device failed with code {rc}")
        outsize = int(host_outsize.value)
        if outsize <= 0 or outsize > maxsize:
            raise RuntimeError(f"lc_encode_device returned invalid outsize={outsize}")
        # Own an exact-sized tensor. Staging buffer stays pooled at maxsize.
        out = torch.empty(outsize, dtype=torch.uint8, device=d_in.device)
        out.copy_(d_out[:outsize])
        return out

    def decode(
        self,
        compressed_data: torch.Tensor,
        original_dtype: str,
        original_shape: list,
        device: str,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Match nvCOMPCodec.decode(...) for KVServeCodec routing."""
        if "lc_algorithm" in kwargs or "lc_meta_path" in kwargs:
            self.update_params(**kwargs)
        if not isinstance(compressed_data, torch.Tensor):
            raise TypeError(
                f"LCCodec.decode expects a torch.Tensor, got {type(compressed_data)}"
            )

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError("LCCodec.decode requires a CUDA device")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        comp = compressed_data.contiguous().view(torch.uint8).reshape(-1)
        if comp.device != dev:
            comp = comp.to(dev)

        shape = [int(s) for s in original_shape]
        if any(s < 0 for s in shape):
            raise ValueError("LCCodec.decode shape must be nonnegative")
        target_dtype = getattr(torch, original_dtype.replace("torch.", ""), None)
        if not isinstance(target_dtype, torch.dtype):
            raise ValueError(f"Invalid LC decode dtype: {original_dtype}")
        out_nbytes = math.prod(shape) * torch.empty((), dtype=target_dtype).element_size()
        if out_nbytes == 0:
            if comp.numel():
                raise ValueError("Nonempty LC payload for an empty output")
            return torch.empty(shape, dtype=target_dtype, device=dev)
        d_out = torch.empty(out_nbytes, dtype=torch.uint8, device=dev)
        host_outsize = ctypes.c_longlong(0)
        with torch.cuda.device(dev):
            rc = self._lib.lc_decode_device(
                ctypes.c_ulonglong(self._chain),
                ctypes.c_void_p(comp.data_ptr()),
                ctypes.c_longlong(comp.numel()),
                ctypes.c_void_p(d_out.data_ptr()),
                ctypes.c_longlong(out_nbytes),
                ctypes.byref(host_outsize),
                ctypes.c_void_p(torch.cuda.current_stream(dev).cuda_stream),
            )
        if rc != 0:
            raise RuntimeError(f"lc_decode_device failed with code {rc}")
        if host_outsize.value != out_nbytes:
            raise RuntimeError("LC runtime returned an unexpected decoded byte count")
        return d_out.view(target_dtype).reshape(shape)
