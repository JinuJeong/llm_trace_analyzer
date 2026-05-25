import os
import enum
import json
import urllib.request
from dataclasses import dataclass, field
from typing import NamedTuple
from pathlib import Path
from typing import List, Optional

CACHE_DIR = Path(os.path.expanduser("~/.cache/gds-llm-analysis/traces"))


class TraceType(enum.Enum):
    MOONCAKE = enum.auto()


class TraceInfo(NamedTuple):
    trace_type: TraceType
    url: str


TRACE_REGISTRY = {
    "mooncake":
    TraceInfo(
        TraceType.MOONCAKE,
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/arxiv-trace/mooncake_trace.jsonl",
    ),
    "conversation":
    TraceInfo(
        TraceType.MOONCAKE,
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/conversation_trace.jsonl",
    ),
    "synthetic":
    TraceInfo(
        TraceType.MOONCAKE,
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/synthetic_trace.jsonl",
    ),
    "toolagent":
    TraceInfo(
        TraceType.MOONCAKE,
        "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces/toolagent_trace.jsonl",
    ),
}


HASH_BLOCK_SIZE = 512


@dataclass
class TraceRequest:
    timestamp: float
    input_length: int
    output_length: int
    hash_ids: List[int] = field(default_factory=list)
    computed_tokens: int = 0


class Trace:

    def __init__(self, requests: List[TraceRequest]):
        self.requests = sorted(requests, key=lambda r: r.timestamp)
        self._simulate_prefix_cache()

    def _simulate_prefix_cache(self):
        cached: set = set()
        for req in self.requests:
            prefix_blocks = 0
            for hid in req.hash_ids:
                if hid in cached:
                    prefix_blocks += 1
                else:
                    break
            req.computed_tokens = prefix_blocks * HASH_BLOCK_SIZE
            cached.update(req.hash_ids)
        self.unique_kv_tokens = len(cached) * HASH_BLOCK_SIZE


def _download_trace(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_suffix(dest.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp_path)
        tmp_path.rename(dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    print(f"Download complete: {dest}")


def _load_mooncake_trace(path: Path) -> List[TraceRequest]:
    requests = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            requests.append(
                TraceRequest(
                    timestamp=entry["timestamp"],
                    input_length=entry["input_length"],
                    output_length=entry["output_length"],
                    hash_ids=entry.get("hash_ids", []),
                )
            )
    return requests


def download_trace(trace_name: str = "", cache_dir: Path = CACHE_DIR) -> Path:
    trace_name = trace_name.lower()
    if trace_name not in TRACE_REGISTRY:
        raise ValueError(
            f"Unkown trace '{trace_name}'. Supported: {TRACE_REGISTRY.keys()}")

    info = TRACE_REGISTRY[trace_name]
    url = info.url

    filename = url.split("/")[-1]
    dest = cache_dir / filename
    if dest.exists():
        return dest

    _download_trace(url, dest)
    return dest


def get_trace(
    trace_name: str = "mooncake",
    local_path: Optional[str] = None,
) -> Trace:
    if trace_name not in TRACE_REGISTRY:
        raise ValueError(
            f"Unkown trace '{trace_name}'. Supported: {TRACE_REGISTRY.keys()}")

    if local_path:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(f"Trace file not found: {path}")
    else:
        path = download_trace(trace_name, CACHE_DIR)

    requests = _load_mooncake_trace(path)
    return Trace(requests)
