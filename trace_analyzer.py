import math
import statistics
import argparse

from transformers import AutoConfig

from trace import get_trace, TRACE_REGISTRY


def _get_llm_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def _num_full_attn_layers(config) -> int:
    if hasattr(config, "layer_types"):
        return sum(1 for lt in config.layer_types if lt == "full_attention")
    return config.num_hidden_layers


def _load_model_info(model_name: str, dtype_bytes: int = 2):
    config = _get_llm_config(AutoConfig.from_pretrained(model_name))
    num_layers = _num_full_attn_layers(config)
    num_kv_heads = getattr(config, "num_key_value_heads",
                           config.num_attention_heads)
    head_dim = getattr(config, "head_dim",
                       config.hidden_size // config.num_attention_heads)
    kv_per_token = 2 * num_kv_heads * head_dim * num_layers * dtype_bytes
    return {
        "model_name": model_name,
        "num_hidden_layers": config.num_hidden_layers,
        "num_full_attn_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype_bytes": dtype_bytes,
        "kv_bytes_per_token": kv_per_token,
    }


def _percentile(data: list, pct: float) -> float:
    if not data:
        return 0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (pct / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return sorted_data[int(k)]
    return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)


def _stats(data: list) -> dict:
    if not data:
        return {}
    return {
        "count": len(data),
        "mean": statistics.mean(data),
        "median": statistics.median(data),
        "stdev": statistics.stdev(data) if len(data) > 1 else 0,
        "min": min(data),
        "max": max(data),
        "p50": _percentile(data, 50),
        "p90": _percentile(data, 90),
        "p95": _percentile(data, 95),
        "p99": _percentile(data, 99),
    }


def analyze(trace, kv_per_token: int = 0) -> dict:
    in_lens = [r.input_length for r in trace.requests]
    computed_lens = [r.computed_tokens for r in trace.requests]
    new_computed_lens = [
        max(0, r.input_length - r.computed_tokens) for r in trace.requests
    ]
    out_lens = [r.output_length for r in trace.requests]

    total_computed = sum(computed_lens)
    total_input = sum(in_lens)

    result = {
        "num_requests": len(trace.requests),
        "input_length": _stats(in_lens),
        "computed_tokens": _stats(computed_lens),
        "new_computed": _stats(new_computed_lens),
        "output_length": _stats(out_lens),
    }

    if kv_per_token > 0:
        unique_kv = trace.unique_kv_tokens * kv_per_token
        avg_computed_kv = statistics.mean(computed_lens) * kv_per_token
        avg_new_computed_kv = statistics.mean(new_computed_lens) * kv_per_token
        result.update({
            "kv_per_token": kv_per_token,
            "unique_kv": unique_kv,
            "avg_computed_kv": avg_computed_kv,
            "avg_new_computed_kv": avg_new_computed_kv,
        })

    return result


def analyze_cache(trace,
                  gpu_kv_cache_bytes: int,
                  kv_per_token: int,
                  policy: str = "lru") -> dict:
    stats = trace.simulate_cache(gpu_kv_cache_bytes, kv_per_token, policy)
    stats["num_requests"] = len(trace.cache_results)
    return stats


def _fmt_bytes(n: float) -> str:
    if n >= 1 << 40:
        return f"{n / (1 << 40):.2f} TiB"
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n:.0f} B"


def _fmt_time(ms: float) -> str:
    s = ms / 1000
    if s < 60:
        return f"{s:.1f}s"
    m = s / 60
    if m < 60:
        return f"{m:.1f}m"
    h = m / 60
    return f"{h:.1f}h"


def print_model_info(info: dict) -> None:
    print(f"Model: {info['model_name']} ({info['num_full_attn_layers']}/{info['num_hidden_layers']} full-attn layers, "
          f"{info['num_kv_heads']} KV heads, {info['head_dim']} head dim, "
          f"{info['dtype_bytes']}B dtype)")


def print_analysis(result: dict) -> None:
    print(f"Total requests:        {result['num_requests']}")
    print(f"Avg input length:      {result['input_length']['mean']:.1f}")
    print(f"Avg reused tokens:     {result['computed_tokens']['mean']:.1f}")
    print(f"Avg new computed:      {result['new_computed']['mean']:.1f}")
    print(f"Avg output length:     {result['output_length']['mean']:.1f}")

    if "avg_computed_kv" in result:
        print()
        print(f"KV per token:          {_fmt_bytes(result['kv_per_token'])}")
        print(f"Avg input KV:          {_fmt_bytes(result['input_length']['mean'] * result['kv_per_token'])}")
        print(f"Avg reused KV:         {_fmt_bytes(result['avg_computed_kv'])}")
        print(f"Avg new computed KV:   {_fmt_bytes(result['avg_new_computed_kv'])}")
        print(f"Avg output KV:         {_fmt_bytes(result['output_length']['mean'] * result['kv_per_token'])}")
        print(f"Total KV set:          {_fmt_bytes(result['unique_kv'])}")


def print_cache_analysis(result: dict, show_hist: bool = False) -> None:
    print()
    print("=" * 50)
    print("GPU KV Cache Simulation")
    print("=" * 50)
    print(f"Policy:                {result['policy'].upper()}")
    print(f"GPU KV cache budget:   {_fmt_bytes(result['capacity_bytes'])}")
    print(f"Capacity (blocks):     {result['capacity_blocks']:,}")
    print(f"Total requests:        {result['num_requests']}")
    print(f"Hit rate (blocks):     {result['hit_rate']*100:.2f}%")
    print(f"Total hit blocks:      {result['total_hit_blocks']:,}")
    total_miss = result['total_miss_blocks']
    cold_miss = result['total_cold_miss']
    cap_miss = result['total_capacity_miss']
    cold_pct = cold_miss / total_miss * 100 if total_miss > 0 else 0
    cap_pct = cap_miss / total_miss * 100 if total_miss > 0 else 0
    print(f"Total miss blocks:     {total_miss:,}")
    print(f"  Cold miss:           {cold_miss:,} ({cold_pct:.2f}%)")
    print(f"  Capacity miss:       {cap_miss:,} ({cap_pct:.2f}%)")
    if result.get('reaccess_count'):
        print(f"Reaccess interval (reqs): "
              f"min={result['gap_req_min']:,}, "
              f"p50={result['gap_req_p50']:,}, "
              f"p99={result['gap_req_p99']:,}, "
              f"max={result['gap_req_max']:,}")
        print(f"Reaccess interval (time): "
              f"min={_fmt_time(result['gap_ts_min'])}, "
              f"p50={_fmt_time(result['gap_ts_p50'])}, "
              f"max={_fmt_time(result['gap_ts_max'])}")
        if show_hist:
            _print_histogram(
                [g[1] for g in result.get('_raw_gaps', [])],
                [g[2] for g in result.get('_raw_gaps', [])],
            )


def _print_histogram(gaps, gaps_ts):
    if not gaps:
        return

    print("\nReaccess interval histogram:")
    print("-" * 65)

    # Request gap histogram
    bins_req = [0, 1, 10, 100, 500, 1000, 5000, 10000, 20000]
    labels_req = ["<1    ", "1-10  ", "10-100", "100-500", "500-1K",
                  "1K-5K ", "5K-10K", "10K-20K", ">20K  "]
    counts_req = [0] * len(labels_req)
    for g in gaps:
        for i in range(len(bins_req) - 1):
            if bins_req[i] <= g < bins_req[i+1]:
                counts_req[i] += 1
                break
        else:
            counts_req[-1] += 1

    max_count = max(counts_req) if counts_req else 0
    width = 40
    for label, count in zip(labels_req, counts_req):
        bar_len = int(count / max_count * width) if max_count > 0 else 0
        bar = "█" * bar_len
        pct = count / len(gaps) * 100 if gaps else 0
        print(f"  {label} |{bar:<{width}}| {count:>6,} ({pct:>5.1f}%)")
    print("-" * 65)

    # Time gap histogram
    ts_gaps = [t / 1000 for t in gaps_ts]
    bins_ts = [0, 1, 5, 10, 30, 60, 120, 300, 600, 1200, 3600]
    labels_ts = ["<1s   ", "1-5s  ", "5-10s ", "10-30s", "30-60s",
                 "1-2m  ", "2-5m  ", "5-10m ", "10-30m", "30-60m", ">1h   "]
    counts_ts = [0] * len(labels_ts)
    for g in ts_gaps:
        for i in range(len(bins_ts) - 1):
            if bins_ts[i] <= g < bins_ts[i+1]:
                counts_ts[i] += 1
                break
        else:
            counts_ts[-1] += 1

    max_count_ts = max(counts_ts) if counts_ts else 0
    print("\nReaccess time histogram:")
    print("-" * 65)
    for label, count in zip(labels_ts, counts_ts):
        bar_len = int(count / max_count_ts * width) if max_count_ts > 0 else 0
        bar = "█" * bar_len
        pct = count / len(ts_gaps) * 100 if ts_gaps else 0
        print(f"  {label} |{bar:<{width}}| {count:>6,} ({pct:>5.1f}%)")
    print("-" * 65)



def main(args):
    trace = get_trace(
        trace_name=args.trace,
        local_path=args.local,
    )

    kv_per_token = 0
    if args.model:
        model_info = _load_model_info(args.model, args.dtype_bytes)
        kv_per_token = model_info["kv_bytes_per_token"]
        print_model_info(model_info)
        print()

    result = analyze(trace, kv_per_token=kv_per_token)
    print_analysis(result)

    if args.gpu_kv_cache_size is not None:
        if kv_per_token <= 0:
            print(
                "Error: --gpu-kv-cache-size requires --model to compute KV cache sizes."
            )
            raise SystemExit(1)
        cache_result = analyze_cache(
            trace,
            gpu_kv_cache_bytes=int(args.gpu_kv_cache_size * (1024**3)),
            kv_per_token=kv_per_token,
            policy=args.cache_policy,
        )
        print_cache_analysis(cache_result, show_hist=args.show_hist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM request trace analysis")
    parser.add_argument(
        "--trace",
        default="mooncake",
        choices=list(TRACE_REGISTRY.keys()),
        help="Trace name (default: mooncake)",
    )
    parser.add_argument(
        "--local",
        default=None,
        help="Path to local trace file",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=
        "HuggingFace model name to compute KV cache sizes (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        help="Bytes per KV element (2=FP16/BF16, 1=FP8) (default: 2)",
    )
    parser.add_argument(
        "--gpu-kv-cache-size",
        type=float,
        default=None,
        help="GPU KV cache budget in GB for cache simulation (e.g. 80, 1.5). "
        "Requires --model to compute block sizes.",
    )
    parser.add_argument(
        "--cache-policy",
        default="lru",
        choices=["lru", "fifo"],
        help="Eviction policy for limited cache simulation (default: lru)",
    )
    parser.add_argument(
        "--show-hist",
        action="store_true",
        help="Show reaccess interval histogram",
    )
    args = parser.parse_args()
    main(args)
