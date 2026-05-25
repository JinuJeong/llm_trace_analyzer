import argparse
import math
import statistics
from trace_utils import get_trace, TRACE_REGISTRY


def _get_llm_config(config):
    if hasattr(config, "text_config"):
        return config.text_config
    return config


def _num_full_attn_layers(config) -> int:
    if hasattr(config, "layer_types"):
        return sum(1 for lt in config.layer_types if lt == "full_attention")
    return config.num_hidden_layers


def kv_bytes_per_token(model_name: str, dtype_bytes: int = 2) -> int:
    from transformers import AutoConfig
    config = _get_llm_config(AutoConfig.from_pretrained(model_name))
    num_layers = _num_full_attn_layers(config)
    num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    return 2 * num_kv_heads * head_dim * num_layers * dtype_bytes


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


def _histogram(data: list, bin_size: int, bar_width: int = 40) -> list:
    if not data:
        return []
    max_val = max(data)
    n_bins = max(1, int(math.ceil(max_val / bin_size)))
    counts = [0] * n_bins
    for v in data:
        idx = min(int(v / bin_size), n_bins - 1)
        counts[idx] += 1
    max_count = max(counts) if counts else 1
    result = []
    for i, c in enumerate(counts):
        bar_len = int(c / max_count * bar_width) if max_count > 0 else 0
        result.append((i * bin_size, (i + 1) * bin_size, c, bar_len))
    return result


def analyze(trace, kv_per_token: int = 0) -> dict:
    in_lens = [r.input_length for r in trace.requests]
    computed_lens = [r.computed_tokens for r in trace.requests]
    new_computed_lens = [max(0, r.input_length - r.computed_tokens) for r in trace.requests]
    out_lens = [r.output_length for r in trace.requests]

    total_computed = sum(computed_lens)
    total_input = sum(in_lens)
    H = total_computed / total_input if total_input > 0 else 0

    per_req_H = []
    for r in trace.requests:
        if r.input_length > 0:
            per_req_H.append(r.computed_tokens / r.input_length)

    result = {
        "num_requests": len(trace.requests),
        "H": H,
        "per_request_H": _stats(per_req_H) if per_req_H else {},
        "input_length": _stats(in_lens),
        "computed_tokens": _stats(computed_lens),
        "new_computed": _stats(new_computed_lens),
        "output_length": _stats(out_lens),
        "total_input_tokens": total_input,
        "total_computed_tokens": total_computed,
        "total_output_tokens": sum(out_lens),
        "unique_kv_tokens": trace.unique_kv_tokens,
        "in_lens": in_lens,
        "computed_lens": computed_lens,
        "new_computed_lens": new_computed_lens,
    }

    if kv_per_token > 0:
        total_input_kv = total_input * kv_per_token
        total_computed_kv = total_computed * kv_per_token
        total_new_computed_kv = sum(new_computed_lens) * kv_per_token
        unique_kv = trace.unique_kv_tokens * kv_per_token
        avg_computed_kv = statistics.mean(computed_lens) * kv_per_token
        avg_new_computed_kv = statistics.mean(new_computed_lens) * kv_per_token
        computed_kv_lens = [c * kv_per_token for c in computed_lens]
        result.update({
            "kv_per_token": kv_per_token,
            "total_input_kv": total_input_kv,
            "total_computed_kv": total_computed_kv,
            "total_new_computed_kv": total_new_computed_kv,
            "unique_kv": unique_kv,
            "avg_computed_kv": avg_computed_kv,
            "avg_new_computed_kv": avg_new_computed_kv,
            "computed_kv_lens": computed_kv_lens,
        })

    return result


def print_analysis(result: dict, hist_bin: int = 0) -> None:
    print(f"Total requests:        {result['num_requests']}")
    print(f"Avg input length:      {result['input_length']['mean']:.1f}")
    print(f"Avg computed tokens:   {result['computed_tokens']['mean']:.1f}")
    print(f"Avg new computed:      {result['new_computed']['mean']:.1f}")
    print(f"Avg output length:     {result['output_length']['mean']:.1f}")

    if "avg_computed_kv" in result:
        print()
        print(f"KV per token:          {_fmt_bytes(result['kv_per_token'])}")
        print(f"Total input KV:        {_fmt_bytes(result['total_input_kv'])}")
        print(f"Total computed KV:     {_fmt_bytes(result['total_computed_kv'])}")
        print(f"Total new computed KV: {_fmt_bytes(result['total_new_computed_kv'])}")
        print(f"Unique KV:             {_fmt_bytes(result['unique_kv'])}")
        print(f"Avg computed KV:       {_fmt_bytes(result['avg_computed_kv'])}")
        print(f"Avg new computed KV:   {_fmt_bytes(result['avg_new_computed_kv'])}")

    if hist_bin > 0:
        print()
        print("New computed tokens histogram:")
        _print_histogram(result["new_computed_lens"], hist_bin)


def _print_histogram(data: list, bin_size: int, bar_width: int = 40) -> None:
    bins = _histogram(data, bin_size, bar_width)
    if not bins:
        print("  No data.")
        return
    for lo, hi, count, bar_len in bins:
        if count == 0:
            continue
        bar = "#" * bar_len
        print(f"  {lo:>8}-{hi:<8} | {bar} {count}")


def _fmt_bytes(n: float) -> str:
    if n >= 1 << 30:
        return f"{n / (1 << 30):.2f} GiB"
    if n >= 1 << 20:
        return f"{n / (1 << 20):.2f} MiB"
    if n >= 1 << 10:
        return f"{n / (1 << 10):.2f} KiB"
    return f"{n:.0f} B"


def main():
    parser = argparse.ArgumentParser(
        description="LLM request trace analysis"
    )
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
        "--hist-bin",
        type=int,
        default=0,
        help="Bin size for new computed tokens histogram (0=off, default: 0)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="HuggingFace model name to compute KV cache sizes (e.g. Qwen/Qwen3-0.6B)",
    )
    parser.add_argument(
        "--dtype-bytes",
        type=int,
        default=2,
        help="Bytes per KV element (2=FP16/BF16, 1=FP8) (default: 2)",
    )
    args = parser.parse_args()

    trace = get_trace(
        trace_name=args.trace,
        local_path=args.local,
    )

    kv_per_token = 0
    if args.model:
        kv_per_token = kv_bytes_per_token(args.model, args.dtype_bytes)
        from transformers import AutoConfig
        config = _get_llm_config(AutoConfig.from_pretrained(args.model))
        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        num_full = _num_full_attn_layers(config)
        print(f"Model: {args.model} ({num_full}/{config.num_hidden_layers} full-attn layers, "
              f"{num_kv_heads} KV heads, {head_dim} head dim, "
              f"{args.dtype_bytes}B dtype)")
        print(f"KV per token: {kv_per_token} B")
        print()

    result = analyze(trace, kv_per_token=kv_per_token)
    print_analysis(result, hist_bin=args.hist_bin)


if __name__ == "__main__":
    main()
