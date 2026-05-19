#!/usr/bin/env python3
"""Summarize CALVIN CoT on/off evaluation runs.

The driver writes a TSV manifest with one row per run. This script reads only
real result artifacts from disk and marks missing runs as "未运行".
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


AVG_KEYS = {
    "avg_seq_len",
    "average_seq_len",
    "average_sequence_length",
    "avg_successful_tasks",
    "mean_successful_tasks",
}
CHAIN_KEYS = {
    "chain_sr",
    "chain_success_rate",
    "chain_success_rates",
    "success_rate",
    "success_rates",
}
RAW_RESULT_KEYS = {
    "results",
    "successes",
    "sequence_results",
    "successful_tasks",
    "success_counter",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="TSV manifest emitted by the run script.")
    parser.add_argument("--output", required=True, type=Path, help="Markdown report path.")
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"[-+]?\d+(?:\.\d+)?", value)
        if match:
            return float(match.group(0))
    return None


def _numeric_list(value: Any) -> list[float] | None:
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            number = _as_float(item)
            if number is None and isinstance(item, dict):
                for key in ("success_counter", "successful_tasks", "num_success", "result", "success"):
                    if key in item:
                        number = _as_float(item[key])
                        break
            if number is None:
                return None
            out.append(number)
        return out
    return None


def _walk_dicts(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for value in obj.values():
            yield from _walk_dicts(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from _walk_dicts(value)


def _find_avg(obj: Any) -> float | None:
    for mapping in _walk_dicts(obj):
        for key, value in mapping.items():
            if str(key).lower() in AVG_KEYS:
                number = _as_float(value)
                if number is not None:
                    return number
    return None


def _chain_from_mapping(mapping: dict[str, Any]) -> list[float] | None:
    values: list[float] = []
    for index in range(1, 6):
        candidates = (str(index), f"{index}/5", f"sr{index}", f"chain_sr_{index}", f"chain_sr[{index}]")
        number = None
        for key in candidates:
            if key in mapping:
                number = _as_float(mapping[key])
                break
        if number is None:
            return None
        values.append(number)
    return _normalize_rates(values)


def _normalize_rates(values: list[float]) -> list[float]:
    rates = [float(value) for value in values[:5]]
    if any(abs(value) > 1.5 for value in rates):
        rates = [value / 100.0 for value in rates]
    return rates


def _find_chain(obj: Any) -> list[float] | None:
    if isinstance(obj, dict):
        mapped = _chain_from_mapping(obj)
        if mapped is not None:
            return mapped
    for mapping in _walk_dicts(obj):
        mapped = _chain_from_mapping(mapping)
        if mapped is not None:
            return mapped
        for key, value in mapping.items():
            key_l = str(key).lower()
            if key_l in CHAIN_KEYS or ("chain" in key_l and "sr" in key_l):
                values = _numeric_list(value)
                if values is not None and len(values) >= 5:
                    return _normalize_rates(values)
    values = _numeric_list(obj)
    if values is not None and len(values) == 5:
        return _normalize_rates(values)
    return None


def _looks_like_sequence_counts(values: list[float]) -> bool:
    return len(values) > 5 and all(0 <= value <= 5 and float(value).is_integer() for value in values)


def _find_raw_results(obj: Any) -> list[int] | None:
    values = _numeric_list(obj)
    if values is not None and _looks_like_sequence_counts(values):
        return [int(value) for value in values]

    for mapping in _walk_dicts(obj):
        for key, value in mapping.items():
            key_l = str(key).lower()
            if key_l in RAW_RESULT_KEYS or ("result" in key_l and "sequence" in key_l):
                values = _numeric_list(value)
                if values is not None and _looks_like_sequence_counts(values):
                    return [int(value) for value in values]
    return None


def _find_num_sequences(obj: Any) -> int | None:
    for mapping in _walk_dicts(obj):
        for key, value in mapping.items():
            if str(key).lower() in {"num_sequences", "n_sequences", "total_sequences"}:
                number = _as_float(value)
                if number is not None:
                    return int(number)
    return None


def _metrics_from_raw(raw: list[int]) -> dict[str, Any]:
    count = len(raw)
    chain = [sum(value >= index for value in raw) / count for index in range(1, 6)]
    return {
        "avg_seq_len": sum(raw) / count,
        "chain_sr": chain,
        "num_sequences": count,
        "source": "raw_results",
    }


def _metrics_from_eval_log(eval_log: Path) -> dict[str, Any] | None:
    if not eval_log.is_file():
        return None
    text = eval_log.read_text(encoding="utf-8", errors="ignore")
    matches = re.findall(r"([1-5])/5\s*:\s*([0-9]+(?:\.[0-9]+)?)%", text)
    if not matches:
        return None
    last: dict[int, float] = {}
    for key, value in matches:
        last[int(key)] = float(value) / 100.0
    if not all(index in last for index in range(1, 6)):
        return None
    chain = [last[index] for index in range(1, 6)]
    return {
        "avg_seq_len": sum(chain),
        "chain_sr": chain,
        "num_sequences": None,
        "source": "eval.log",
    }


def load_metrics(result_json: Path, eval_log: Path) -> dict[str, Any]:
    if not result_json.is_file():
        fallback = _metrics_from_eval_log(eval_log)
        if fallback is not None:
            fallback["available"] = True
            return fallback
        return {"available": False, "reason": "missing results.json"}

    try:
        data = json.loads(result_json.read_text(encoding="utf-8-sig"))
    except Exception as exc:  # pragma: no cover - report should keep going.
        return {"available": False, "reason": f"cannot parse results.json: {exc}"}

    raw = _find_raw_results(data)
    if raw:
        metrics = _metrics_from_raw(raw)
        metrics["available"] = True
        return metrics

    avg = _find_avg(data)
    chain = _find_chain(data)
    if chain is not None and avg is None:
        avg = sum(chain)
    if avg is not None or chain is not None:
        metrics = {
            "available": True,
            "avg_seq_len": avg,
            "chain_sr": chain,
            "num_sequences": _find_num_sequences(data),
            "source": "results.json",
        }
        return metrics

    fallback = _metrics_from_eval_log(eval_log)
    if fallback is not None:
        fallback["available"] = True
        fallback["source"] = "eval.log"
        return fallback
    return {"available": False, "reason": "no supported metrics found"}


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [dict(row) for row in reader]


def count_mp4(log_dir: Path) -> int:
    mp4_dir = log_dir / "mp4"
    if not mp4_dir.is_dir():
        return 0
    return sum(1 for _ in mp4_dir.rglob("*.mp4"))


def fmt_avg(value: Any) -> str:
    return "未运行" if value is None else f"{float(value):.3f}"


def fmt_pct(value: Any) -> str:
    return "未运行" if value is None else f"{float(value) * 100:.1f}%"


def fmt_delta(value: float | None, suffix: str = "") -> str:
    if value is None:
        return "未运行"
    return f"{value:+.3f}{suffix}"


def fmt_delta_pp(value: float | None) -> str:
    if value is None:
        return "未运行"
    return f"{value * 100:+.1f} pp"


def enrich_rows(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in rows:
        log_dir = Path(row["log_dir"])
        result_json = Path(row["result_json"])
        eval_log = Path(row["eval_log"])
        metrics = load_metrics(result_json, eval_log)
        row_out: dict[str, Any] = dict(row)
        row_out["metrics"] = metrics
        row_out["mp4_count"] = count_mp4(log_dir)
        enriched.append(row_out)
    return enriched


def metric_for(rows: list[dict[str, Any]], model_id: str, cot: str) -> dict[str, Any] | None:
    for row in rows:
        if row["model_id"] == model_id and row["cot"] == cot:
            metrics = row["metrics"]
            return metrics if metrics.get("available") else None
    return None


def model_order(rows: list[dict[str, Any]]) -> list[tuple[str, str]]:
    seen: set[str] = set()
    order: list[tuple[str, str]] = []
    for row in rows:
        model_id = row["model_id"]
        if model_id not in seen:
            seen.add(model_id)
            order.append((model_id, row["model_label"]))
    return order


def build_report(rows: list[dict[str, Any]], manifest: Path) -> str:
    lines: list[str] = []
    lines.append("# CALVIN Qwen3.5 CoT Evaluation Report")
    lines.append("")
    lines.append(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Manifest: `{manifest}`")
    lines.append("- Missing or unparsable result files are marked as `未运行`.")
    lines.append("")

    lines.append("## 3x2 Metrics")
    lines.append("")
    lines.append(
        "| Model | CoT | Status | Avg Seq Len | SR@1 | SR@2 | SR@3 | SR@4 | SR@5 | MP4 | Result |"
    )
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        metrics = row["metrics"]
        available = metrics.get("available", False)
        chain = metrics.get("chain_sr") if available else None
        chain_values = chain if isinstance(chain, list) and len(chain) >= 5 else [None] * 5
        status = row.get("status", "")
        if not available:
            status = "未运行" if status in {"dry_run", "planned", ""} else status
        result_cell = f"`{row['result_json']}`" if Path(row["result_json"]).is_file() else "未运行"
        lines.append(
            "| {model} | {cot} | {status} | {avg} | {sr1} | {sr2} | {sr3} | {sr4} | {sr5} | {mp4} | {result} |".format(
                model=row["model_label"],
                cot=row["cot"],
                status=status,
                avg=fmt_avg(metrics.get("avg_seq_len") if available else None),
                sr1=fmt_pct(chain_values[0]),
                sr2=fmt_pct(chain_values[1]),
                sr3=fmt_pct(chain_values[2]),
                sr4=fmt_pct(chain_values[3]),
                sr5=fmt_pct(chain_values[4]),
                mp4=row["mp4_count"],
                result=result_cell,
            )
        )
    lines.append("")

    lines.append("## CoT On Minus Off")
    lines.append("")
    lines.append("| Model | Avg Seq Len Δ | SR@1 Δ | SR@2 Δ | SR@3 Δ | SR@4 Δ | SR@5 Δ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    deltas: dict[str, dict[str, Any]] = {}
    for model_id, label in model_order(rows):
        off = metric_for(rows, model_id, "CoT_off")
        on = metric_for(rows, model_id, "CoT_on")
        if not off or not on:
            lines.append(f"| {label} | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |")
            continue
        off_chain = off.get("chain_sr")
        on_chain = on.get("chain_sr")
        if not isinstance(off_chain, list) or not isinstance(on_chain, list):
            lines.append(f"| {label} | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 | 未运行 |")
            continue
        avg_delta = float(on.get("avg_seq_len", 0.0)) - float(off.get("avg_seq_len", 0.0))
        chain_delta = [float(on_chain[i]) - float(off_chain[i]) for i in range(5)]
        deltas[model_id] = {"label": label, "avg": avg_delta, "chain": chain_delta}
        lines.append(
            "| {label} | {avg} | {sr1} | {sr2} | {sr3} | {sr4} | {sr5} |".format(
                label=label,
                avg=fmt_delta(avg_delta),
                sr1=fmt_delta_pp(chain_delta[0]),
                sr2=fmt_delta_pp(chain_delta[1]),
                sr3=fmt_delta_pp(chain_delta[2]),
                sr4=fmt_delta_pp(chain_delta[3]),
                sr5=fmt_delta_pp(chain_delta[4]),
            )
        )
    lines.append("")

    lines.append("## Conclusions")
    lines.append("")
    for model_id, label in model_order(rows):
        delta = deltas.get(model_id)
        if delta is None:
            lines.append(f"- {label}: CoT_on 或 CoT_off 未运行，无法比较完成链长度和各档 SR。")
            continue
        chain_delta = delta["chain"]
        best_index = max(range(5), key=lambda index: chain_delta[index])
        regressions = [
            f"SR@{index + 1}({fmt_delta_pp(value)})"
            for index, value in enumerate(chain_delta)
            if value < -1e-12
        ]
        if chain_delta[best_index] > 0:
            best_text = f"最大提升在 SR@{best_index + 1}，{fmt_delta_pp(chain_delta[best_index])}"
        else:
            best_text = f"没有正向提升；最大变化在 SR@{best_index + 1}，{fmt_delta_pp(chain_delta[best_index])}"
        regression_text = "；退化档位：" + ", ".join(regressions) if regressions else "；未见链长档位退化"
        lines.append(
            f"- {label}: CoT_on - CoT_off 的 avg_seq_len 为 {fmt_delta(delta['avg'])}；{best_text}{regression_text}。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    rows = enrich_rows(read_manifest(args.manifest))
    report = build_report(rows, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report + "\n", encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
