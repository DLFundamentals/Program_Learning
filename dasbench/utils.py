from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


PRIVATE_PREFIXES = ("optimum_", "_")


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def load_jsonl(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def public_instance(instance: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in instance.items()
        if not any(key.startswith(prefix) for prefix in PRIVATE_PREFIXES)
    }


def candidate_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "problem": manifest["problem"],
        "metric_definition": manifest["metric_definition"],
        "instance_schema_version": manifest["instance_schema_version"],
        "instance_params": manifest.get("instance_params", {}),
        "distribution_note": (
            "Instances are drawn from an unknown structured distribution. "
            "Infer and exploit recurring structure from the training data."
        ),
    }


def parse_cli_value(text: str) -> Any:
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_key_value_pairs(items: list[str] | None) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected KEY=VALUE argument, got {item!r}.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Expected non-empty key in {item!r}.")
        payload[key] = parse_cli_value(value.strip())
    return payload


def timestamp_token() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
