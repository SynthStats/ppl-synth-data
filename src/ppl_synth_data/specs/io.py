from __future__ import annotations

import json
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from ppl_synth_data.specs.models import EnvSpec, FourFamilySpec

_ENVSPEC_READER: TypeAdapter[EnvSpec | FourFamilySpec] = TypeAdapter(EnvSpec | FourFamilySpec)


def read_compile_inputs(path: Path) -> tuple[list[EnvSpec | FourFamilySpec], list[dict]]:
    """Keep invalid typed declarations as refusals without repairing their fields."""
    specs = []
    refusals = []
    seen = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        raw = json.loads(line)
        env_id = raw.get("env_id")
        if env_id in seen:
            raise ValueError("Stage B requires unique env_id values across EnvSpecs")
        seen.add(env_id)
        if "features" not in raw:
            specs.append(_ENVSPEC_READER.validate_python(raw))
            continue
        try:
            specs.append(FourFamilySpec.model_validate(raw))
        except ValidationError as exc:
            reasons = []
            for error in exc.errors(include_url=False, include_context=False):
                message = error["msg"].removeprefix("Value error, ")
                if message.startswith("/"):
                    pointer = message.partition(":")[0]
                else:
                    parts = []
                    cursor = raw
                    for part in error["loc"]:
                        if isinstance(cursor, dict) and part in (
                            cursor.get("dist"),
                            cursor.get("family"),
                        ):
                            continue  # Pydantic union tags are not document keys.
                        parts.append(str(part).replace("~", "~0").replace("/", "~1"))
                        if isinstance(cursor, dict):
                            cursor = cursor.get(part)
                        elif (
                            isinstance(cursor, list)
                            and isinstance(part, int)
                            and part < len(cursor)
                        ):
                            cursor = cursor[part]
                        else:
                            cursor = None
                    pointer = "/" + "/".join(parts)
                reasons.append(
                    {
                        "json_pointer": pointer,
                        "missing_semantics": message,
                        "intended_mechanism": json.dumps(raw, ensure_ascii=False, sort_keys=True),
                    }
                )
            scenario = raw.get("scenario", {})
            refusals.append(
                {
                    "env_id": env_id,
                    "source_row": line_number,
                    "source_row_sha256": scenario.get("source_row_sha256"),
                    "scenario_intent": scenario.get("intent"),
                    "extractor": "ppl-synth-data Stage B typed validation",
                    "refusals": reasons,
                }
            )
    return specs, refusals


def read_envspecs(path: Path) -> list[EnvSpec | FourFamilySpec]:
    if not path.exists():
        raise FileNotFoundError(f"required artifact missing: {path}")

    rows: list[EnvSpec | FourFamilySpec] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rows.append(_ENVSPEC_READER.validate_python(json.loads(line)))
    return rows
