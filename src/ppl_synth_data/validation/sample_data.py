from __future__ import annotations

import json
from typing import Any


def serialize_sample_data(sample_data: dict[str, Any]) -> str:
    parts = []
    for key, value in sample_data.items():
        key_str = json.dumps(key)
        if hasattr(value, "tolist"):
            parts.append(f"{key_str}: np.array({json.dumps(value.tolist())})")
        else:
            parts.append(f"{key_str}: {json.dumps(value)}")
    return "{" + ", ".join(parts) + "}"
