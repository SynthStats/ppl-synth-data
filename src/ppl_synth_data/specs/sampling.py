from __future__ import annotations

import hashlib
from typing import Any

from ppl_synth_data.specs.models import Complexity
from ppl_synth_data.specs.registry import SUPPORTED_FAMILIES

DOMAINS = (
    "biology",
    "healthcare",
    "finance",
    "technology",
    "environmental",
    "social_science",
    "physics",
    "sports",
    "psychology",
    "ecology",
    "engineering",
    "economics",
)

COMPLEXITIES = (Complexity.simple, Complexity.medium, Complexity.complex)


def sample_domain_family(seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
    domain_idx = int.from_bytes(digest[:8], "big") % len(DOMAINS)
    family_idx = int.from_bytes(digest[8:16], "big") % len(SUPPORTED_FAMILIES)
    return DOMAINS[domain_idx], SUPPORTED_FAMILIES[family_idx]


def build_generation_plan(
    target_count: int,
    families: tuple[str, ...] | None = None,
    domains: tuple[str, ...] | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    fams = families or SUPPORTED_FAMILIES
    doms = domains or DOMAINS
    complexities = COMPLEXITIES

    # complexity-first ordering: first N slots cycle through all families
    # before moving to the next complexity level. This means limit=13
    # gives one record per family (all at "simple").
    base: list[dict[str, Any]] = []
    for cx in complexities:
        for fam in fams:
            digest = hashlib.sha256(f"{seed}:{fam}:{cx}".encode()).digest()
            dom_idx = int.from_bytes(digest[:8], "big") % len(doms)
            base.append(
                {
                    "family": fam,
                    "complexity": cx.value,
                    "domain": doms[dom_idx],
                    "diversity_seed": int.from_bytes(digest[8:16], "big"),
                }
            )

    plan = base[:target_count]

    idx = 0
    while len(plan) < target_count:
        fam = fams[idx % len(fams)]
        digest = hashlib.sha256(f"{seed}:fill:{idx}".encode()).digest()
        dom_idx = int.from_bytes(digest[:4], "big") % len(doms)
        cx_idx = int.from_bytes(digest[4:8], "big") % len(complexities)
        plan.append(
            {
                "family": fam,
                "complexity": complexities[cx_idx].value,
                "domain": doms[dom_idx],
                "diversity_seed": int.from_bytes(digest[8:16], "big"),
            }
        )
        idx += 1

    return plan


def coverage_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[tuple[str, str], int] = {}
    domain_counts: dict[str, int] = {}

    for rec in records:
        family = rec.get("generative_family", rec.get("family", "unknown"))
        complexity = rec.get("complexity", "unknown")
        domain = rec.get("domain", "unknown")

        key = (family, complexity)
        counts[key] = counts.get(key, 0) + 1
        domain_counts[domain] = domain_counts.get(domain, 0) + 1

    gaps = [
        (fam, cx.value)
        for fam in SUPPORTED_FAMILIES
        for cx in COMPLEXITIES
        if (fam, cx.value) not in counts
    ]

    return {
        "total": len(records),
        "family_complexity_counts": {f"{k[0]}:{k[1]}": v for k, v in sorted(counts.items())},
        "domain_counts": dict(sorted(domain_counts.items())),
        "gaps": [f"{fam}:{cx}" for fam, cx in gaps],
        "n_gaps": len(gaps),
    }
