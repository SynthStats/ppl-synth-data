"""Submit audit batches to GPT via the OpenAI Batch API."""

from __future__ import annotations

import json
import logging
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from openai import OpenAI

from ppl_synth_data.config import Settings
from ppl_synth_data.llm.openai_batch import (
    create_batch,
    download_file_content,
    make_responses_batch_line,
    parse_batch_output_jsonl,
    poll_batch_until_terminal,
    upload_batch_file,
    write_batch_jsonl,
)
from ppl_synth_data.pipeline.io import atomic_write_text

log = logging.getLogger(__name__)

DEFAULT_MODEL = "gpt-4.1"
JOB_FILE = "job.json"


@dataclass
class AuditJob:
    job_id: str
    request_ids: list[str]
    model: str

    def save(self, out: Path) -> None:
        out.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out / JOB_FILE, json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, out: Path) -> AuditJob:
        data = json.loads((out / JOB_FILE).read_text(encoding="utf-8"))
        return cls(**data)


def _make_client(settings: Settings | None) -> OpenAI:
    s = settings or Settings()
    return OpenAI(api_key=s.require_openai(), organization=s.openai_org_id)


def submit(
    batch_texts: dict[int, str],
    model: str = DEFAULT_MODEL,
    out: Path | None = None,
    settings: Settings | None = None,
) -> AuditJob:
    """Upload all batch prompts as a single OpenAI Batch API job."""
    client = _make_client(settings)

    lines = [
        make_responses_batch_line(
            custom_id=f"batch-{n}",
            model=model,
            system="",
            user=text,
        )
        for n, text in sorted(batch_texts.items())
    ]

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".jsonl", delete=False, encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)

    try:
        write_batch_jsonl(lines, tmp_path)
        file_id = upload_batch_file(client, tmp_path)
    finally:
        tmp_path.unlink(missing_ok=True)

    job_id = create_batch(
        client,
        input_file_id=file_id,
        metadata={"description": f"dataset-audit {len(lines)} batches"},
    )

    job = AuditJob(
        job_id=job_id,
        request_ids=[f"batch-{n}" for n in sorted(batch_texts)],
        model=model,
    )
    if out is not None:
        job.save(out)
    return job


def collect(
    job: AuditJob,
    out: Path,
    settings: Settings | None = None,
) -> dict[int, str]:
    """Poll until the batch job completes, then download outputs to *out*."""
    client = _make_client(settings)

    batch = poll_batch_until_terminal(client, job.job_id)
    if batch.status != "completed":
        raise RuntimeError(f"Batch job {job.job_id} ended with status={batch.status}")
    if not batch.output_file_id:
        raise RuntimeError(f"Batch job {job.job_id} completed without output_file_id")

    output_jsonl = out / f"batch_output_{job.job_id}.jsonl"
    download_file_content(client, batch.output_file_id, output_jsonl)

    parsed = parse_batch_output_jsonl(output_jsonl)
    if parsed.failures:
        sample = next(iter(parsed.failures.items()))
        raise RuntimeError(
            f"Batch job had {len(parsed.failures)} failed request(s); "
            f"example {sample[0]}={sample[1]}"
        )

    results: dict[int, str] = {}
    for custom_id, text in parsed.successes.items():
        try:
            n = int(custom_id.removeprefix("batch-"))
        except ValueError:
            log.warning("unexpected custom_id in batch output: %r, skipping", custom_id)
            continue
        output_file = out / f"batch_{n}_output.txt"
        atomic_write_text(output_file, text)
        results[n] = text

    return results


def load_outputs(out: Path, batch_nums: list[int]) -> dict[int, str]:
    """Load already-downloaded batch output files from *out*."""
    results: dict[int, str] = {}
    for n in batch_nums:
        p = out / f"batch_{n}_output.txt"
        if p.exists():
            results[n] = p.read_text(encoding="utf-8")
    return results


def outputs_complete(out: Path, batch_nums: list[int]) -> bool:
    return all((out / f"batch_{n}_output.txt").exists() for n in batch_nums)
