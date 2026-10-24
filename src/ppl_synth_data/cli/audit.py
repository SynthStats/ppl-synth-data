from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from ppl_synth_data.cli import audit_app


def _load_manifest(out: Path) -> list[dict]:
    manifest_path = out / "batches.json"
    if not manifest_path.exists():
        raise typer.BadParameter(
            f"No batches.json in {out}: run 'ppldf audit prepare' first",
            param_hint="--out",
        )
    return json.loads(manifest_path.read_text(encoding="utf-8"))


@audit_app.command("prepare")
def audit_prepare(
    data: Annotated[Path, typer.Option(help="Directory with train/val/test JSONL files")],
    out: Annotated[Path, typer.Option(help="Output directory for batch prompt files")],
    batch_size: Annotated[int, typer.Option(help="Entries per batch")] = 20,
    system_prompt_file: Annotated[
        Path | None,
        typer.Option(
            help="Custom system prompt file "
            "(must contain {batch_num}/{start}/{end} template fields)"
        ),
    ] = None,
    dry_run: Annotated[bool, typer.Option(help="Print counts without writing files")] = False,
) -> None:
    """Batch a JSONL dataset into Markdown prompt files ready for GPT submission."""
    from ppl_synth_data.audit.batch import (
        DEFAULT_SYSTEM_PROMPT,
        build_batch_text,
        build_entry_list,
        make_batches,
    )

    system_prompt = (
        system_prompt_file.read_text(encoding="utf-8")
        if system_prompt_file
        else DEFAULT_SYSTEM_PROMPT
    )

    entries = build_entry_list(data)
    batches = make_batches(entries, batch_size=batch_size)

    typer.echo(f"Total entries: {len(entries)}  ({len(batches)} batches of \u2264{batch_size})")
    for b in batches:
        typer.echo(f"  Batch {b.batch_num}: entries {b.start}-{b.end}")

    if dry_run:
        typer.echo("Dry run: no files written.")
        return

    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for b in batches:
        batch_file = out / f"batch_{b.batch_num}.md"
        batch_file.write_text(build_batch_text(b, system_prompt), encoding="utf-8")
        manifest.append(
            {
                "batch_num": b.batch_num,
                "start": b.start,
                "end": b.end,
                "file": batch_file.name,
            }
        )

    (out / "batches.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    typer.echo(f"Wrote {len(batches)} batch files + batches.json to {out}/")


@audit_app.command("submit")
def audit_submit(
    out: Annotated[Path, typer.Option(help="Batch files directory (from 'prepare')")],
    model: Annotated[str, typer.Option(help="GPT model to use")] = "gpt-4.1",
    skip_existing: Annotated[
        bool,
        typer.Option("--skip-existing/--resubmit", help="Skip if job.json already exists"),
    ] = True,
) -> None:
    """Submit all prepared batches to the OpenAI Batch API in one job."""
    from ppl_synth_data.audit.runner import JOB_FILE, submit

    job_path = out / JOB_FILE
    if skip_existing and job_path.exists():
        existing = json.loads(job_path.read_text())
        typer.echo(f"Job already submitted: {existing['job_id']}: use --resubmit to force")
        return

    manifest = _load_manifest(out)
    batch_texts = {
        entry["batch_num"]: (out / entry["file"]).read_text(encoding="utf-8") for entry in manifest
    }

    typer.echo(f"Submitting {len(batch_texts)} batches to OpenAI Batch API (model={model})...")
    job = submit(batch_texts=batch_texts, model=model, out=out)
    typer.echo(f"Job submitted: {job.job_id}")
    typer.echo(f"Requests: {job.request_ids}")
    typer.echo(f"Saved \u2192 {out / JOB_FILE}")


@audit_app.command("collect")
def audit_collect(
    out: Annotated[Path, typer.Option(help="Batch files directory (from 'prepare')")],
) -> None:
    """Poll the submitted batch job, download results, write batch_N_output.txt files."""
    from ppl_synth_data.audit.runner import JOB_FILE, AuditJob, collect

    job_path = out / JOB_FILE
    if not job_path.exists():
        typer.echo(f"No {JOB_FILE} in {out}: run 'ppldf audit submit' first.", err=True)
        raise typer.Exit(1)

    job = AuditJob.load(out)
    typer.echo(f"Polling batch job {job.job_id} (model={job.model})...")
    results = collect(job=job, out=out)

    typer.echo(f"Downloaded {len(results)} batch outputs to {out}/")
    for n in sorted(results):
        typer.echo(f"  batch_{n}_output.txt  ({len(results[n])} chars)")


@audit_app.command("parse")
def audit_parse(
    out: Annotated[Path, typer.Option(help="Batch files directory (from 'collect')")],
    data: Annotated[Path, typer.Option(help="Data directory (for split mapping)")],
) -> None:
    """Parse downloaded batch outputs and write report.md."""
    from ppl_synth_data.audit.parser import parse_log
    from ppl_synth_data.audit.report import render_report
    from ppl_synth_data.audit.runner import load_outputs

    manifest = _load_manifest(out)
    batch_nums = [entry["batch_num"] for entry in manifest]
    outputs = load_outputs(out, batch_nums)

    missing = [n for n in batch_nums if n not in outputs]
    if missing:
        typer.echo(
            f"Missing output for batches {missing}: run 'ppldf audit collect' first.",
            err=True,
        )
        if not outputs:
            raise typer.Exit(1)

    all_results = []
    for n, text in sorted(outputs.items()):
        results = parse_log(text, batch_num=n)
        typer.echo(f"  Batch {n}: {len(results)} entries parsed")
        all_results.extend(results)

    if not all_results:
        typer.echo("No results parsed: check batch output files.", err=True)
        raise typer.Exit(1)

    report_path = out / "report.md"
    report_path.write_text(render_report(all_results, data), encoding="utf-8")

    pass_n = sum(1 for r in all_results if r.verdict == "PASS")
    flag_n = sum(1 for r in all_results if r.verdict == "FLAG")
    crit_n = sum(1 for r in all_results if r.verdict == "CRITICAL")
    typer.echo(
        f"\nTotal: {len(all_results)} entries  "
        f"\u2705 PASS={pass_n}  \u26a0\ufe0f FLAG={flag_n}  \U0001f6a8 CRITICAL={crit_n}"
    )
    typer.echo(f"Report \u2192 {report_path}")


@audit_app.command("all")
def audit_all(
    data: Annotated[Path, typer.Option(help="Directory with train/val/test JSONL files")],
    out: Annotated[
        Path | None,
        typer.Option(help="Output dir (default: artifacts/audit_TIMESTAMP/)"),
    ] = None,
    batch_size: Annotated[int, typer.Option(help="Entries per batch")] = 20,
    model: Annotated[str, typer.Option(help="GPT model to use")] = "gpt-4.1",
    system_prompt_file: Annotated[Path | None, typer.Option()] = None,
    dry_run: Annotated[bool, typer.Option()] = False,
) -> None:
    """Full audit pipeline: prepare \u2192 submit \u2192 collect \u2192 parse \u2192 report."""
    import datetime

    if out is None:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = data.parent / "artifacts" / f"audit_{ts}"
        typer.echo(f"Output dir: {out}")

    audit_prepare(
        data=data,
        out=out,
        batch_size=batch_size,
        system_prompt_file=system_prompt_file,
        dry_run=dry_run,
    )
    if dry_run:
        return

    audit_submit(out=out, model=model)
    audit_collect(out=out)
    audit_parse(out=out, data=data)
