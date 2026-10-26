"""Corrupt-JSON mid-stream raises JournalCorruptionError.

A fully-terminated line that does not decode to valid JSON is a real corruption signal (disk
error, concurrent writer, tampering); distinct from the trailing-partial-line case, which is
silently skipped. The journal must raise loudly rather than drop subsequent events.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ppl_synth_data.eval.persistence import (
    EpisodeJournal,
    FsyncMode,
    JournalCorruptionError,
)
from tests.eval.test_episode_journal_roundtrip import _emit_complete_episode


@pytest.fixture
def corrupted_journal(tmp_path: Path) -> EpisodeJournal:
    journal = EpisodeJournal.for_episode(tmp_path, "ep-corrupt", fsync_mode=FsyncMode.never)
    _emit_complete_episode(journal, n_turns=1)
    # Insert a corrupt line with trailing newline (distinct from torn tail).
    with open(journal.paths.events_path, "a", encoding="utf-8") as fh:
        fh.write("{this is not valid json}\n")
    return journal


def test_read_events_raises_on_corrupt_line(corrupted_journal: EpisodeJournal) -> None:
    with pytest.raises(JournalCorruptionError, match="corrupt event line"):
        corrupted_journal.read_events()


def test_error_names_the_file_path(corrupted_journal: EpisodeJournal) -> None:
    with pytest.raises(JournalCorruptionError, match=str(corrupted_journal.paths.events_path)):
        corrupted_journal.read_events()


def test_torn_tail_still_silently_skipped(tmp_path: Path) -> None:
    """An incomplete trailing line leaves complete journal events readable."""
    journal = EpisodeJournal.for_episode(tmp_path, "ep-torn", fsync_mode=FsyncMode.never)
    _emit_complete_episode(journal, n_turns=1)
    # Partial trailing line (no newline); torn write; must be tolerated.
    with open(journal.paths.events_path, "a", encoding="utf-8") as fh:
        fh.write('{"partial":')
    events = journal.read_events()
    assert events[-1].event_type == "episode_finalised"
