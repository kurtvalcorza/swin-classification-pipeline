"""Tier C review findings T85-1..T85-4 (2026-10-25)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "tutorials" / "swin_classification_colab.ipynb"


def _nb():
    return json.loads(NB.read_text(encoding="utf-8"))


def _markdown():
    md = "\n".join("".join(c["source"]) for c in _nb()["cells"] if c["cell_type"] == "markdown")
    return " ".join(md.split())


def test_t85_1_worked_answer_matches_the_hosted_run():
    md = _markdown()
    assert "stays clearly above zero" not in md and "cross-entropy keeps falling" not in md
    assert "cross-entropy 0.0077 nats" in md


def test_t85_2_no_stale_no_hosted_run_claim():
    md = _markdown()
    assert "No hosted run of this revision has been recorded yet" not in md
    assert "`fdef1509`" in md


def test_t85_3_dat9_overlap_statement():
    md = _markdown()
    assert "**Overlap with the training data:**" in md and "verified by construction" in md


def test_t85_4_spec_items():
    nb = _nb()
    dimer = nb["metadata"]["dimer"]
    assert dimer["notebook_spec"] == "2.5" and dimer["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    head = "\n".join("".join(c["source"]) for c in nb["cells"][:first_code])
    assert "Notice.** For research, training and evaluation only." in head
    md = _markdown()
    assert "[1] Z. Liu" in md and "`DIMER_ISOLATED_ENV`" in md
