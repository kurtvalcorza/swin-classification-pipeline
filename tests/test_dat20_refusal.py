"""DAT19/DAT20: a refused BYOD input shows one `Refused: <file>: <reason>` line and no traceback.

Two halves, both read from the generated notebook:
- the router (Section 1, template `dat20-router-refusal/1` from ml-worker
  `integrations/dimer/fleet-specs/notebook-router/`) relays an exception carrying `dimer_refusal` as one
  line, while every other exception keeps its traceback;
- the BYOD cells turn every user-input problem into `BYODRefusal` and remove the earlier run's results and
  `outputs/` files before they stop.
"""

# ruff: noqa: E501  -- refusal messages and parametrized cases read best on one line
from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
import signal
import subprocess
import sys
import types
import zipfile
from multiprocessing.connection import Connection
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials/swin_classification_colab.ipynb"
BYOD_CELL = "swin_classification-16"
CARRIED = (
    "class BYODRefusal(ValueError):\n"
    '    """A refused BYOD input: ``BYODRefusal(file_name, reason_and_fix)``."""\n\n'
    "    dimer_refusal = True\n\n"
    "    def __init__(self, file_name, reason):\n"
    '        super().__init__(f"{file_name}: {reason}")\n'
    "        self.file_name = file_name\n"
    "        self.reason = reason\n\n"
    "    def _render_traceback_(self):\n"
    '        return [f"Refused: {self}"]\n'
)


def _code_cells(path: Path = NOTEBOOK) -> list[dict]:
    return [c for c in json.loads(path.read_text(encoding="utf-8"))["cells"] if c["cell_type"] == "code"]


def _source(cell: dict) -> str:
    return "".join(cell["source"])


def _cell(cell_id: str, path: Path = NOTEBOOK) -> str:
    return next(_source(c) for c in _code_cells(path) if c.get("id") == cell_id)


# --- router, kernel side (portable) -------------------------------------------------------------------------------


def _router_namespace(start: str) -> dict:
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    chunk = source[source.index(start) : source.index("def _is_user_cell():")]
    namespace = {"os": os, "sys": sys, "signal": signal, "subprocess": subprocess, "Connection": Connection, "__name__": "router"}
    exec(compile(chunk, "<router>", "exec"), namespace)
    return namespace


class _Pipe:
    def __init__(self, messages=()):
        self.messages, self.sent = list(messages), []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _kernel_runtime(messages):
    namespace = _router_namespace("class IsolatedCellError(RuntimeError):")
    runtime = object.__new__(namespace["IsolatedRuntime"])
    runtime._send, runtime._recv, runtime._display = _Pipe(), _Pipe(messages), lambda *a, **k: None
    return namespace, runtime


def test_router_relays_a_refusal_as_one_line_without_a_traceback(capsys):
    """Acceptance: fails on main, whose router has no refusal channel."""
    namespace, runtime = _kernel_runtime([("refusal", "hostname: not readable")])
    with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
        runtime.run("x = 1\n")
    assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
    assert caught.value.__cause__ is None and caught.value.__suppress_context__
    assert capsys.readouterr().err == ""
    assert runtime._send.sent == [("run", "x = 1\n")]


def test_router_keeps_the_traceback_of_a_genuine_error(capsys):
    namespace, runtime = _kernel_runtime([("error", "Traceback (most recent call last):\nRuntimeError: bug\n", "RuntimeError: bug")])
    with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
        runtime.run("x = 1\n")
    assert "Traceback (most recent call last)" in capsys.readouterr().err


def test_router_normal_cell_output_is_unchanged(capsys):
    _, runtime = _kernel_runtime([("stream", "stdout", "hello\n"), ("done",)])
    assert runtime.run("print('hello')\n") is None
    assert capsys.readouterr() == ("hello\n", "")


def test_router_template_is_applied():
    source = next(_source(c) for c in _code_cells() if '_WORKER_SOURCE = r"""' in _source(c))
    assert source.index('elif kind == "refusal":') < source.index('elif kind == "error":')
    assert 'if getattr(exc, "dimer_refusal", False):' in source
    header = "Its router refusal channel (DAT20) is applied from ml-worker integrations/dimer/fleet-specs/notebook-router/."
    assert header in (ROOT / "tools" / "build_notebook.py").read_text(encoding="utf-8")


@pytest.mark.skipif(os.name == "nt", reason="the isolated-cell router is POSIX-only (pass_fds, start_new_session)")
def test_router_worker_end_to_end(capsys):
    namespace = _router_namespace('_WORKER_SOURCE = r"""')
    runtime = namespace["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        runtime.run("class R(ValueError):\n    dimer_refusal = True\nkept = 41\n")
        with pytest.raises(namespace["IsolatedCellRefusal"]) as caught:
            runtime.run("raise R('hostname: not readable')\n")
        assert caught.value._render_traceback_() == ["Refused: hostname: not readable"]
        assert "Traceback" not in capsys.readouterr().err
        runtime.run("print(kept + 1)\n")  # the worker namespace survives a refusal
        assert capsys.readouterr().out == "42\n"
        with pytest.raises(namespace["IsolatedCellError"], match="RuntimeError: bug"):
            runtime.run("raise RuntimeError('bug')\n")
        assert "Traceback (most recent call last)" in capsys.readouterr().err
    finally:
        runtime.close()


def test_byod_refusal_is_carried_verbatim():
    assert CARRIED in _cell(BYOD_CELL)


def _refusal_line(exc: BaseException) -> str:
    assert getattr(exc, "dimer_refusal", False), f"not a refusal: {exc!r}"
    (line,) = exc._render_traceback_()
    assert "\n" not in line and "Traceback" not in line
    return line


STEM = BYOD_CELL.rsplit("-", 1)[0]
IMAGE_CELL = next(c.get("id") for c in _code_cells() if _source(c).startswith("USE_BYOD_IMAGE = False"))
MODULE = importlib.import_module("swin_classification_pipeline.pipeline")
EARLIER = {"input_manifest": "old", "training_result": "old", "fresh": "old", "report": "old", "prediction": "old", "payload": "old"}


def _namespace() -> dict:
    ns = {k: v for k, v in vars(MODULE).items() if not k.startswith("__")}
    ns.update(__name__="__main__", Path=Path, shutil=shutil, json=json, zipfile=zipfile)
    return ns


def _seed_outputs(tmp_path: Path) -> None:
    out = tmp_path / "outputs"
    out.mkdir(exist_ok=True)
    for name in ("input_manifest.json", "result.json", "validation_predictions.csv"):
        (out / f"{STEM}_{name}").write_text("{}", encoding="utf-8")
    (out / "unrelated.txt").write_text("kept", encoding="utf-8")


def _section4(monkeypatch, tmp_path, path: str, ns: dict | None = None, source: str | None = None) -> dict:
    monkeypatch.chdir(tmp_path)
    _seed_outputs(tmp_path)
    ns = ns if ns is not None else {**_namespace(), **EARLIER}
    source = (source or _cell(BYOD_CELL)).replace("USE_BYOD = False", "USE_BYOD = True", 1).replace("BYOD_ZIP_PATH = ''", f"BYOD_ZIP_PATH = {path!r}", 1)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, "<section 4>", "exec"), ns)
    return ns


def _png(colour) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (40, 40), colour).save(buffer, format="PNG")
    return buffer.getvalue()


def _zip(path: Path, entries: dict) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def _good_zip(path: Path) -> Path:
    entries = {}
    for split, n in (("train", 4), ("val", 2)):
        for cls, base in (("cool", (20, 40, 200)), ("warm", (200, 60, 20))):
            for i in range(n):
                entries[f"{split}/{cls}/{i}.png"] = _png((base[0] + 3 * i + (50 if split == "val" else 0), base[1], base[2]))
    return _zip(path, entries)


def _cases(tmp_path: Path) -> dict:
    (tmp_path / "hostname").write_text("runtime\n", encoding="utf-8")
    _zip(tmp_path / "flat.zip", {"a.png": _png((1, 2, 3))})
    _zip(tmp_path / "evil.zip", {"../evil.png": _png((1, 2, 3))})
    entries = {f"{s}/cool/{i}.png": _png((10 + i, 40, 200)) for s, k in (("train", 3), ("val", 2)) for i in range(k)}
    entries["train/cool/notes.txt"] = b"stray"
    _zip(tmp_path / "stray.zip", entries)
    return {
        "a text file, not a ZIP (the scripted /etc/hostname run)": ("hostname", "File is not a zip file"),
        "no train/ folder": ("flat.zip", "ZIP must contain train/"),
        "traversing member": ("evil.zip", "absolute or traversing archive member rejected"),
        "stray file in a class folder": ("stray.zip", "VISION_UNRECOGNIZED_IMAGE_EXTENSION: Unrecognized image extension"),
        "missing path": ("absent.zip", "does not exist or is not a file"),
    }


@pytest.mark.parametrize("case", ["a text file, not a ZIP (the scripted /etc/hostname run)", "no train/ folder", "traversing member",
                                  "stray file in a class folder", "missing path"])
def test_dataset_problems_are_one_refused_line_and_remove_earlier_results(monkeypatch, tmp_path, case):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    file_name, rule = _cases(tmp_path)[case]
    ns = {**_namespace(), **EARLIER}
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, str(tmp_path / file_name), ns)
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    assert "run this cell again" in line and "earlier results of this session and their outputs/ files were removed" in line
    assert not set(EARLIER) & set(ns), "an earlier run's results survived the refusal"
    assert sorted(p.name for p in (tmp_path / "outputs").iterdir()) == ["unrelated.txt"]


def test_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): there is no upload dialog outside Google Colab")


@pytest.mark.parametrize(("uploaded", "file_name"), [({}, "(no file)"), ({"a.zip": b"1", "b.zip": b"2"}, "a.zip, b.zip")])
def test_cancelled_or_multiple_upload_is_one_refused_line(monkeypatch, tmp_path, uploaded, file_name):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: uploaded)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith(f"Refused: {file_name}: Upload exactly one ZIP file")


def test_dataset_genuine_internal_error_keeps_its_traceback(monkeypatch, tmp_path):
    ns = {**_namespace(), **EARLIER}

    def broken(*_args, **_kwargs):
        raise RuntimeError("internal bug")

    ns["locate_dataset_root"] = broken
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section4(monkeypatch, tmp_path, str(_good_zip(tmp_path / "good.zip")), ns)
    assert not getattr(caught.value, "dimer_refusal", False)
    assert set(EARLIER) <= set(ns) and (tmp_path / "outputs" / f"{STEM}_result.json").exists()  # nothing removed


def test_good_zip_and_default_sample_still_load(monkeypatch, tmp_path):
    ns = _section4(monkeypatch, tmp_path, str(_good_zip(tmp_path / "good.zip")))
    assert ns["sample_kind"] == "BYOD" and set(ns["inventory"]) == {"cool", "warm"}
    ns = _namespace()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_cell(BYOD_CELL), "<section 4>", "exec"), ns)
    assert ns["sample_kind"] == "synthetic"


@pytest.mark.skipif(not Path("/etc/hostname").is_file(), reason="needs a POSIX /etc/hostname, as in a fresh Colab runtime")
def test_scripted_refusal_input_is_the_real_etc_hostname(monkeypatch, tmp_path):
    """The hosted refusal run sets USE_BYOD=True;BYOD_ZIP_PATH='/etc/hostname': a text file present in every fresh runtime."""
    with pytest.raises(ValueError) as caught:
        _section4(monkeypatch, tmp_path, "/etc/hostname")
    assert _refusal_line(caught.value).startswith("Refused: hostname: File is not a zip file")


# --- Section 9 image ----------------------------------------------------------------------------------------------


class _Fresh:
    def predict(self, paths):
        return [{"predictedClass": "warm", "path": str(paths[0])}]


def _section9(monkeypatch, tmp_path, path: str, source: str | None = None) -> dict:
    """Section 4 on its default path (it defines WORK and the DAT20 helpers), then Section 9 with USE_BYOD_IMAGE."""
    monkeypatch.chdir(tmp_path)
    ns = _namespace()
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(_cell(BYOD_CELL), "<section 4>", "exec"), ns)
    _seed_outputs(tmp_path)
    ns.update(fresh=_Fresh(), prediction="old", payload="old", new_image_origin="old")
    source = (source or _cell(IMAGE_CELL)).replace("USE_BYOD_IMAGE = False", "USE_BYOD_IMAGE = True", 1).replace("BYOD_IMAGE_PATH = ''", f"BYOD_IMAGE_PATH = {path!r}", 1)
    with contextlib.redirect_stdout(io.StringIO()):
        exec(compile(source, "<section 9>", "exec"), ns)
    return ns


@pytest.mark.parametrize(("make", "file_name", "rule"), [
    (lambda p: (p / "fake.png").write_text("text", encoding="utf-8"), "fake.png", "VISION_IMAGE_DECODE_FAILED"),
    (lambda p: (p / "hostname").write_text("text", encoding="utf-8"), "hostname", "unsupported image extension"),
    (lambda p: None, "nope.png", "image not found"),
])
def test_image_problems_are_one_refused_line_and_remove_the_earlier_prediction(monkeypatch, tmp_path, make, file_name, rule):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    make(tmp_path)
    with pytest.raises(ValueError) as caught:
        _section9(monkeypatch, tmp_path, str(tmp_path / file_name))
    line = _refusal_line(caught.value)
    assert line.startswith(f"Refused: {file_name}: ") and rule in line, line
    remaining = sorted(p.name for p in (tmp_path / "outputs").iterdir())
    assert f"{STEM}_result.json" not in remaining and f"{STEM}_validation_predictions.csv" in remaining


def test_image_upload_outside_colab_is_one_refused_line(monkeypatch, tmp_path):
    monkeypatch.setitem(sys.modules, "google.colab", None)
    with pytest.raises(ValueError) as caught:
        _section9(monkeypatch, tmp_path, "")
    assert _refusal_line(caught.value).startswith("Refused: (no file): there is no upload dialog outside Google Colab")


def test_good_image_is_predicted_and_a_prediction_error_keeps_its_traceback(monkeypatch, tmp_path):
    (tmp_path / "ok.png").write_bytes(_png((200, 60, 20)))
    ns = _section9(monkeypatch, tmp_path, str(tmp_path / "ok.png"))
    assert ns["prediction"]["predictedClass"] == "warm" and ns["new_image_origin"] == "user-supplied image (BYOD)"
    source = _cell(IMAGE_CELL).replace("(prediction,) = fresh.predict([new_image_path])", "raise RuntimeError('internal bug')", 1)
    assert "internal bug" in source
    with pytest.raises(RuntimeError, match="internal bug") as caught:
        _section9(monkeypatch, tmp_path, str(tmp_path / "ok.png"), source)
    assert not getattr(caught.value, "dimer_refusal", False)
