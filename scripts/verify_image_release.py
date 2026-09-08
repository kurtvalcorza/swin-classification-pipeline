"""Verify that runnable image-smoke evidence binds the committed WorkerReleases.

This check is intentionally local: it does not need worker or contract checkouts.
`scripts/verify_release.py` proves manifest/source/document identities; this script
proves that the image evidence committed by the Executor is the evidence cited by
both WorkerReleases and that its immutable image/source identities match them.
"""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


class InputError(ValueError):
    pass


def _pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise InputError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise InputError(f"non-finite JSON number is forbidden: {value}")


def _load(relative: str) -> Any:
    return json.loads(
        (ROOT / relative).read_text(encoding="utf-8"),
        object_pairs_hook=_pairs_no_duplicates,
        parse_constant=_reject_constant,
        parse_float=Decimal,
        parse_int=Decimal,
    )


def _canonical_number(value: Decimal) -> str:
    if not value.is_finite():
        raise InputError("non-finite JSON number is forbidden")
    if value == 0:
        return "0"

    sign_negative, digits_tuple, raw_exponent = value.as_tuple()
    trailing_zeroes = 0
    for digit in reversed(digits_tuple):
        if digit != 0:
            break
        trailing_zeroes += 1
    exponent = int(raw_exponent) + trailing_zeroes
    digits = "".join(str(digit) for digit in digits_tuple).rstrip("0")
    sign = "-" if sign_negative else ""

    if exponent >= 0:
        return sign + digits + ("0" * exponent)
    point = len(digits) + exponent
    if point > 0:
        return sign + digits[:point] + "." + digits[point:]
    return sign + "0." + ("0" * (-point)) + digits


def _canonical_text(value: Any) -> str:
    if value is None:
        return "null"
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, Decimal):
        return _canonical_number(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(_canonical_text(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{" + ",".join(
            json.dumps(key, ensure_ascii=False, separators=(",", ":"))
            + ":"
            + _canonical_text(value[key])
            for key in sorted(value)
        ) + "}"
    raise InputError(f"unsupported JSON value type: {type(value).__name__}")


def _digest_document(value: Any) -> str:
    encoded = _canonical_text(value).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def main() -> int:
    evidence = _load("release/image-smoke-evidence.json")
    validator_release = _load("release/worker-release-validator.json")
    finetuner_release = _load("release/worker-release-finetuner.json")

    images = evidence.get("images")
    validator_image = images.get("validator") if isinstance(images, dict) else None
    finetuner_image = images.get("finetuner") if isinstance(images, dict) else None
    evidence_digest = _digest_document(evidence)

    checks = [
        ("image smoke status PASSED", evidence.get("status") == "PASSED"),
        (
            "image smoke digest cited by validator release",
            evidence_digest in validator_release.get("conformanceEvidenceDigests", []),
        ),
        (
            "image smoke digest cited by finetuner release",
            evidence_digest in finetuner_release.get("conformanceEvidenceDigests", []),
        ),
        (
            "validator image digest bound to evidence",
            isinstance(validator_image, dict)
            and validator_image.get("imageDigest") == validator_release.get("imageDigest"),
        ),
        (
            "validator source revision bound to evidence",
            isinstance(validator_image, dict)
            and validator_image.get("sourceRevision")
            == validator_release.get("sourceRevision"),
        ),
        (
            "finetuner image digest bound to evidence",
            isinstance(finetuner_image, dict)
            and finetuner_image.get("imageDigest") == finetuner_release.get("imageDigest"),
        ),
        (
            "finetuner source revision bound to evidence",
            isinstance(finetuner_image, dict)
            and finetuner_image.get("sourceRevision")
            == finetuner_release.get("sourceRevision"),
        ),
    ]

    failed = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
    print(f"{len(checks) - len(failed)}/{len(checks)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
