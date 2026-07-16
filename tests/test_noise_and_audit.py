from pathlib import Path

from lowresource_ir.audit import append_record, verify_log
from lowresource_ir.noise import apply_ocr_noise


def test_ocr_noise_is_deterministic():
    text = "صاف پانی منصوبہ ہر ماہ چیک کیا جاتا ہے۔"
    first = apply_ocr_noise(text, noise_rate=0.2, seed=10)
    second = apply_ocr_noise(text, noise_rate=0.2, seed=10)
    assert first == second
    assert isinstance(first, str)


def test_audit_log_verifies(tmp_path: Path):
    path = tmp_path / "audit.jsonl"
    append_record(path, {"experiment": "unit", "seed": 1})
    append_record(path, {"experiment": "unit", "seed": 2})
    status = verify_log(path)
    assert status["valid"]
    assert status["records"] == 2
