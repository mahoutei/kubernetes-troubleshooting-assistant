from pathlib import Path

from evals.validate_questions import certification_errors, main

QUESTIONS = Path("evals/questions.jsonl")


def _blank_cert(tmp_path) -> Path:
    cert = tmp_path / "CERTIFICATION.md"
    cert.write_text(
        "# Gold Set Certification\n\n"
        "- Reviewer: ____________________\n"
        "- Date: ____________________\n"
        "- Notes: ____________________\n"
    )
    return cert


def test_valid_gold_set_passes(capsys):
    assert main([str(QUESTIONS)]) == 0
    assert "ok: 60 questions" in capsys.readouterr().out


def test_unsigned_certification_is_rejected(tmp_path, capsys):
    cert = _blank_cert(tmp_path)
    assert main([str(QUESTIONS), "--require-certified", "--certification", str(cert)]) == 1
    out = capsys.readouterr().out
    assert "Reviewer is blank" in out
    assert "Date is blank" in out


def test_signed_certification_passes(tmp_path):
    cert = tmp_path / "CERTIFICATION.md"
    cert.write_text(
        "# Gold Set Certification\n\n"
        "- Reviewer: Ada Lovelace\n"
        "- Date: 2026-10-30\n"
        "- Notes: reviewed all 60 records\n"
    )
    assert main([str(QUESTIONS), "--require-certified", "--certification", str(cert)]) == 0


def test_certification_helpers_treat_underscores_as_blank(tmp_path):
    cert = _blank_cert(tmp_path)
    errors = certification_errors(cert)
    assert "certification: Reviewer is blank" in errors
    assert "certification: Notes is blank" in errors
