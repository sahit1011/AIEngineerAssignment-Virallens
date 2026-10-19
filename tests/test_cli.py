"""The command line, driven in-process through ``main()``; no API calls."""

import pytest
from pydantic import ValidationError

from main import main


def test_ingest_then_eval_retrieval(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "index"))
    main(["ingest"])
    assert "Indexed 370 chunks from 5 filings" in capsys.readouterr().out
    main(["eval-retrieval"])
    report = capsys.readouterr().out
    assert "15 questions, k=6" in report
    assert "bm25+dense" in report


def test_eval_retrieval_without_an_index_points_to_ingest(tmp_path, monkeypatch):
    monkeypatch.setenv("INDEX_DIR", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError, match="main.py ingest"):
        main(["eval-retrieval"])


def test_ask_with_an_empty_api_key_fails_before_any_work(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    with pytest.raises(ValidationError, match="groq_api_key"):
        main(["ask", "Are pure omissions actionable?"])
