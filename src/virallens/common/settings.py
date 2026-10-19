"""Filesystem locations, read from the environment and ``.env`` and validated at startup.

Defaults follow the repository layout and resolve against the working directory, so commands run
from the repository root (as the README and the Docker image do). Each can be overridden by an
environment variable of the same name, e.g. ``DATA_DIR=/srv/filings``.
"""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Paths(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    data_dir: Path = Path("data")
    index_dir: Path = Path("index")
    eval_questions: Path = Path("eval/questions.json")
    fastembed_cache_dir: Path = Path(".cache/fastembed")
    hf_cache_dir: Path = Path(".cache/hf")
