"""Run the Q2 study: ``python -m virallens.embedder [--seeds 0 1 2] [--out research/outputs]``."""

import argparse
import logging
from pathlib import Path

from virallens.common.settings import Paths
from virallens.embedder.study import SEEDS, run_study, write_outputs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    parser.add_argument("--out", type=Path, default=Path("research/outputs"))
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    for noisy in ("httpx", "huggingface_hub", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    write_outputs(run_study(Paths(), tuple(args.seeds)), args.out)
    print((args.out / "summary.md").read_text())


if __name__ == "__main__":
    main()
