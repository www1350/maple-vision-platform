"""Independently reopen and verify an exported owned-window replay bundle."""

import argparse
import json
from pathlib import Path

from maple_vision.owned_evidence import verify_owned_window_evidence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    args = parser.parse_args()
    result = verify_owned_window_evidence(args.bundle)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
