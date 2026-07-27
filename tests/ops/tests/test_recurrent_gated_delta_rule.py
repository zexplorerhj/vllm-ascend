"""CLI entry point for the recurrent gated-delta-rule curve benchmark."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from recurrent_gated_delta_rule.benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
