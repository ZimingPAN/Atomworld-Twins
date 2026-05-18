#!/usr/bin/env python3
"""Summarize v108 vacancy-pair rank hard-negative diagnostics.

This wrapper intentionally reuses the v107 rank-summary reader. The v108 eval
path writes richer compact rank payloads, including top-k false-positive and
same-source/same-destination hard-negative composition.
"""

from __future__ import annotations

from diagnose_vacancy_pair_rank_v107 import main


if __name__ == "__main__":
    main()
