from __future__ import annotations

import sys

IS_COLAB_ENV: bool = "google.colab" in sys.modules
GRID_SIZE: int = 15
# PAC5-4: categories with at most 1 of 4 optional features
PAC5_4_CATEGORIES: tuple[str, ...] = ("10000", "11000", "10100", "10010", "10001")
# Default number of worker processes of dataset classes and scripts
DEFAULT_NUM_WORKERS: int = 0
