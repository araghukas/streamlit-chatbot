"""Misc. utilities for dynamic propagation."""

import sys

def running_in_jupyter() -> bool:
    """Check if the code is running in a Jupyter notebook."""
    return "ipykernel" in sys.modules