"""
version.py — Single source of truth for DISSOLVE version metadata.

Bump this file when cutting a release; everything else reads from here.
"""

VERSION      = "2.0-gpu"               # semver-style release tag
APP_NAME     = "DISSOLVE"
FULL_NAME    = "Degradation In Silico SOLVEr"
PRODUCT_NAME = "Dissolve 2.0 -- Biodegradable Zinc Alloy Simulator"
BUILD_DATE   = "2026-06-04"
AUTHOR       = "Henaka Ariyarathna"

# Convenience string used by argparse --version and log headers
VERSION_STRING = f"{APP_NAME} {VERSION}"
