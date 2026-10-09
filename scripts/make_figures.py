#!/usr/bin/env python3
"""Entry point for figure generation.

The published manuscript figures are retained in the repository. The complete
figure-generation workflow remains in the accompanying analysis notebook.
"""
from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="CircEnvFormer figure-generation entry point.")
    parser.add_argument("--list", action="store_true", help="List the main manuscript figures.")
    args = parser.parse_args()

    if args.list:
        for name in [
            "scheme.png",
            "WCSS_with_Aurora_errorbars.png",
            "clusters_environment.png",
            "Z500_SLP_4clusters.png",
            "Q850_winds_4clusters.png",
        ]:
            print(name)
    else:
        print("See notebooks/working_example.ipynb for the analysis and plotting workflow used in the manuscript.")


if __name__ == "__main__":
    main()
