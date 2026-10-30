#!/usr/bin/env python3
"""Final V10c builder with the diagnosed 16-file semantic selection budget."""

from __future__ import annotations

import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
V10C_DIR = HERE.parent / "final_graph_v10c_20260610"
if str(V10C_DIR) not in sys.path:
    sys.path.insert(0, str(V10C_DIR))

from step01_build_single_sample_graph_v10c_semantic_budget import (
    PackageSemanticBudgetGraphBuilderV10C,
)


GRAPH_VERSION = "v10c_formal_file16_source_clone_filtered700k_20260612"


class FinalV10CFile16GraphBuilder(PackageSemanticBudgetGraphBuilderV10C):
    """Formal graph builder locked after file-budget coverage diagnosis."""

    max_selected_files = 16

    def build(self):
        graph = super().build()
        graph["graph_version"] = GRAPH_VERSION
        graph["meta"].update(
            {
                "control_profile": GRAPH_VERSION,
                "budget_profile": "formal_package_semantic_budget_file16",
                "max_py_files": self.max_selected_files,
                "file_budget_decision": (
                    "locked_to_16_after_2000_sample_8_16_24_coverage_diagnosis"
                ),
                "source_clone_split_profile": "final_filtered700k_v1",
            }
        )
        return graph


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("src_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(FinalV10CFile16GraphBuilder(args.src_dir).build()["meta"], indent=2))
