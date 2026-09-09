"""Tests for GSE curation helpers."""

from __future__ import annotations

import unittest

import pandas as pd

from geoscouter.core.curation import (
    build_curated_gse_table,
    classify_health_status,
    extract_panel_size,
    gse_health_status,
    unified_platform_name,
)
from geoscouter.core.filters import filter_gses_by_metadata, metadata_group_options


def _sample_active_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Series": "GSE1",
                "Title": "Healthy lung Xenium 5k panel",
                "Samples": 4,
                "Platforms": "GPL123",
                "Platform_title": "Xenium In Situ Analyzer",
                "Platform_technology": "other",
                "Supplementary file": "a.gz",
            },
            {
                "Series": "GSE2",
                "Title": "Breast cancer CosMx spatial",
                "Samples": 6,
                "Platforms": "GPL456",
                "Platform_title": "CosMx Spatial Molecular Imager",
                "Platform_technology": "other",
                "Supplementary file": "b.gz",
            },
        ]
    )


class TestCuration(unittest.TestCase):
    def test_unified_platform_name_xenium_cosmx(self):
        self.assertEqual(
            unified_platform_name("Xenium In Situ Analyzer", "other"),
            "Xenium",
        )
        self.assertEqual(
            unified_platform_name("CosMx Spatial Molecular Imager", "other"),
            "CosMx",
        )

    def test_classify_health_status_cancer_vs_healthy(self):
        self.assertEqual(classify_health_status("breast cancer tumor"), "cancer")
        self.assertEqual(classify_health_status("healthy control lung"), "healthy")
        self.assertEqual(classify_health_status("unknown specimen"), "other")

    def test_health_override_takes_precedence(self):
        row = pd.Series({"Title": "Healthy lung study"})
        status, source = gse_health_status(row, None, override="cancer")
        self.assertEqual(status, "cancer")
        self.assertEqual(source, "manual")

    def test_extract_panel_size_from_characteristics(self):
        meta = pd.DataFrame(
            [
                {
                    "gsm_id": "GSM1",
                    "gse_id": "GSE1",
                    "panel": "human 5k",
                }
            ]
        )
        row = pd.Series({"Title": "Spatial study"})
        self.assertEqual(extract_panel_size(row, meta), "human 5k")

    def test_metadata_group_options_merges_aliases(self):
        meta = pd.DataFrame(
            [
                {
                    "gse_id": "GSE1",
                    "gsm_id": "GSM1",
                    "cell type": "T cell",
                    "tissue": "lung",
                    "disease": "asthma",
                }
            ]
        )
        options = metadata_group_options([meta])
        self.assertIn("T cell", options["Cell type"])
        self.assertIn("lung", options["Organ / tissue"])
        self.assertIn("asthma", options["Disease"])

    def test_filter_gses_by_metadata_groups(self):
        meta = pd.DataFrame(
            [
                {
                    "gse_id": "GSE1",
                    "gsm_id": "GSM1",
                    "cell type": "T cell",
                    "tissue": "lung",
                },
                {
                    "gse_id": "GSE2",
                    "gsm_id": "GSM2",
                    "cell type": "B cell",
                    "tissue": "liver",
                },
            ]
        )
        meta1 = meta.iloc[:1].copy()
        meta2 = meta.iloc[1:].copy()
        matched = filter_gses_by_metadata(
            [meta1, meta2],
            group_filters={"Cell type": ["T cell"], "Organ / tissue": ["lung"]},
        )
        self.assertEqual(matched, ["GSE1"])

    def test_build_curated_gse_table_columns(self):
        active = _sample_active_df()
        meta = [
            pd.DataFrame(
                [
                    {
                        "gse_id": "GSE1",
                        "gsm_id": "GSM1",
                        "cell type": "epithelial",
                        "tissue": "lung",
                        "disease": "healthy",
                    }
                ]
            )
        ]
        table = build_curated_gse_table(
            active,
            meta,
            gse_selection=["GSE1"],
        )
        self.assertEqual(len(table), 1)
        self.assertEqual(table.iloc[0]["Unified_platform"], "Xenium")
        self.assertIn("Health_status", table.columns)
        self.assertIn("GEO link", table.columns)
        self.assertIn("Panel_size", table.columns)


if __name__ == "__main__":
    unittest.main()
