"""Tests for supervised file-structure similarity."""

from __future__ import annotations

import unittest

import pandas as pd

from geoscouter.core.similarity import (
    SIGNATURE_NODE,
    build_structure_signature,
    calculate_supervised_edges,
    filename_to_pattern,
    series_pattern_sets,
    supervised_similarity_score,
)


def _series_files(rows: dict[str, list[str]]) -> pd.Series:
    return pd.Series(rows, dtype=object)


class TestFilenameToPattern(unittest.TestCase):
    def test_replaces_geo_ids(self):
        self.assertEqual(
            filename_to_pattern("GSM123456_R1.fastq.gz"),
            "{sample}_r1.fastq.gz",
        )
        self.assertEqual(
            filename_to_pattern("GSE237849_gene_counts.csv"),
            "{series}_gene_counts.csv",
        )

    def test_keeps_static_names(self):
        self.assertEqual(filename_to_pattern("barcodes.tsv.gz"), "barcodes.tsv.gz")


class TestStructureSignature(unittest.TestCase):
    def setUp(self):
        self.series_files = _series_files(
            {
                "GSE1": ["GSM1_R1.fastq.gz", "GSM1_R2.fastq.gz", "counts.csv"],
                "GSE2": ["GSM2_R1.fastq.gz", "GSM2_R2.fastq.gz", "counts.csv"],
                "GSE3": ["GSM3_R1.fastq.gz", "other.txt"],
            }
        )

    def test_signature_keeps_shared_patterns(self):
        signature, training = build_structure_signature(
            self.series_files, ["GSE1", "GSE2"], min_support_ratio=0.8
        )
        self.assertEqual(training, {"GSE1", "GSE2"})
        self.assertIn("{sample}_r1.fastq.gz", signature)
        self.assertIn("{sample}_r2.fastq.gz", signature)
        self.assertIn("counts.csv", signature)

    def test_supervised_score_for_matching_gse(self):
        signature, _ = build_structure_signature(
            self.series_files, ["GSE1", "GSE2"], min_support_ratio=0.8
        )
        patterns = series_pattern_sets(self.series_files)["GSE2"]
        score = supervised_similarity_score(signature, patterns)
        self.assertEqual(score, 1.0)

    def test_supervised_edges_skip_training(self):
        _, training, _, edges = calculate_supervised_edges(
            self.series_files, ["GSE1", "GSE2"], min_support_ratio=0.8
        )
        self.assertEqual(training, {"GSE1", "GSE2"})
        target_gses = {e[1] for e in edges}
        self.assertIn("GSE3", target_gses)
        self.assertNotIn("GSE1", target_gses)
        self.assertTrue(all(e[0] == SIGNATURE_NODE for e in edges))


if __name__ == "__main__":
    unittest.main()
