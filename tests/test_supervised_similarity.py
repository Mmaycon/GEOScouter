"""Tests for supervised file-structure similarity."""

from __future__ import annotations

import unittest

import pandas as pd

from geoscouter.core.similarity import (
    SIGNATURE_NODE,
    build_structure_signature,
    calculate_supervised_edges,
    calculate_supervised_edges_from_rules,
    dedupe_pattern_editor_df,
    default_match_for_filename,
    discover_pattern_candidates,
    filename_to_pattern,
    geo_series_url,
    manual_pattern_editor_df,
    parse_manual_patterns,
    rule_matches_filenames,
    rules_from_manual_patterns,
    score_supervised_candidates,
    series_pattern_sets,
    signature_rules_from_dataframe,
    structural_match_for_filename,
    supervised_comparison_table_from_rules,
    supervised_similarity_from_rules,
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

    def test_xenium_suffix(self):
        self.assertEqual(
            default_match_for_filename("GSM8313612.xenium.txt.gz"),
            "xenium.txt.gz",
        )

    def test_transcripts_suffix(self):
        self.assertEqual(
            default_match_for_filename(
                "GSM8313612_0013717_asthma_healthy_transcripts.csv.gz"
            ),
            "transcripts.csv.gz",
        )

    def test_structural_match_keeps_file_type_not_extension(self):
        self.assertEqual(
            structural_match_for_filename(
                "GSM1_0013717_asthma_healthy_transcripts.parquet.gz"
            ),
            "transcripts.parquet.gz",
        )
        self.assertNotEqual(
            structural_match_for_filename(
                "GSM1_0013717_asthma_healthy_transcripts.parquet.gz"
            ),
            "parquet.gz",
        )

    def test_structural_match_strips_study_tokens(self):
        match = structural_match_for_filename(
            "GSM1_0013717_asthma_healthy_cell_matrix.mtx.gz"
        )
        self.assertEqual(match, "cell_matrix.mtx.gz")
        self.assertNotIn("asthma", match)
        self.assertNotIn("healthy", match)

    def test_structural_match_keeps_compound_matrix_name(self):
        self.assertEqual(
            structural_match_for_filename("GSM123.filtered_feature_bc_matrix.h5"),
            "filtered_feature_bc_matrix.h5",
        )


class TestPatternRules(unittest.TestCase):
    def test_rule_matches_by_suffix(self):
        files = {
            "GSM9_0013717_asthma_healthy_transcripts.csv.gz",
            "GSM9.xenium.txt.gz",
        }
        self.assertTrue(rule_matches_filenames("transcripts.csv.gz", files))
        self.assertTrue(rule_matches_filenames("xenium.txt.gz", files))
        self.assertFalse(rule_matches_filenames("matrix.mtx.gz", files))

    def test_rule_matches_extension_variant(self):
        files = {"GSM9_0013717_asthma_healthy_transcripts.csv"}
        self.assertTrue(rule_matches_filenames("transcripts.csv.gz", files))
        self.assertTrue(rule_matches_filenames("transcripts.csv", files))

    def test_discover_keeps_distinct_matrix_types(self):
        series_files = _series_files(
            {
                "GSE1": [
                    "GSM1_cell_matrix.mtx.gz",
                    "GSM1_raw_matrix.mtx.gz",
                    "GSM1_transcripts.csv.gz",
                    "GSM1.xenium.txt.gz",
                ],
            }
        )
        df = discover_pattern_candidates(series_files, ["GSE1"], 0.8)
        self.assertGreaterEqual(len(df), 4)
        auto_patterns = set(df["Auto-detected"])
        match_patterns = set(df["Match pattern"])
        self.assertIn(
            filename_to_pattern("GSM1_cell_matrix.mtx.gz"),
            auto_patterns,
        )
        self.assertIn(
            filename_to_pattern("GSM1_raw_matrix.mtx.gz"),
            auto_patterns,
        )
        self.assertIn("cell_matrix.mtx.gz", match_patterns)
        self.assertIn("raw_matrix.mtx.gz", match_patterns)

    def test_similar_gse_scores_high_with_extension_variants(self):
        series_files = _series_files(
            {
                "GSE1": [
                    "GSM1_cell_matrix.mtx.gz",
                    "GSM1_raw_matrix.mtx.gz",
                    "GSM1_transcripts.csv.gz",
                    "GSM1.xenium.txt.gz",
                ],
                "GSE2": [
                    "GSM9_cell_matrix.mtx.gz",
                    "GSM9_raw_matrix.mtx.gz",
                    "GSM9_transcripts.csv",
                    "GSM9.xenium.txt.gz",
                ],
            }
        )
        rules_df = discover_pattern_candidates(series_files, ["GSE1"], 0.8)
        rules = signature_rules_from_dataframe(rules_df)
        score, matched, missing = supervised_similarity_from_rules(
            rules, series_files["GSE2"]
        )
        self.assertGreaterEqual(score, 0.99)
        self.assertEqual(missing, [])
        self.assertGreaterEqual(len(matched), 4)

    def test_discover_groups_xenium_suffixes(self):
        series_files = _series_files(
            {
                "GSE1": [
                    "GSM8313612.xenium.txt.gz",
                    "GSM8313612_0013717_asthma_healthy_transcripts.csv.gz",
                    "GSM8313612_0013717_asthma_healthy_cell_matrix.mtx.gz",
                ],
                "GSE2": [
                    "GSM8313613.xenium.txt.gz",
                    "GSM8313613_0013718_asthma_healthy_transcripts.csv.gz",
                    "GSM8313613_0013718_asthma_healthy_cell_matrix.mtx.gz",
                ],
            }
        )
        df = discover_pattern_candidates(series_files, ["GSE1", "GSE2"], 0.5)
        matches = set(df["Match pattern"])
        self.assertIn("transcripts.csv.gz", matches)
        self.assertIn("cell_matrix.mtx.gz", matches)
        self.assertNotIn("matrix.mtx.gz", matches)
        self.assertIn("xenium.txt.gz", matches)

    def test_supervised_from_rules(self):
        series_files = _series_files(
            {
                "GSE1": ["GSM1_healthy_transcripts.csv.gz", "GSM1.xenium.txt.gz"],
                "GSE2": ["GSM2_other_transcripts.csv.gz"],
            }
        )
        rules = signature_rules_from_dataframe(
            pd.DataFrame(
                [
                    {
                        "Include": True,
                        "Match pattern": "transcripts.csv.gz",
                        "Weight": 1.0,
                        "rule_id": "r1",
                    },
                    {
                        "Include": True,
                        "Match pattern": "xenium.txt.gz",
                        "Weight": 1.0,
                        "rule_id": "r2",
                    },
                ]
            )
        )
        score, matched, missing = supervised_similarity_from_rules(
            rules, series_files["GSE2"]
        )
        self.assertAlmostEqual(score, 0.5)
        self.assertEqual(matched, ["transcripts.csv.gz"])
        self.assertEqual(missing, ["xenium.txt.gz"])

        _, training, edges = calculate_supervised_edges_from_rules(
            series_files, ["GSE1"], rules
        )
        self.assertEqual(training, {"GSE1"})
        self.assertTrue(any(e[1] == "GSE2" for e in edges))

    def test_dedupe_pattern_editor_df_merges_same_match_pattern(self):
        df = pd.DataFrame(
            [
                {
                    "Include": False,
                    "Match pattern": "matrix.mtx.gz",
                    "Auto-detected": "auto_a",
                    "Example filenames": "file_a",
                    "Training GSEs": "1/2",
                    "Weight": 0.5,
                    "rule_id": "r1",
                },
                {
                    "Include": True,
                    "Match pattern": "matrix.mtx.gz",
                    "Auto-detected": "auto_b",
                    "Example filenames": "file_b",
                    "Training GSEs": "2/2",
                    "Weight": 1.0,
                    "rule_id": "r2",
                },
            ]
        )
        out = dedupe_pattern_editor_df(df)
        self.assertEqual(len(out), 1)
        self.assertTrue(out.iloc[0]["Include"])
        self.assertIn("auto_a", out.iloc[0]["Auto-detected"])
        self.assertIn("auto_b", out.iloc[0]["Auto-detected"])

    def test_rules_from_manual_patterns(self):
        rules = rules_from_manual_patterns(
            ["transcripts.parquet.gz", "xenium.txt.gz", "transcripts.parquet.gz"]
        )
        self.assertEqual(len(rules), 2)
        patterns = {r["match_pattern"] for r in rules}
        self.assertEqual(patterns, {"transcripts.parquet.gz", "xenium.txt.gz"})

    def test_parse_manual_patterns(self):
        parsed = parse_manual_patterns(
            "transcripts.parquet.gz\nxenium.txt.gz, matrix.mtx.gz"
        )
        self.assertEqual(
            parsed,
            ["transcripts.parquet.gz", "xenium.txt.gz", "matrix.mtx.gz"],
        )

    def test_score_all_gses_when_no_training(self):
        series_files = _series_files(
            {
                "GSE1": ["GSM1.xenium.txt.gz"],
                "GSE2": ["GSM2.xenium.txt.gz", "GSM2_transcripts.csv.gz"],
            }
        )
        rules = rules_from_manual_patterns(["xenium.txt.gz", "transcripts.csv.gz"])
        training, scores = score_supervised_candidates(series_files, [], rules)
        self.assertEqual(training, set())
        self.assertEqual(scores["GSE1"], 0.5)
        self.assertEqual(scores["GSE2"], 1.0)

    def test_geo_series_url(self):
        self.assertEqual(
            geo_series_url("gse325935"),
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE325935",
        )

    def test_comparison_table_includes_geo_link(self):
        series_files = _series_files(
            {
                "GSE1": ["GSM1.xenium.txt.gz"],
                "GSE2": ["GSM2.xenium.txt.gz"],
            }
        )
        rules = rules_from_manual_patterns(["xenium.txt.gz"])
        table = supervised_comparison_table_from_rules(series_files, [], rules)
        self.assertIn("GEO link", table.columns)
        self.assertEqual(
            table.loc[table["GSE"] == "GSE2", "GEO link"].iloc[0],
            geo_series_url("GSE2"),
        )


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
