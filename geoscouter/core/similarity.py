"""Supplementary-file similarity between GEO series."""

from __future__ import annotations

import math
import re
from itertools import combinations

import networkx as nx
import pandas as pd

FILE_RESOURCE_COL = "File type/resource"
SIGNATURE_NODE = "File structure signature"
GEO_SERIES_URL = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"

_GSM_RE = re.compile(r"GSM\d+", re.IGNORECASE)
_GSE_RE = re.compile(r"GSE\d+", re.IGNORECASE)
_SRR_RE = re.compile(r"SRR\d+", re.IGNORECASE)
_SRX_RE = re.compile(r"SRX\d+", re.IGNORECASE)
_LONG_NUM_RE = re.compile(r"\d{6,}")
_PLACEHOLDER_PREFIX_RE = re.compile(r"^(\{(sample|series|run|id)\}[_\.]*)+")
_GENERIC_STEMS = frozenset(
    {
        "matrix",
        "parquet",
        "fastq",
        "fasta",
        "csv",
        "tsv",
        "txt",
        "h5",
        "mtx",
        "json",
        "gz",
        "tar",
        "bz2",
        "zip",
        "data",
        "file",
        "counts",
    }
)

SIMILARITY_HELP = """
**What is the similarity score?**

For each GSE, we collect every **full supplementary filename** (including extension)
from the `File type/resource` column.

Two series are compared with **Jaccard similarity** on those filename sets:

`similarity = |filenames in common| / |all unique filenames across both series|`

Scores range from **0** (no overlap) to **1** (identical supplementary file lists).

**How to interpret thresholds**

| Score | Typical meaning |
|-------|-----------------|
| 0.9+  | Very similar supplementary file layouts (often related studies or reprocessed data) |
| 0.7–0.9 | Substantial overlap; worth comparing side by side |
| 0.5–0.7 | Partial overlap; exploratory link |
| < 0.5 | Weak overlap; may still share biology but different file packaging |

Small changes (e.g. 0.80 vs 0.85) usually mean a few extra or missing files — use the network
plot and hover tooltips to inspect which series cluster together.

**Reference mode**

Optionally pick one GSE as a **reference layout**. The network still shows **general pairwise**
similarity (light gray edges, same Jaccard metric). **Reference edges** connect only the
reference GSE to every other series (star topology), drawn on top in a distinct style so you
can compare each series' supplementary filenames against your chosen favorite layout.
"""

SUPERVISED_SIMILARITY_HELP = """
**Supervised file structure signature**

Pick one or more **training GSEs** whose supplementary file layout you want to use as a template.
The app proposes **match patterns** from their filenames: structural file-type tokens such as
`transcripts.parquet.gz`, `cell_matrix.mtx.gz`, or `xenium.txt.gz` (sample and study IDs are stripped).

Review the pattern table: **include** the rules you want, and **edit the match text** if needed.
**Auto-detected** shows the full normalized filename; **Match pattern** is the structural token
used for scoring.

Every **non-training** GSE is scored with **weighted coverage** of your included rules.
A rule matches when any supplementary filename **contains** the match text (case-insensitive).

`similarity = sum(weights of matched rules) / sum(weights of included rules)`

Training GSEs define the signature but are not ranked against it.
"""


def normalize_filename(name) -> str:
    if pd.isnull(name) or not str(name).strip():
        return ""
    return str(name).strip()


def filename_to_pattern(name) -> str:
    """Normalize a filename to a reusable structure token."""
    raw = normalize_filename(name)
    if not raw:
        return ""
    base = raw.replace("\\", "/").split("/")[-1]
    pattern = _GSM_RE.sub("{sample}", base)
    pattern = _GSE_RE.sub("{series}", pattern)
    pattern = _SRR_RE.sub("{run}", pattern)
    pattern = _SRX_RE.sub("{run}", pattern)
    pattern = _LONG_NUM_RE.sub("{id}", pattern)
    return pattern.lower()


_GENERIC_SUFFIXES = frozenset(
    {"gz", "tar.gz", "txt.gz", "csv.gz", "tsv.gz", "json.gz", "tif.gz", "ome.gz"}
)


def suffix_candidates_from_pattern(pattern: str, min_len: int = 8) -> list[str]:
    """Progressive suffix tokens from a normalized pattern (shortest meaningful first)."""
    if not pattern:
        return []
    seen: set[str] = set()
    ordered: list[str] = []

    def _add(candidate: str) -> None:
        c = candidate.strip().lower()
        if c and "." in c and c not in seen:
            seen.add(c)
            ordered.append(c)

    _add(pattern)
    parts = pattern.split("_")
    for i in range(1, len(parts)):
        _add("_".join(parts[i:]))
    dot_parts = pattern.split(".")
    for i in range(1, len(dot_parts)):
        _add(".".join(dot_parts[i:]))

    meaningful = [
        c for c in sorted(ordered, key=len) if len(c) >= min_len and c not in _GENERIC_SUFFIXES
    ]
    if meaningful:
        return meaningful
    non_generic = [c for c in sorted(ordered, key=len) if c not in _GENERIC_SUFFIXES]
    if non_generic:
        return non_generic
    return sorted(ordered, key=len)


def _strip_placeholder_prefix(pattern: str) -> str:
    """Remove leading sample/run placeholders and separators."""
    return _PLACEHOLDER_PREFIX_RE.sub("", str(pattern).strip().lower())


def _stem_before_extensions(segment: str) -> str:
    """Primary name token before compression or compound extensions."""
    base = str(segment).strip().lower()
    while base.endswith((".gz", ".bz2", ".zip")):
        if base.endswith(".gz"):
            base = base[:-3]
        elif base.endswith(".bz2"):
            base = base[:-4]
        else:
            base = base[:-4]
    return base.split(".")[0] if "." in base else base


def _is_generic_stem(stem: str) -> bool:
    """True when a token looks like a bare extension or numeric sample token."""
    token = str(stem).strip().lower()
    if not token:
        return True
    if token.isdigit():
        return True
    return token in _GENERIC_STEMS


def _is_generic_only_segment(segment: str) -> bool:
    return _is_generic_stem(_stem_before_extensions(segment))


def _longest_non_placeholder_suffix(pattern: str) -> str:
    """Fallback: longest suffix without placeholders that is not generic-only."""
    valid: list[str] = []
    for candidate in suffix_candidates_from_pattern(pattern):
        if "{" in candidate:
            continue
        parts = candidate.split("_")
        dotted = [part for part in parts if "." in part]
        if not dotted:
            continue
        if not _is_generic_only_segment(dotted[-1]):
            valid.append(candidate)
    return valid[-1] if valid else ""


def structural_match_for_pattern(pattern: str) -> str:
    """Extract a structural file-type token from a normalized filename pattern."""
    normalized = str(pattern).strip().lower()
    if not normalized:
        return ""

    remainder = _strip_placeholder_prefix(normalized)
    if not remainder:
        remainder = normalized

    parts = remainder.split("_")
    dotted_indices = [idx for idx, part in enumerate(parts) if "." in part]
    if dotted_indices:
        idx = dotted_indices[-1]
        match = parts[idx]

        def _leading_token(text: str) -> str:
            return text.split("_")[0].split(".")[0]

        while idx > 0 and (
            _is_generic_only_segment(match) or len(_leading_token(match)) < 4
        ):
            idx -= 1
            match = f"{parts[idx]}_{match}"
        if (
            match != remainder
            and remainder.endswith(f"_{match}")
            and not _is_generic_stem(_stem_before_extensions(remainder))
            and len(remainder) - len(match) <= len(_leading_token(remainder)) + 1
        ):
            return remainder
        return match

    fallback = _longest_non_placeholder_suffix(normalized)
    return fallback or remainder or normalized


def structural_match_for_filename(filename: str) -> str:
    """Structural file-type token for supervised matching (sample/study IDs stripped)."""
    pattern = filename_to_pattern(filename)
    if not pattern:
        return ""
    return structural_match_for_pattern(pattern)


def default_match_for_filename(filename: str) -> str:
    """Default structural match token for a supplementary filename."""
    return structural_match_for_filename(filename)


def _extension_variants(text: str) -> set[str]:
    """Return lowercase match strings including optional .gz stripping."""
    base = str(text).strip().lower()
    if not base:
        return set()
    variants = {base}
    if base.endswith(".gz"):
        variants.add(base[:-3])
    elif "." in base:
        variants.add(f"{base}.gz")
    return variants


def _text_matches_needle(needle: str, hay: str) -> bool:
    """True when needle matches hay exactly or as a substring, with .gz equivalence."""
    if not needle or not hay:
        return False
    for needle_v in _extension_variants(needle):
        if hay == needle_v or needle_v in hay:
            return True
        for hay_v in _extension_variants(hay):
            if hay_v == needle_v or needle_v in hay_v:
                return True
    return False


def rule_matches_filenames(match_pattern: str, filenames: set[str]) -> bool:
    """True when any filename contains the match text (normalized, case-insensitive)."""
    needle = str(match_pattern).strip().lower()
    if not needle:
        return False
    for fname in filenames:
        hay_pattern = filename_to_pattern(fname)
        hay_raw = normalize_filename(fname).lower()
        if _text_matches_needle(needle, hay_pattern) or _text_matches_needle(
            needle, hay_raw
        ):
            return True
    return False


def _normalize_training_list(
    series_files: pd.Series, training_gses: list[str]
) -> list[str]:
    training: list[str] = []
    seen: set[str] = set()
    for gse in training_gses:
        g = str(gse).strip().upper()
        if not g or g in seen or g not in series_files.index:
            continue
        seen.add(g)
        training.append(g)
    return training


def discover_pattern_candidates(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> pd.DataFrame:
    """
    Propose editable signature rules grouped by normalized filename pattern.

    Returns a DataFrame with columns:
    Include, Match pattern, Auto-detected, Example filenames, Training GSEs, Weight
    """
    training = _normalize_training_list(series_files, training_gses)
    if not training:
        return pd.DataFrame(
            columns=[
                "Include",
                "Match pattern",
                "Auto-detected",
                "Example filenames",
                "Training GSEs",
                "Weight",
                "rule_id",
            ]
        )

    n = len(training)
    min_support = max(1, math.ceil(min_support_ratio * n))
    groups: dict[str, dict] = {}

    for gse in training:
        seen_auto: set[str] = set()
        for fname in series_files[gse]:
            auto = filename_to_pattern(fname)
            if not auto or auto in seen_auto:
                continue
            seen_auto.add(auto)
            structural = structural_match_for_filename(fname)
            if not structural:
                continue
            bucket = groups.setdefault(
                auto,
                {
                    "match_pattern": structural,
                    "auto_patterns": set(),
                    "examples": [],
                    "gses": set(),
                },
            )
            bucket["auto_patterns"].add(auto)
            if len(bucket["examples"]) < 3:
                bucket["examples"].append(normalize_filename(fname))
            bucket["gses"].add(gse)

    rows = []
    for idx, (auto, data) in enumerate(
        sorted(groups.items(), key=lambda x: (-len(x[1]["gses"]), x[0]))
    ):
        gse_count = len(data["gses"])
        structural = data["match_pattern"]
        rows.append(
            {
                "Include": gse_count >= min_support,
                "Match pattern": structural,
                "Auto-detected": auto,
                "Example filenames": "; ".join(data["examples"]),
                "Training GSEs": f"{gse_count}/{n}",
                "Weight": round(gse_count / n, 3),
                "rule_id": f"rule_{idx}_{auto[:40]}",
            }
        )

    return dedupe_pattern_editor_df(pd.DataFrame(rows))


def geo_series_url(gse: str) -> str:
    """NCBI GEO series accession page URL."""
    acc = str(gse).strip().upper()
    if not acc:
        return ""
    return GEO_SERIES_URL.format(gse=acc)


def _training_gse_fraction(label: str) -> float:
    """Parse '2/3' style Training GSEs column to a fraction."""
    text = str(label).strip()
    if "/" not in text:
        return 0.0
    num, den = text.split("/", 1)
    try:
        n = float(num.strip())
        d = float(den.strip())
        return n / d if d else 0.0
    except ValueError:
        return 0.0


def dedupe_pattern_editor_df(df: pd.DataFrame) -> pd.DataFrame:
    """Merge pattern rows that share the same Match pattern (case-insensitive)."""
    if df is None or df.empty or "Match pattern" not in df.columns:
        return df

    merged: dict[str, dict] = {}
    order: list[str] = []

    for _, row in df.iterrows():
        match = str(row.get("Match pattern", "")).strip()
        if not match:
            continue
        key = match.lower()
        if key not in merged:
            order.append(key)
            merged[key] = {
                "Include": bool(row.get("Include", False)),
                "Match pattern": match,
                "auto_detected": [],
                "examples": [],
                "training_label": str(row.get("Training GSEs", "")),
                "weight": float(row.get("Weight", 1.0)),
            }
        bucket = merged[key]
        bucket["Include"] = bucket["Include"] or bool(row.get("Include", False))
        auto = str(row.get("Auto-detected", "")).strip()
        if auto:
            bucket["auto_detected"].extend(
                part.strip() for part in auto.split(";") if part.strip()
            )
        examples = str(row.get("Example filenames", "")).strip()
        if examples:
            bucket["examples"].extend(
                part.strip() for part in examples.split(";") if part.strip()
            )
        label = str(row.get("Training GSEs", "")).strip()
        if _training_gse_fraction(label) > _training_gse_fraction(bucket["training_label"]):
            bucket["training_label"] = label
            try:
                bucket["weight"] = float(row.get("Weight", bucket["weight"]))
            except (TypeError, ValueError):
                pass

    rows = []
    for idx, key in enumerate(order):
        data = merged[key]
        auto_unique = list(dict.fromkeys(data["auto_detected"]))[:5]
        example_unique = list(dict.fromkeys(data["examples"]))[:5]
        match = data["Match pattern"]
        rows.append(
            {
                "Include": data["Include"],
                "Match pattern": match,
                "Auto-detected": "; ".join(auto_unique),
                "Example filenames": "; ".join(example_unique),
                "Training GSEs": data["training_label"],
                "Weight": data["weight"],
                "rule_id": f"rule_{idx}_{match[:40]}",
            }
        )

    if not rows:
        return df.iloc[0:0].copy()
    return pd.DataFrame(rows)


def parse_manual_patterns(text: str) -> list[str]:
    """Split manual pattern input on newlines and commas."""
    patterns: list[str] = []
    seen: set[str] = set()
    for line in str(text).replace(",", "\n").splitlines():
        pat = line.strip()
        if not pat:
            continue
        key = pat.lower()
        if key in seen:
            continue
        seen.add(key)
        patterns.append(pat)
    return patterns


def manual_pattern_editor_df(patterns: list[str]) -> pd.DataFrame:
    """Build a minimal editable pattern table from manual file-type tokens."""
    rows = []
    for idx, pat in enumerate(patterns):
        rows.append(
            {
                "Include": True,
                "Match pattern": pat,
                "Weight": 1.0,
                "rule_id": f"manual_{idx}_{pat[:40]}",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=["Include", "Match pattern", "Weight", "rule_id"]
        )
    return pd.DataFrame(rows)


def rules_from_manual_patterns(patterns: list[str]) -> list[dict]:
    """Convert manual pattern strings to signature rules (deduped, equal weight)."""
    unique = parse_manual_patterns("\n".join(str(p) for p in patterns))
    return signature_rules_from_dataframe(manual_pattern_editor_df(unique))


def signature_rules_from_dataframe(rules_df: pd.DataFrame) -> list[dict]:
    """Convert edited pattern table to rule dicts for scoring."""
    if rules_df is None or rules_df.empty:
        return []
    rules: list[dict] = []
    for _, row in rules_df.iterrows():
        if not row.get("Include", False):
            continue
        match = str(row.get("Match pattern", "")).strip()
        if not match:
            continue
        try:
            weight = float(row.get("Weight", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        rules.append(
            {
                "rule_id": str(row.get("rule_id", match)),
                "match_pattern": match,
                "weight": weight,
            }
        )
    return rules


def supervised_similarity_from_rules(
    rules: list[dict], filenames: set[str]
) -> tuple[float, list[str], list[str]]:
    """Weighted coverage score and matched / missing rule match strings."""
    if not rules:
        return 0.0, [], []
    total = sum(r["weight"] for r in rules)
    matched = []
    missing = []
    score_weight = 0.0
    for rule in rules:
        pat = rule["match_pattern"]
        if rule_matches_filenames(pat, filenames):
            matched.append(pat)
            score_weight += rule["weight"]
        else:
            missing.append(pat)
    score = score_weight / total if total else 0.0
    return score, matched, missing


def score_supervised_candidates(
    series_files: pd.Series,
    training_gses: list[str],
    rules: list[dict],
) -> tuple[set[str], dict[str, float]]:
    """Score every non-training GSE against the signature rules."""
    training = set(_normalize_training_list(series_files, training_gses))
    scores: dict[str, float] = {}

    for gse in series_files.index:
        if gse in training:
            continue
        score, _, _ = supervised_similarity_from_rules(rules, series_files[gse])
        scores[gse] = score

    return training, scores


def calculate_supervised_edges_from_rules(
    series_files: pd.Series,
    training_gses: list[str],
    rules: list[dict],
    edge_threshold: float = 0.0,
) -> tuple[list[dict], set[str], list[tuple[str, str, float]]]:
    """Star edges for candidates at or above edge_threshold."""
    training, scores = score_supervised_candidates(
        series_files, training_gses, rules
    )
    edges = [
        (SIGNATURE_NODE, gse, score)
        for gse, score in scores.items()
        if score >= edge_threshold
    ]
    return rules, training, edges


def supervised_comparison_table_from_rules(
    series_files: pd.Series,
    training_gses: list[str],
    rules: list[dict],
) -> pd.DataFrame:
    """Rows for each non-training GSE sorted by rule-based supervised similarity."""
    training = set(_normalize_training_list(series_files, training_gses))
    rows = []

    for gse in series_files.index:
        if gse in training:
            continue
        score, matched, missing = supervised_similarity_from_rules(
            rules, series_files[gse]
        )
        rows.append(
            {
                "GSE": gse,
                "GEO link": geo_series_url(gse),
                "Similarity to signature": score,
                "Matching rules": ", ".join(matched) if matched else "",
                "Missing rules": ", ".join(missing) if missing else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "GEO link",
                "Similarity to signature",
                "Matching rules",
                "Missing rules",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Similarity to signature", ascending=False)
        .reset_index(drop=True)
    )


def applied_signature_rules_table(rules: list[dict]) -> pd.DataFrame:
    if not rules:
        return pd.DataFrame(columns=["Match pattern", "Weight"])
    return pd.DataFrame(
        [
            {"Match pattern": r["match_pattern"], "Weight": r["weight"]}
            for r in rules
        ]
    )


def _filename_column(df: pd.DataFrame) -> str:
    if FILE_RESOURCE_COL in df.columns:
        return FILE_RESOURCE_COL
    return "Supplementary file"


def series_filename_sets(df: pd.DataFrame) -> pd.Series:
    """Map each Series to the set of normalized supplementary filenames."""
    col = _filename_column(df)
    return df.groupby("Series")[col].apply(
        lambda files: {normalize_filename(f) for f in files if normalize_filename(f)}
    )


def series_pattern_sets(series_files: pd.Series) -> pd.Series:
    """Map each Series to the set of normalized filename structure patterns."""
    return series_files.apply(
        lambda files: {filename_to_pattern(f) for f in files if filename_to_pattern(f)}
    )


def _jaccard(a: set, b: set) -> float:
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def calculate_similarity_edges(df: pd.DataFrame):
    series_files = series_filename_sets(df)
    edges = [
        (s1, s2, _jaccard(series_files[s1], series_files[s2]))
        for s1, s2 in combinations(series_files.index, 2)
        if len(series_files[s1] | series_files[s2]) > 0
    ]
    graph = nx.Graph()
    graph.add_nodes_from(series_files.index)
    graph.add_weighted_edges_from(e for e in edges if e[2] > 0)
    return series_files, edges, graph


def calculate_reference_edges(
    series_files: pd.Series, reference_gse: str
) -> list[tuple[str, str, float]]:
    """Star edges from reference GSE to every other series (Jaccard on filename sets)."""
    ref = reference_gse.strip().upper()
    if ref not in series_files.index:
        return []
    ref_set = series_files[ref]
    return [
        (ref, other, _jaccard(ref_set, series_files[other]))
        for other in series_files.index
        if other != ref
    ]


def reference_vs_gse_filenames(
    series_files: pd.Series, reference_gse: str, other_gse: str
) -> tuple[float, list[str], list[str], list[str]]:
    """Jaccard score and sorted shared / reference-only / other-only filenames."""
    ref = reference_gse.strip().upper()
    other = other_gse.strip().upper()
    ref_set = series_files.get(ref, set())
    other_set = series_files.get(other, set())
    shared = sorted(ref_set & other_set)
    only_ref = sorted(ref_set - other_set)
    only_other = sorted(other_set - ref_set)
    return _jaccard(ref_set, other_set), shared, only_ref, only_other


def reference_comparison_table(
    series_files: pd.Series, reference_gse: str
) -> pd.DataFrame:
    """Rows for each non-reference GSE, sorted by similarity descending."""
    ref = reference_gse.strip().upper()
    rows = []
    for other in series_files.index:
        if other == ref:
            continue
        score, shared, only_ref, only_other = reference_vs_gse_filenames(
            series_files, ref, other
        )
        rows.append(
            {
                "GSE": other,
                "Similarity to reference": score,
                "Shared files": ", ".join(shared) if shared else "",
                "Only in reference": ", ".join(only_ref) if only_ref else "",
                "Only in GSE": ", ".join(only_other) if only_other else "",
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "Similarity to reference",
                "Shared files",
                "Only in reference",
                "Only in GSE",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Similarity to reference", ascending=False)
        .reset_index(drop=True)
    )


def build_structure_signature(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> tuple[dict[str, float], set[str]]:
    """
    Build weighted pattern signature from training GSEs.

    Returns (pattern -> weight, normalized training GSE ids).
    """
    training = []
    seen: set[str] = set()
    for gse in training_gses:
        g = str(gse).strip().upper()
        if not g or g in seen or g not in series_files.index:
            continue
        seen.add(g)
        training.append(g)

    if not training:
        return {}, set()

    n = len(training)
    min_support = max(1, math.ceil(min_support_ratio * n))
    pattern_freq: dict[str, int] = {}

    for gse in training:
        for fname in series_files[gse]:
            pat = filename_to_pattern(fname)
            if pat:
                pattern_freq[pat] = pattern_freq.get(pat, 0) + 1

    signature = {
        pat: count / n
        for pat, count in pattern_freq.items()
        if count >= min_support
    }
    return signature, set(training)


def supervised_similarity_score(
    signature: dict[str, float], candidate_patterns: set[str]
) -> float:
    """Weighted coverage of signature patterns in a candidate GSE."""
    if not signature:
        return 0.0
    matched = sum(weight for pat, weight in signature.items() if pat in candidate_patterns)
    total = sum(signature.values())
    return matched / total if total else 0.0


def pattern_jaccard(a: set[str], b: set[str]) -> float:
    return _jaccard(a, b)


def calculate_supervised_edges(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> tuple[dict[str, float], set[str], pd.Series, list[tuple[str, str, float]]]:
    """
    Star edges from virtual signature node to non-training GSEs.

    Returns (signature, training_set, pattern_sets, edges).
    """
    signature, training = build_structure_signature(
        series_files, training_gses, min_support_ratio=min_support_ratio
    )
    pattern_sets = series_pattern_sets(series_files)
    edges: list[tuple[str, str, float]] = []

    for gse in series_files.index:
        if gse in training:
            continue
        score = supervised_similarity_score(signature, pattern_sets[gse])
        if score > 0:
            edges.append((SIGNATURE_NODE, gse, score))

    return signature, training, pattern_sets, edges


def supervised_comparison_table(
    series_files: pd.Series,
    training_gses: list[str],
    min_support_ratio: float = 0.8,
) -> pd.DataFrame:
    """Rows for each non-training GSE sorted by supervised similarity."""
    signature, training, pattern_sets, _ = calculate_supervised_edges(
        series_files, training_gses, min_support_ratio=min_support_ratio
    )
    sig_patterns = set(signature)
    rows = []

    for gse in series_files.index:
        if gse in training:
            continue
        patterns = pattern_sets[gse]
        score = supervised_similarity_score(signature, patterns)
        matched = sorted(sig_patterns & patterns)
        missing = sorted(sig_patterns - patterns)
        extra = sorted(patterns - sig_patterns)
        rows.append(
            {
                "GSE": gse,
                "Similarity to signature": score,
                "Matching patterns": ", ".join(matched) if matched else "",
                "Missing signature patterns": ", ".join(missing) if missing else "",
                "Extra patterns": ", ".join(extra) if extra else "",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "GSE",
                "Similarity to signature",
                "Matching patterns",
                "Missing signature patterns",
                "Extra patterns",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values("Similarity to signature", ascending=False)
        .reset_index(drop=True)
    )


def signature_patterns_table(signature: dict[str, float]) -> pd.DataFrame:
    """Display signature patterns and their training-set weights."""
    if not signature:
        return pd.DataFrame(columns=["Pattern", "Weight in training set"])
    rows = [
        {"Pattern": pat, "Weight in training set": weight}
        for pat, weight in sorted(signature.items(), key=lambda x: (-x[1], x[0]))
    ]
    return pd.DataFrame(rows)
