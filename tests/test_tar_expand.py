"""Tests for tar archive expansion helpers."""

from __future__ import annotations

import gzip
import io
import tarfile
import unittest
from unittest.mock import patch

from geoscouter.core.tar_expand import (
    expand_tar_rows,
    is_tar_filename,
    list_tar_members_from_bytes,
    normalize_geo_url,
)


def _make_tar_bytes(filenames: list[str]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name in filenames:
            data = b"test"
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _make_targz_bytes(filenames: list[str]) -> bytes:
    raw = _make_tar_bytes(filenames)
    out = io.BytesIO()
    with gzip.GzipFile(fileobj=out, mode="wb") as gz:
        gz.write(raw)
    return out.getvalue()


class TestIsTarFilename(unittest.TestCase):
    def test_plain_tar(self):
        self.assertTrue(is_tar_filename("GSE12345_RAW.tar"))
        self.assertTrue(is_tar_filename("bundle.TAR"))

    def test_compressed_tar(self):
        self.assertTrue(is_tar_filename("data.tar.gz"))
        self.assertTrue(is_tar_filename("data.tgz"))

    def test_non_tar(self):
        self.assertFalse(is_tar_filename("sample.fastq.gz"))
        self.assertFalse(is_tar_filename(""))
        self.assertFalse(is_tar_filename("not-tar.txt"))


class TestNormalizeGeoUrl(unittest.TestCase):
    def test_ftp_to_https(self):
        url = "ftp://ftp.ncbi.nlm.nih.gov/geo/series/GSE1/GSE123/file.tar"
        self.assertEqual(
            normalize_geo_url(url),
            "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE1/GSE123/file.tar",
        )


class TestListTarMembersFromBytes(unittest.TestCase):
    def test_plain_tar_lists_files(self):
        data = _make_tar_bytes(["a.txt", "b/fastq.gz", "c.bam"])
        members = list_tar_members_from_bytes(data, "test.tar")
        self.assertEqual(members, ["a.txt", "b/fastq.gz", "c.bam"])

    def test_targz_lists_files(self):
        data = _make_targz_bytes(["x.txt", "y.txt"])
        members = list_tar_members_from_bytes(data, "test.tar.gz")
        self.assertEqual(members, ["x.txt", "y.txt"])

    def test_corrupt_returns_empty(self):
        self.assertEqual(list_tar_members_from_bytes(b"not a tar", "bad.tar"), [])


class TestExpandTarRows(unittest.TestCase):
    def _base_row(self, filename: str) -> dict:
        return {
            "Series": "GSE999",
            "Title": "Example",
            "Supplementary file": filename,
            "Size": "1 Mb",
            "File type/resource": filename,
        }

    @patch("geoscouter.core.tar_expand.list_remote_tar_members")
    def test_expands_tar_and_drops_archive_row(self, mock_list):
        mock_list.return_value = ["sample1.fastq.gz", "sample2.fastq.gz"]
        rows = [
            self._base_row("plain.txt"),
            self._base_row("GSE999_RAW.tar"),
        ]
        url_map = {"GSE999_RAW.tar": "https://example.com/GSE999_RAW.tar"}

        result = expand_tar_rows(rows, "GSE999", url_map, max_depth=0)

        names = [r["Supplementary file"] for r in result]
        self.assertEqual(names, ["plain.txt", "sample1.fastq.gz", "sample2.fastq.gz"])
        self.assertNotIn("GSE999_RAW.tar", names)
        mock_list.assert_called_once()

    @patch("geoscouter.core.tar_expand.list_remote_tar_members")
    def test_keeps_tar_row_when_expansion_fails(self, mock_list):
        mock_list.return_value = []
        rows = [self._base_row("missing.tar")]

        result = expand_tar_rows(rows, "GSE999", {}, max_depth=0)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["Supplementary file"], "missing.tar")

    @patch("geoscouter.core.tar_expand.list_remote_tar_members")
    def test_non_tar_rows_unchanged(self, mock_list):
        rows = [self._base_row("counts.csv")]
        result = expand_tar_rows(rows, "GSE999", {}, max_depth=0)
        self.assertEqual(result, rows)
        mock_list.assert_not_called()


if __name__ == "__main__":
    unittest.main()
