"""Regression tests for precision normalization."""

import unittest

from sim.precision import normalize_precision


class NormalizePrecisionTests(unittest.TestCase):
    def test_normalizes_documented_aliases_case_insensitively(self) -> None:
        self.assertEqual(normalize_precision("BFLOAT16"), "bf16")
        self.assertEqual(normalize_precision("Float16"), "fp16")
        self.assertEqual(normalize_precision("FLOAT32"), "fp32")

    def test_preserves_supported_low_precision_names(self) -> None:
        for precision in ("fp8", "int8", "fp4", "nvfp4", "mxfp4"):
            with self.subTest(precision=precision):
                self.assertEqual(normalize_precision(precision.upper()), precision)

    def test_preserves_unknown_and_empty_string_values_after_lowercasing(self) -> None:
        self.assertEqual(normalize_precision("CustomPrecision"), "customprecision")
        self.assertEqual(normalize_precision(""), "")

    def test_rejects_non_string_values(self) -> None:
        for precision in (None, 16):
            with self.subTest(precision=precision):
                with self.assertRaises(AttributeError):
                    normalize_precision(precision)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
