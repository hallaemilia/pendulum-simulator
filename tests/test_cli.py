from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pendulum_counterfactuals.cli import build_parser, main
from pendulum_counterfactuals.verification import verify_paired_dataset


class GenerateCommandTest(unittest.TestCase):
    def test_generate_then_verify_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            code = main(
                [
                    "generate",
                    "--output",
                    str(output),
                    "--worlds",
                    "1",
                    "--seed",
                    "51",
                    "--resolutions",
                    "24",
                    "--target-quantiles",
                    "0.5",
                    "--preset",
                    "huawei-grid",
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue((output / "manifest.json").is_file())
            self.assertTrue((output / "pairs.csv").is_file())
            verify_paired_dataset(output)  # Should not raise.

    def test_generate_refuses_an_existing_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "dataset"
            existing.mkdir()
            code = main(
                [
                    "generate",
                    "--output",
                    str(existing),
                    "--worlds",
                    "1",
                    "--seed",
                    "1",
                    "--resolutions",
                    "24",
                ]
            )
            self.assertEqual(code, 1)


class VerifyCommandTest(unittest.TestCase):
    def test_verify_reports_failure_for_a_missing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            code = main(["verify", str(Path(temporary) / "does-not-exist")])
            self.assertEqual(code, 1)

    def test_verify_reports_success_for_a_real_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "dataset"
            main(["generate", "--output", str(output), "--worlds", "1", "--seed", "2", "--resolutions", "24"])
            code = main(["verify", str(output)])
            self.assertEqual(code, 0)


class RenderCommandTest(unittest.TestCase):
    def test_render_writes_a_png_and_reports_descendant_factors(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pendulum.png"
            code = main(
                [
                    "render",
                    "--angle",
                    "0.2",
                    "--light",
                    "1.2",
                    "--resolution",
                    "32",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)
            self.assertTrue(output.is_file())

    def test_render_accepts_huawei_raw_factor_space(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "pendulum.png"
            code = main(
                [
                    "render",
                    "--angle",
                    "10",
                    "--light",
                    "100",
                    "--resolution",
                    "32",
                    "--factor-space",
                    "huawei_raw",
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(code, 0)


class ParserTest(unittest.TestCase):
    def test_parser_requires_a_subcommand(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


if __name__ == "__main__":
    unittest.main()
