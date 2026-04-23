import csv

from src.utils.results import SUMMARY_FIELDNAMES, append_result_row, compact_result_csv


def test_append_result_row_uses_fixed_summary_schema(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    append_result_row(csv_path, {"backend": "pndm", "dataset": "cifar10", "nfe": 8, "solver_steps": 4})
    append_result_row(
        csv_path,
        {
            "backend": "pndm",
            "dataset": "cifar10",
            "model_asset": "model_a",
            "solver": "euler",
            "schedule": "base",
            "nfe": 8,
            "num_samples": 50000,
            "fid": 3.5,
            "step_methods": ["euler"] * 8,
        },
    )

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    assert fieldnames == list(SUMMARY_FIELDNAMES)
    assert len(rows) == 2
    assert rows[0]["backend"] == "pndm"
    assert rows[0]["dataset"] == "cifar10"
    assert rows[0]["model_asset"] == ""
    assert rows[0]["nfe"] == "8"
    assert rows[1]["solver"] == "euler"
    assert rows[1]["fid"] == "3.5"


def test_compact_result_csv_can_drop_invalid_rows(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    append_result_row(
        csv_path,
        {
            "backend": "pndm",
            "dataset": "cifar10",
            "model_asset": "model_a",
            "solver": "euler",
            "schedule": "base",
            "nfe": 8,
            "num_samples": 50000,
            "fid": 3.5,
        },
    )
    append_result_row(
        csv_path,
        {
            "backend": "pndm",
            "dataset": "cifar10",
            "model_asset": "model_a",
            "solver": "heun2",
            "schedule": "linear",
            "nfe": 8,
            "num_samples": 50000,
            "fid": 99.0,
        },
    )

    compact_result_csv(csv_path, keep_row=lambda row: row["solver"] != "heun2")

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["solver"] == "euler"


def test_append_result_row_replaces_existing_identity_row(tmp_path) -> None:
    csv_path = tmp_path / "results.csv"
    base_row = {
        "backend": "pndm",
        "dataset": "cifar10",
        "model_asset": "model_a",
        "solver": "euler",
        "schedule": "base",
        "nfe": 8,
        "num_samples": 50000,
        "fid": 9.0,
    }
    append_result_row(csv_path, base_row)
    append_result_row(csv_path, {**base_row, "fid": 3.5})

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 1
    assert rows[0]["fid"] == "3.5"
