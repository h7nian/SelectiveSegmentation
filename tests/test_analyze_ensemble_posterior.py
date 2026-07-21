import json
from scripts.analyze.ensemble import (
    BASELINE_ARTIFACT_TYPE,
    MAIN_ARTIFACT_TYPE,
    METHODS,
    analyze,
)
from selectseg.artifacts import sha256_file


def _write_artifact(root, dataset, condition, artifact_type, rows):
    directory = root / dataset / condition / "run"
    directory.mkdir(parents=True)
    records = directory / "records.jsonl"
    records.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest = {
        "artifact_type": artifact_type,
        "dataset": dataset,
        "condition": condition,
        "num_rows": len(rows),
    }
    if artifact_type == MAIN_ARTIFACT_TYPE:
        manifest["jsonl_sha256"] = sha256_file(records)
    else:
        manifest["records_sha256"] = sha256_file(records)
    (directory / "manifest.json").write_text(json.dumps(manifest))


def test_complete_ten_condition_analysis_is_symmetric_and_predeclared(tmp_path):
    datasets = ["pet", "kvasir", "fives", "isic", "tn3k"]
    models = ["clipseg-target", "deeplabv3-target"]
    main_root = tmp_path / "main"
    baseline_root = tmp_path / "baseline"
    main_fields = {
        field for field, _ in METHODS if not field.startswith("confidence_ensemble")
        and field != "confidence_threshold_iou_stability"
    }
    auxiliary_fields = {
        field for field, _ in METHODS if field not in main_fields
    }
    for dataset in datasets:
        for model in models:
            main_rows = []
            baseline_rows = []
            for index in range(5):
                risks = {
                    "risk_dice": index / 10,
                    "risk_nhd": (4 - index) / 10,
                    "risk_nhd95": abs(2 - index) / 10,
                }
                main_rows.append(
                    {
                        "schema_version": 2,
                        "sample_id": f"sample-{index}",
                        "image_index": index,
                        **risks,
                        **{field: float(index + offset / 100) for offset, field in enumerate(sorted(main_fields))},
                    }
                )
                baseline_rows.append(
                    {
                        "schema_version": 1,
                        "sample_id": f"sample-{index}",
                        "image_index": index,
                        **risks,
                        **{field: float(4 - index + offset / 100) for offset, field in enumerate(sorted(auxiliary_fields))},
                    }
                )
            _write_artifact(main_root, dataset, model, MAIN_ARTIFACT_TYPE, main_rows)
            _write_artifact(
                baseline_root, dataset, model, BASELINE_ARTIFACT_TYPE, baseline_rows
            )

    contract = {
        "analysis_id": "test",
        "status": "predeclared-before-reading-ensemble-baseline-outputs",
        "conditions": {"datasets": datasets, "models": models, "count": 10},
        "primary_comparisons": [
            {"risk": "dice", "left": "confidence_ensemble_q_dice", "right": "confidence_dice_m32"},
            {"risk": "nhd", "left": "confidence_ensemble_q_nhd", "right": "confidence_nhd_m32"},
            {"risk": "nhd95", "left": "confidence_ensemble_q_nhd95", "right": "confidence_nhd95_m32"},
        ],
        "reporting": {
            "bootstrap_resamples": 10,
            "aurc_scale": 100,
            "tie_policy": "test ties",
        },
        "interpretation_gates": {
            "ensemble_supports_dice_expressivity_hypothesis": {"required_condition_wins": 7},
            "level_set_remains_preferred_for_nhd": {"required_condition_wins": 7},
        },
    }
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(contract))
    result = analyze(
        contract_path,
        main_root,
        baseline_root,
        bootstrap_resamples=10,
    )
    assert len(result["conditions"]) == 10
    assert set(result["summary"]).issuperset(
        {"risk_dice", "risk_nhd", "risk_nhd95", "dice_gate_passed"}
    )
    for condition in result["conditions"]:
        for risk in ("risk_dice", "risk_nhd", "risk_nhd95"):
            assert len(condition["risks"][risk]["methods"]) == len(METHODS)
