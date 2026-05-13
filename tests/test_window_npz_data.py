from __future__ import annotations

from pathlib import Path

from tsf.data.window_csv import (
    build_fixed_group_split,
    build_fixed_scenario_split,
    fit_window_normalization,
    load_window_dataset,
)


VALIDATION_SCENARIOS = (
    "train_recipe_005",
    "train_recipe_011",
    "train_recipe_017",
    "train_recipe_023",
    "train_recipe_029",
)


def test_indpensim_window_npz_shapes_and_columns() -> None:
    dataset = load_window_dataset(Path("resources/datasets/indpensim"))

    assert dataset.windows.shape == (818, 120, 23)
    assert dataset.targets.shape == (818,)
    assert dataset.target_name == "penicillin_offline"
    assert dataset.feature_names[:3] == ("time_h", "aeration", "agitator_rpm")
    assert dataset.feature_names[-1] == "ammonia_shots"


def test_indpensim_fixed_scenario_split_has_no_eval_leakage() -> None:
    dataset = load_window_dataset(Path("resources/datasets/indpensim"))
    split = build_fixed_scenario_split(dataset, validation_scenarios=VALIDATION_SCENARIOS)

    train_scenarios = {
        dataset.sample_metadata[int(index)]["scenario_name"]
        for index in split.train_indices
    }
    val_scenarios = {
        dataset.sample_metadata[int(index)]["scenario_name"]
        for index in split.val_indices
    }
    test_buckets = {
        dataset.sample_metadata[int(index)]["domain_bucket"]
        for index in split.test_indices
    }

    assert val_scenarios == set(VALIDATION_SCENARIOS)
    assert not train_scenarios.intersection(VALIDATION_SCENARIOS)
    assert test_buckets == {"same_family", "near", "far"}
    assert split.train_indices.size + split.val_indices.size == 541
    assert split.test_indices.size == 277


def test_window_normalization_uses_training_subset() -> None:
    dataset = load_window_dataset(Path("resources/datasets/indpensim"))
    split = build_fixed_scenario_split(dataset, validation_scenarios=VALIDATION_SCENARIOS)

    normalization = fit_window_normalization(
        dataset.windows[split.train_indices],
        dataset.targets[split.train_indices],
    )
    transformed = normalization.transform_windows(dataset.windows[split.train_indices])

    assert transformed.shape == (split.train_indices.size, 120, 23)
    assert abs(float(transformed.reshape(-1, 23).mean())) < 1e-5


def test_thickener_window_npz_uses_dataset_specific_metadata_columns() -> None:
    dataset = load_window_dataset(Path("resources/datasets/thickener_dewatering"))

    assert dataset.windows.shape == (5149, 30, 5)
    assert dataset.targets.shape == (5149,)
    assert dataset.target_name == "underflow_concentration"
    assert "scenario_description" in dataset.sample_metadata[0]
    assert dataset.feature_names == (
        "q_in",
        "p2",
        "p3",
        "phase_pressurizing",
        "phase_discharging",
    )


def test_ladle_window_npz_supports_multi_target_columns() -> None:
    dataset = load_window_dataset(Path("resources/datasets/ladle_preheating"))

    assert dataset.windows.shape == (19840, 60, 14)
    assert dataset.targets.shape == (19840, 5)
    assert dataset.target_names == (
        "温度_t_plus_01",
        "温度_t_plus_02",
        "温度_t_plus_03",
        "温度_t_plus_04",
        "温度_t_plus_05",
    )
    assert "process_id" in dataset.sample_metadata[0]


def test_thickener_fixed_split_uses_train_scenario_validation() -> None:
    dataset = load_window_dataset(Path("resources/datasets/thickener_dewatering"))
    split = build_fixed_scenario_split(
        dataset,
        validation_scenarios=("train_jitter_6",),
        test_domain_buckets=("near", "far"),
    )

    val_scenarios = {
        dataset.sample_metadata[int(index)]["scenario_name"]
        for index in split.val_indices
    }
    train_buckets = {
        dataset.sample_metadata[int(index)]["domain_bucket"]
        for index in split.train_indices
    }
    test_buckets = {
        dataset.sample_metadata[int(index)]["domain_bucket"]
        for index in split.test_indices
    }

    assert val_scenarios == {"train_jitter_6"}
    assert train_buckets == {"train"}
    assert test_buckets == {"near", "far"}
    assert split.train_indices.size == 1355
    assert split.val_indices.size == 271
    assert split.test_indices.size == 3523


def test_ladle_fixed_process_split_has_no_process_leakage() -> None:
    dataset = load_window_dataset(Path("resources/datasets/ladle_preheating"))
    split = build_fixed_group_split(
        dataset,
        validation_column="process_name",
        validation_groups=("process_02", "process_06", "process_11", "process_26", "process_29"),
        train_column="split",
        train_value="train",
        test_group_column="process_name",
        test_groups=("process_13", "process_23", "process_33"),
        scenario_column="process_name",
    )

    train_processes = {
        dataset.sample_metadata[int(index)]["process_name"]
        for index in split.train_indices
    }
    val_processes = {
        dataset.sample_metadata[int(index)]["process_name"]
        for index in split.val_indices
    }
    test_processes = {
        dataset.sample_metadata[int(index)]["process_name"]
        for index in split.test_indices
    }

    assert val_processes == {"process_02", "process_06", "process_11", "process_26", "process_29"}
    assert test_processes == {"process_13", "process_23", "process_33"}
    assert not train_processes.intersection(val_processes)
    assert not train_processes.intersection(test_processes)
    assert split.train_indices.size == 15236
    assert split.val_indices.size == 2500
    assert split.test_indices.size == 2104
