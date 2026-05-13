from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


SEMANTIC_PROMPT_VERSION = "tsf-semantic-cards-v1"
SEMANTIC_CARD_INPUT_FIELDS = (
    "variable_name",
    "canonical_name",
    "unit",
    "physical_meaning",
    "process_role",
    "information_role",
    "temporal_relation_to_target",
    "relations_to_other_variables",
)
SEMANTIC_CARD_FIELDS = (
    *SEMANTIC_CARD_INPUT_FIELDS,
    "embedding_text",
)


@dataclass(frozen=True)
class VariableSpec:
    name: str
    unit: str = ""
    sampling_frequency: str = ""
    availability: str = "online"
    role: str = ""
    description: str = ""

    def to_prompt_block(self) -> str:
        fields = [
            f"- name: {self.name}",
            f"  unit: {self.unit or 'unknown'}",
            f"  sampling_frequency: {self.sampling_frequency or 'unknown'}",
            f"  availability: {self.availability}",
            f"  role: {self.role or 'unknown'}",
            f"  description: {self.description or 'not provided'}",
        ]
        return "\n".join(fields)

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "VariableSpec":
        return cls(
            name=str(payload["name"]),
            unit=str(payload.get("unit", "")),
            sampling_frequency=str(payload.get("sampling_frequency", "")),
            availability=str(payload.get("availability", "online")),
            role=str(payload.get("role", payload.get("raw_role", ""))),
            description=str(payload.get("description", payload.get("raw_description", ""))),
        )

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "unit": self.unit,
            "sampling_frequency": self.sampling_frequency,
            "availability": self.availability,
            "role": self.role,
            "description": self.description,
        }


@dataclass(frozen=True)
class ForecastTaskSpec:
    dataset: str
    target: str
    horizon: str
    online_boundary: str
    process_background: str
    variables: tuple[VariableSpec, ...]
    task_type: str = ""
    input_window_definition: str = ""
    split_description: str = ""
    scenario_descriptions: tuple[str, ...] = ()

    def build_semantic_prompt(self) -> str:
        variable_blocks = "\n".join(variable.to_prompt_block() for variable in self.variables)
        optional_context = []
        if self.task_type:
            optional_context.append(f"Task type: {self.task_type}")
        if self.input_window_definition:
            optional_context.append(f"Input window definition: {self.input_window_definition}")
        if self.split_description:
            optional_context.append(f"Train/eval split description: {self.split_description}")
        if self.scenario_descriptions:
            scenario_block = "\n".join(f"- {item}" for item in self.scenario_descriptions)
            optional_context.append(
                "Dataset split / process context for protocol boundary only:\n"
                f"{scenario_block}"
            )
        optional_context_text = "\n".join(optional_context)
        if optional_context_text:
            optional_context_text = f"{optional_context_text}\n"
        return (
            "You are generating task-semantic variable cards for the TSF method: "
            "LLM-Guided Task-Semantic Field Factorization for industrial time-series forecasting.\n\n"
            f"Prompt version: {SEMANTIC_PROMPT_VERSION}\n\n"
            "Your job is to define the meaning of each input variable for this specific "
            "forecasting task. Do not predict numeric values. Do not infer from test "
            "statistics, labels, future windows, model errors, or any information "
            "unavailable before training.\n\n"
            f"Dataset: {self.dataset}\n"
            f"Prediction target: {self.target}\n"
            f"Forecast horizon / sampling task: {self.horizon}\n"
            f"Online information boundary: {self.online_boundary}\n"
            f"Process background: {self.process_background}\n\n"
            f"{optional_context_text}"
            "Input variables:\n"
            f"{variable_blocks}\n\n"
            "Return strict JSON with this shape:\n"
            "{\n"
            '  "cards": [\n'
            "    {\n"
            '      "variable_name": "must exactly match the input variable name",\n'
            '      "canonical_name": "stable English name for the variable",\n'
            '      "unit": "standard unit symbol or short English unit name",\n'
            '      "physical_meaning": "concise English description of what the variable measures or controls",\n'
            '      "process_role": "concise English description of the process role",\n'
            '      "information_role": "control | state | observation | derived | time | history_target | other",\n'
            '      "temporal_relation_to_target": "concise English description of the temporal relation",\n'
            '      "relations_to_other_variables": ["important English task-level relations"]\n'
            "    }\n"
            "  ]\n"
            "}\n\n"
            "Rules:\n"
            "1. Preserve the input variable order exactly.\n"
            "2. Use only pre-training task information given above.\n"
            "3. Write all descriptive fields in concise English, even when the dataset "
            "documentation is Chinese. Keep variable_name unchanged, but translate the other "
            "fields into clear English technical phrases.\n"
            "4. Do not write embedding_text yourself. The code will render a compact, fixed-form "
            "English embedding_text from the structured fields.\n"
            "5. Keep each field short and technical. Do not add protocol warnings, audit notes, "
            "or generic safety boilerplate to variable semantics.\n"
            "6. Use scenario or split context only to clarify process background; "
            "do not create scenario-specific or sample-specific variable semantics.\n"
            "7. Do not refer to column order, such as first column, previous column, "
            "第几列, 上一列, or 下一列.\n"
            "8. Do not add variables, remove variables, rename variable_name, or include markdown."
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        return tuple(variable.name for variable in self.variables)

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "ForecastTaskSpec":
        variables_payload = payload.get("variables", payload.get("ordered_input_variables"))
        if not isinstance(variables_payload, list):
            raise ValueError("task spec must contain a variables or ordered_input_variables list")
        variables: list[VariableSpec] = []
        for item in variables_payload:
            if not isinstance(item, dict):
                raise ValueError("each variable spec must be a mapping")
            variables.append(VariableSpec.from_mapping(item))
        if not variables:
            raise ValueError("task spec must contain at least one variable")
        return cls(
            dataset=str(payload["dataset"]),
            target=str(payload["target"]),
            horizon=str(payload["horizon"]),
            online_boundary=str(payload["online_boundary"]),
            process_background=str(payload["process_background"]),
            variables=tuple(variables),
            task_type=str(payload.get("task_type", "")),
            input_window_definition=str(payload.get("input_window_definition", "")),
            split_description=str(payload.get("split_description", "")),
            scenario_descriptions=tuple(
                str(item) for item in payload.get("scenario_descriptions", ())
            ),
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "target": self.target,
            "horizon": self.horizon,
            "online_boundary": self.online_boundary,
            "process_background": self.process_background,
            "variables": [variable.as_dict() for variable in self.variables],
            "task_type": self.task_type,
            "input_window_definition": self.input_window_definition,
            "split_description": self.split_description,
            "scenario_descriptions": list(self.scenario_descriptions),
        }


def load_task_spec(task_spec_path: Path) -> ForecastTaskSpec:
    payload = json.loads(task_spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("task spec JSON must contain an object")
    return ForecastTaskSpec.from_mapping(payload)


@dataclass(frozen=True)
class SemanticCard:
    variable_name: str
    description: str
    source: str = "llm_generated"
    confidence: float | None = None

    def normalized_description(self) -> str:
        return " ".join(self.description.split())


def build_template_task_spec() -> ForecastTaskSpec:
    return ForecastTaskSpec(
        dataset="template_dataset",
        target="target_variable",
        horizon="one-step ahead or current-time estimation, to be fixed by dataset protocol",
        online_boundary="only variables available at prediction time may be used",
        process_background=(
            "Replace this with the dataset's process background, equipment context, "
            "sampling frequency, and deployment boundary."
        ),
        task_type="time-series forecasting or soft sensing",
        input_window_definition="replace with the dataset's legal history window",
        split_description="replace with the independent process or scenario split policy",
        scenario_descriptions=(
            "replace with concise train/eval scenario descriptions if available",
        ),
        variables=(
            VariableSpec(
                name="feed_flow",
                unit="m3/h",
                sampling_frequency="1 min",
                availability="online at prediction time",
                role="manipulated or measured process input",
                description="Material entering the process.",
            ),
            VariableSpec(
                name="reactor_temperature",
                unit="degC",
                sampling_frequency="1 min",
                availability="online at prediction time",
                role="process state",
                description="Thermal state observed before or at the prediction time.",
            ),
            VariableSpec(
                name="target_history",
                unit="same as target",
                sampling_frequency="1 min",
                availability="past values only",
                role="historical target observation",
                description="Lagged target values within the defined history window.",
            ),
        ),
    )
