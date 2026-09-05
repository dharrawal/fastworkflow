from __future__ import annotations

import json
from pathlib import Path

import pytest

from fastworkflow.review_adapters import (
    exp028_answer_rating_to_sidecar_export,
    ido_rating_to_sidecar_export,
    sidecar_export_to_exp028_answer_rating,
    sidecar_export_to_ido_rating,
)

FIXTURES = Path(__file__).parent / "fixtures" / "review_adapters"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("sidecar_name", "contract_name", "to_contract", "to_sidecar"),
    [
        (
            "ido-sidecar-export.json",
            "ido-rating-v1.json",
            sidecar_export_to_ido_rating,
            ido_rating_to_sidecar_export,
        ),
        (
            "exp028-sidecar-export.json",
            "exp028-answer-rating-v1.json",
            sidecar_export_to_exp028_answer_rating,
            exp028_answer_rating_to_sidecar_export,
        ),
    ],
)
def test_adapter_golden_documents_round_trip_every_semantic_field(
    sidecar_name, contract_name, to_contract, to_sidecar
):
    sidecar_export = _load(sidecar_name)
    contract = _load(contract_name)

    assert to_contract(sidecar_export) == contract
    assert to_sidecar(contract) == sidecar_export
