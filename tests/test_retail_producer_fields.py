"""The retail example's producers carry typed id fields, and the 17 ``from_command_output`` declarations name them
(bd fix-qjne.4, option B).

``get_user_details``, ``get_order_details``, ``get_product_details`` and ``list_all_product_types`` keep their one
text field and return ``CommandResponse(response=<the text JSON as before>, output=<Output with id lists>)``, so what
a user or an agent reads (the retail benchmark reads it) is byte-identical to the bare-Output response of before.
"""
from __future__ import annotations

import inspect
import os
import shutil

import pytest
from dotenv import dotenv_values
from pydantic import create_model

import fastworkflow
from fastworkflow import ModuleType
from fastworkflow.planning import command_preview
from fastworkflow.planning import planning_commands as pc
from fastworkflow.engine.command_executor import frame_command_result, paging_line
from fastworkflow.observation_offloading.facts import output_page
from fastworkflow.planning.planning_phases import _fitting_uid_column
from tests.test_field_sources import RETAIL, ROOT

U, O, P, L = "get_user_details", "get_order_details", "get_product_details", "list_all_product_types"

# command -> field -> the declaration (a str, or a list of them) the example writes
DECLARATIONS = {
    "get_order_details": {"order_id": f"{U}.order_ids"},
    "get_product_details": {"product_id": [f"{O}.product_ids", f"{L}.product_ids"]},
    "modify_pending_order_address": {"order_id": f"{U}.order_ids"},
    "modify_pending_order_payment": {"order_id": f"{U}.order_ids", "payment_method_id": f"{U}.payment_method_ids"},
    "cancel_pending_order": {"order_id": f"{U}.order_ids"},
    "return_delivered_order_items": {
        "order_id": f"{U}.order_ids", "item_ids": f"{O}.item_ids", "payment_method_id": f"{O}.payment_method_ids"},
    "exchange_delivered_order_items": {
        "order_id": f"{U}.order_ids", "item_ids": f"{O}.item_ids", "new_item_ids": f"{P}.item_ids",
        "payment_method_id": [f"{O}.payment_method_ids", f"{U}.payment_method_ids"]},
    "modify_pending_order_items": {
        "order_id": f"{U}.order_ids", "item_ids": f"{O}.item_ids", "new_item_ids": f"{P}.item_ids",
        "payment_method_id": [f"{O}.payment_method_ids", f"{U}.payment_method_ids"]},
}


@pytest.fixture(scope="module")
def data():
    from fastworkflow.examples.retail_workflow.retail_data import load_data
    return load_data()


@pytest.fixture(scope="module")
def routing():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    fastworkflow.RoutingRegistry.clear_registry()
    return fastworkflow.RoutingRegistry.get_definition(RETAIL)


def _run(routing, command, **arguments):
    """The framework-framed output of *command* run on the example's data, and its Input."""
    generator = routing.get_command_class(command, ModuleType.RESPONSE_GENERATION_INFERENCE)()
    input_cls = routing.get_command_class(command, ModuleType.COMMAND_PARAMETERS_CLASS)
    parameters = input_cls(**arguments) if input_cls is not None else None
    result = generator(None, command, parameters)
    output_model = inspect.getmodule(type(generator)).Signature.Output
    framed = frame_command_result(result, command_name=command, output_model=output_model,
                                  command_parameters=parameters)
    return framed.command_response, parameters


# ------------------------------------------------------------------ the declarations load

def test_the_example_loads_with_the_17_declarations_and_every_edge_resolves(tmp_path):
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    folder = str(tmp_path / "retail")
    shutil.copytree(RETAIL, folder, ignore=shutil.ignore_patterns("___*", "__pycache__"))
    assert command_preview.load_problems(folder) == []
    resolution = pc.resolve_field_sources(folder)
    assert resolution.errors == ()
    for command, fields in DECLARATIONS.items():
        input_fields = resolution.signatures[command].Input.model_fields
        for field, declared in fields.items():
            extra = input_fields[field].json_schema_extra
            assert extra["from_command_output"] == declared, (command, field)
            wanted = [declared] if isinstance(declared, str) else declared
            assert [str(entry) for entry in extra["available_from"]] == wanted, (command, field, extra)


def test_there_are_17_declarations():
    # one per old bare declaration: the two-producer fields count once each
    assert sum(len(fields) for fields in DECLARATIONS.values()) == 17


# ------------------------------------------------------------------ the response text is byte-identical

def _old_text(field: str, text: str) -> str:
    """What the bare Output of before answered: the Output as JSON, with only its text field."""
    return create_model("Before", **{field: (str, ...)})(**{field: text}).model_dump_json()


def test_get_user_details_response_is_unchanged_for_every_user_and_an_unknown_one(routing, data):
    from fastworkflow.examples.retail_workflow.tools.get_user_details import GetUserDetails
    for user_id in [*data["users"], "nobody_here_0"]:
        response, parameters = _run(routing, U, user_id=user_id)
        text = GetUserDetails.invoke(data=data, user_id=user_id)
        assert response.response == _old_text("user_details", text), user_id


def test_get_order_details_response_is_unchanged_for_every_order_and_an_unknown_one(routing, data):
    from fastworkflow.examples.retail_workflow.tools.get_order_details import GetOrderDetails
    for order_id in [*data["orders"], "#W0000000X"]:
        response, _ = _run(routing, O, order_id=order_id)
        text = GetOrderDetails.invoke(data=data, order_id=order_id)
        assert response.response == _old_text("order_details", text), order_id


def test_get_product_details_response_is_unchanged_for_every_product_and_an_unknown_one(routing, data):
    from fastworkflow.examples.retail_workflow.tools.get_product_details import GetProductDetails
    for product_id in [*data["products"], "0000000000"]:
        response, _ = _run(routing, P, product_id=product_id)
        text = GetProductDetails.invoke(data=data, product_id=product_id)
        assert response.response == _old_text("product_details", text), product_id


def test_list_all_product_types_response_is_unchanged(routing, data):
    from fastworkflow.examples.retail_workflow.tools.list_all_product_types import ListAllProductTypes
    response, _ = _run(routing, L)
    assert response.response == f"list of product types: {ListAllProductTypes.invoke(data=data)}"


# ------------------------------------------------------------------ the id lists come from the structured data

def test_get_user_details_lists_the_orders_and_payment_methods(routing, data):
    for user_id, user in data["users"].items():
        response, _ = _run(routing, U, user_id=user_id)
        assert response.output.order_ids == user["orders"]
        assert response.output.payment_method_ids == list(user["payment_methods"])
    assert any(user["orders"] for user in data["users"].values())


def test_get_order_details_lists_items_products_and_payment_methods(routing, data):
    for order_id, order in data["orders"].items():
        response, _ = _run(routing, O, order_id=order_id)
        assert response.output.item_ids == [item["item_id"] for item in order["items"]]
        assert set(response.output.product_ids) == {item["product_id"] for item in order["items"]}
        assert len(response.output.product_ids) == len(set(response.output.product_ids))
        assert response.output.payment_method_ids == list(dict.fromkeys(
            p["payment_method_id"] for p in order["payment_history"]))


def test_get_product_details_lists_the_items(routing, data):
    for product_id, product in data["products"].items():
        response, _ = _run(routing, P, product_id=product_id)
        assert response.output.item_ids == [v["item_id"] for v in product["variants"].values()]


def test_list_all_product_types_lists_every_product_id(routing, data):
    response, _ = _run(routing, L)
    assert sorted(response.output.product_ids) == sorted(data["products"])


@pytest.mark.parametrize("command, arguments, empty", [
    (U, {"user_id": "nobody_here_0"}, ("order_ids", "payment_method_ids")),
    (O, {"order_id": "#W0000000X"}, ("item_ids", "product_ids", "payment_method_ids")),
    (P, {"product_id": "0000000000"}, ("item_ids",)),
])
def test_an_unknown_id_answers_the_error_text_and_empty_lists(routing, command, arguments, empty):
    response, _ = _run(routing, command, **arguments)
    assert "Error:" in response.response
    for name in empty:
        assert getattr(response.output, name) == []


# ------------------------------------------------------------------ the new list fields change nothing else

def test_the_producers_are_not_pages_and_get_no_paging_line(routing, data):
    user_id = next(iter(data["users"]))
    order_id = next(iter(data["orders"]))
    product_id = next(iter(data["products"]))
    for command, arguments in [(U, {"user_id": user_id}), (O, {"order_id": order_id}),
                               (P, {"product_id": product_id}), (L, {})]:
        response, parameters = _run(routing, command, **arguments)
        assert output_page(response.output) is None, command
        assert paging_line(response.output, parameters) == "", command
        assert "Rows " not in response.response, command


def test_a_uid_column_fits_a_consumer_only_where_one_list_names_it(routing, data):
    consumer = routing.get_command_class(P, ModuleType.COMMAND_PARAMETERS_CLASS)
    user_out = _run(routing, U, user_id=next(u for u, v in data["users"].items() if v["orders"]))[0].output
    order_out = _run(routing, O, order_id=next(iter(data["orders"])))[0].output
    # two id columns in one Output: never a row pick by name
    assert _fitting_uid_column("order_id", [consumer], U, user_out.model_dump()) is None
    assert _fitting_uid_column("item_id", [consumer], O, order_out.model_dump()) is None
    # one id column the consumer's declaration names: the generic row pick may use it (the retail example has no
    # planning command, so nothing picks rows today)
    listing = _run(routing, L)[0].output
    assert _fitting_uid_column("product_id", [consumer], L, listing.model_dump()) == "product_ids"
    product_out = _run(routing, P, product_id=next(iter(data["products"])))[0].output
    modify = routing.get_command_class("modify_pending_order_items", ModuleType.COMMAND_PARAMETERS_CLASS)
    assert _fitting_uid_column("new_item_ids", [modify], P, product_out.model_dump()) == "item_ids"
