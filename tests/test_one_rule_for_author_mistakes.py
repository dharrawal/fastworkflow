"""One rule for an author's mistake (bd fix-ftdl.2).

COMMAND-LOCAL: a mistake inside one command file fails THAT command only. It does not load (a planning command is not
in the planning map; an execution command whose file cannot be read is not registered, and one whose classes are read
raises its load error on every class fetch), ``command_preview.load_problems`` lists it as ``<command>: <file>:
<message>`` with what to write instead, and the rest of the workflow loads.

WORKFLOW-GLOBAL: a mistake in something the workflow shares stops the workflow, with one message that names the file.

Each case is built on a TEMP COPY of the library example's commands (``tests.test_field_sources._build_workflow``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from dotenv import dotenv_values

import fastworkflow
from fastworkflow import ModuleType, command_preview, workflow_agent
from fastworkflow import planning_commands as pc
from fastworkflow.command_directory import CommandLoadError
from fastworkflow.command_routing import RoutingRegistry
from tests.test_command_preview import PING, PONG
from tests.context_declarations import declare_inherits
from tests.test_field_sources import ROOT, _build_workflow


@pytest.fixture(autouse=True)
def _init():
    fastworkflow.init(env_vars=dict(dotenv_values(os.path.join(ROOT, "fastworkflow", "examples", "fastworkflow.env"))))
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()
    workflow_agent.split_settings.cache_clear()
    yield
    RoutingRegistry.clear_registry()
    pc.resolve_field_sources.cache_clear()


_EXEC = '''import fastworkflow
from fastworkflow import descends_to, user_approval_for_execution, UserApproval
from pydantic import BaseModel, Field


class Signature:
    """Say hello."""

    class Input(BaseModel):
        name: str = Field(description="who to greet"{input_extra})

    class Output(BaseModel):
        text: str = Field(description="the greeting"{output_extra})

    plain_utterances = ["say hello to Ann"]


{decorators}
class ResponseGenerator:
{attributes}
    def __call__(self, workflow: fastworkflow.Workflow, {parameters}) -> Signature.Output:
        return Signature.Output(text="hello")
'''


def _exec(output_extra="", decorators="", attributes="    pass", parameters="command: str, input: Signature.Input",
          input_extra=""):
    return _EXEC.format(output_extra=output_extra, decorators=decorators, attributes=attributes, parameters=parameters,
                        input_extra=input_extra)


_PLAN = '''from pydantic import BaseModel, Field
from fastworkflow import writes
from fastworkflow.planning import run
{imports}


class Signature:
    """Inspect a book."""

    class Input(BaseModel):
        query: str = Field(description="words"{input_extra})

    plain_utterances = ["inspect a book"]

{before}
{decorators}
class PlanGenerator{bases}:
{body}
'''
_STEPS = '''    def steps(self, input):
        return [run("find_book", query=input.query), "say what was found"]'''


def _plan(imports="", input_extra="", before="", decorators="", bases="", body=_STEPS):
    return _PLAN.format(imports=imports, input_extra=input_extra, before=before, decorators=decorators, bases=bases,
                        body=body)


_FACTS_OF_STR = _plan(imports="from .find_book import Signature as FindBook", before='''
class Facts(BaseModel):
    count: str
''', body='''    def find(self, input) -> Facts:
        return [run("find_book", query=input.query)]

    def __result__(self, find):
        return "found"''')

_INDIRECT_FOREIGN_BASE = _plan(imports="from .base_plan import PlanGenerator as Other",
                               before="class _Shared(Other):\n    pass\n", bases="(_Shared)", body="    pass")

#: a planning command with fixed steps (no Input, one plain method, no __result__) whose run() step names no command
_FIXED_STEPS_TYPO = '''from fastworkflow.planning import run


class Signature:
    """List every book."""

    plain_utterances = ["list every book"]


class PlanGenerator:
    def steps(self):
        return [run("find_bok", query="all"), "say what was found"]
'''

def _phased(phases: str, before: str = "") -> str:
    """A Library planning command with phases over find_book (the load-check sweep's planning declarations)."""
    return _plan(imports="from fastworkflow.planning import phase, do_not_run\nfrom .find_book import Signature as FindBook",
                 before=before, body=phases)


_FIND = '''    def find(self, input) -> FindBook.Output:
        return [run("find_book", query=input.query)]
'''
_RESULT = '''
    def __result__(self, find):
        return "found"'''

#: an execution command whose row id (book_uid) is not its only required input
_LEND = _exec().replace('        name: str = Field(description="who to greet")',
                        '        book_uid: str = Field(description="the book")\n        days: int = Field(description="for how long")')

_BOTH_GENERATORS = _exec() + '''

class PlanGenerator:
    def steps(self):
        return ["say hi"]
'''

#: case -> (execution commands, planning commands, the command that must not load, text its one problem holds)
LOCAL = {
    "call cycle": ({}, {"ping": PING, "pong": PONG}, "ping", "a planning command cannot call itself, even through others: ping -> pong -> ping"),
    # a step name resolves nearest scope first, as the runtime does: a command that shadows the global command of its
    # name and runs that name calls itself (bd fix-ftdl.9.4 item 1)
    "self-call through a shadowed name": ({}, {"inspect_book": _plan(), "Library/inspect_book": _plan(body='''    def steps(self, input):
        return [run("inspect_book", query=input.query), "say what was found"]''')}, "Library/inspect_book", "a planning command cannot call itself: Library/inspect_book -> Library/inspect_book"),
    "facts field of another type": ({}, {"Library/inspect_book": _FACTS_OF_STR}, "Library/inspect_book", "Facts.count is str; a facts field is typed with a command's Output"),
    "ui hint on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"ui": "bogus"}')}, {}, "Library/greet", "field 'text': ui must be a dict"),
    "removed key on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"enum": ["a"]}')}, {}, "Library/greet", 'enum: ["a"] is removed: write the field\'s type as Literal["a"]'),
    "from_command_output on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"from_command_output": "find_book.titles"}')}, {}, "Library/greet", "Signature.Output.text: an Output field cannot declare from_command_output"),
    "available_from on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"available_from": ["find_book"]}')}, {}, "Library/greet", "Signature.Output.text: available_from is written by the framework"),
    "misspelled descends_to target": ({"Library/greet": _exec(decorators='@descends_to("Bok")')}, {}, "Library/greet", '@descends_to("Bok") names no context of the workflow; did you mean "Book"?'),
    "two-argument __call__": ({"Library/greet": _exec(parameters="input: Signature.Input")}, {}, "Library/greet", "ResponseGenerator.__call__ must take three arguments, (self, workflow, command, input), where input is None for a command with no Input; it takes 2 here"),
    "defaulted third argument": ({"Library/greet": _exec(parameters="command: str, input: Signature.Input = None")}, {}, "Library/greet", "input has a default here, which it must not have"),
    "approval_template naming no field": ({"Library/greet": _exec(decorators="@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)", attributes='    approval_template = "Greet {nobody}"')}, {}, "Library/greet", "approval_template 'Greet {nobody}' names {nobody}, which is not a field of Input; write a field name in braces (its fields: name)"),
    "approval_template with no approval": ({"Library/greet": _exec(attributes='    approval_template = "Greet {name}"')}, {}, "Library/greet", "approval_template is read only for a command that asks for approval"),
    "foreign PlanGenerator base reached indirectly": ({}, {"Library/base_plan": _plan(), "Library/inspect_book": _INDIRECT_FOREIGN_BASE}, "Library/inspect_book", "PlanGenerator inherits from Library/base_plan's PlanGenerator"),
    "both generator classes": ({"Library/greet": _BOTH_GENERATORS}, {}, "Library/greet", "defines both ResponseGenerator and PlanGenerator; a command file holds one of them"),
    "db_lookup on a planning field": ({}, {"Library/inspect_book": _plan(input_extra=', json_schema_extra={"db_lookup": True}')}, "Library/inspect_book", "db_lookup is retired"),
    "a file that does not parse": ({"Library/greet": _exec() + "\ndef (:\n"}, {}, "Library/greet", "SyntaxError: invalid syntax"),
    "a file that does not import": ({"Library/greet": "import no_such_module_b2\n" + _exec()}, {}, "Library/greet", "ModuleNotFoundError: No module named 'no_such_module_b2'"),
    "retired writes attribute": ({"Library/greet": _exec(attributes="    writes = True")}, {}, "Library/greet", "writes = True is removed: write @writes"),
    # One rule for every field's json_schema_extra keys (bd fix-ftdl.8.22): a key that is not an option of its position
    # fails that command, on an execution command's Input and on every Output field too.
    "unknown key on an execution Input field": ({"Library/greet": _exec(input_extra=', json_schema_extra={"x_note": 1}')}, {}, "Library/greet", "field 'name': unknown option 'x_note'; its options are copied_from_command_text"),
    "requires on an execution Input field": ({"Library/greet": _exec(input_extra=', json_schema_extra={"requires": ["x"]}')}, {}, "Library/greet", "field 'name': unknown option 'requires'"),
    "db_lookup on an execution field": ({"Library/greet": _exec(input_extra=', json_schema_extra={"db_lookup": True}')}, {}, "Library/greet", "db_lookup is retired"),
    "unknown key on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"colour": "red"}')}, {}, "Library/greet", "field 'text': unknown option 'colour'; an Output field takes ui"),
    "retired key on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"in_request": True}')}, {}, "Library/greet", "field 'text': in_request: True is removed: write"),
    "Input option on an Output field": ({"Library/greet": _exec(output_extra=', json_schema_extra={"copied_from_command_text": True}')}, {}, "Library/greet", "field 'text': copied_from_command_text is an option of an Input field"),
    # The load-check sweep: a generator is a class defined in the command file; a file with no Signature and no
    # generator is not a command; __call__ is a plain def.
    "an assigned ResponseGenerator": ({"Library/greet": _exec().replace("class ResponseGenerator:", "class _Greet:") + "\n\nResponseGenerator = _Greet\n"}, {}, "Library/greet", "ResponseGenerator is imported or assigned here, not defined; a generator is a class defined in the command file"),
    "an imported PlanGenerator": ({}, {"Library/base_plan": _plan(), "Library/inspect_book": _plan(imports="from .base_plan import PlanGenerator", body="    pass").split("\n\n\nclass PlanGenerator")[0] + "\n"}, "Library/inspect_book", "PlanGenerator is imported or assigned here, not defined"),
    "a helper module named like a command": ({"Library/text_helpers": "def shout(text):\n    return text.upper()\n"}, {}, "Library/text_helpers", "is not a command: it defines no Signature, ResponseGenerator or PlanGenerator. A file in _commands/ is a command; a helper module belongs in the package _commands/_shared/ (_commands/_shared/text_helpers.py)"),
    "async __call__": ({"Library/greet": _exec().replace("    def __call__", "    async def __call__")}, {}, "Library/greet", "ResponseGenerator.__call__ is async def here; it must be a plain def"),
    # display hints against the field's type (the load-check sweep): a badge on a number is allowed
    "hero on a list": ({"Library/greet": _exec(output_extra=', json_schema_extra={"ui": {"hero": True}}').replace("text: str = Field(description=\"the greeting\"", "text: list[str] = Field(description=\"the greeting\"")}, {}, "Library/greet", "ui.hero marks the one scalar to show first; list[str] is a list"),
    "percent format on a str": ({"Library/greet": _exec(output_extra=', json_schema_extra={"ui": {"format": "percent"}}')}, {}, "Library/greet", "ui.format \"percent\" is for an int or float value (or a list of one); str is not one"),
    "currency format on a str": ({"Library/greet": _exec(output_extra=', json_schema_extra={"ui": {"format": "currency"}}')}, {}, "Library/greet", "ui.format \"currency\" is for an int or float value (or a list of one); str is not one"),
    # an Input's model validators run after its fields (the one engine of validate_input)
    "before-mode model validator on an execution Input": ({"Library/greet": _exec().replace("from pydantic import BaseModel, Field", "from pydantic import BaseModel, Field, model_validator").replace("        name: str = Field(description=\"who to greet\")\n", "        name: str = Field(description=\"who to greet\")\n\n        @model_validator(mode=\"before\")\n        @classmethod\n        def tidy(cls, data):\n            return data\n")}, {}, "Library/greet", "Signature.Input: model validator 'tidy' is mode=\"before\"; an Input's model validators must be mode=\"after\""),
    "wrap-mode model validator on a planning Input": ({}, {"Library/inspect_book": _plan(imports="from pydantic import model_validator").replace("        query: str = Field(description=\"words\")\n", "        query: str = Field(description=\"words\")\n\n        @model_validator(mode=\"wrap\")\n        @classmethod\n        def tidy(cls, data, handler):\n            return handler(data)\n")}, "Library/inspect_book", "Signature.Input: model validator 'tidy' is mode=\"wrap\""),
    "fixed steps naming no command": ({}, {"Library/every_book": _FIXED_STEPS_TYPO}, "Library/every_book", "a step runs 'find_bok', which is not a command of the workflow"),
    # planning declarations (the load-check sweep)
    "descends_to a context that is not occupiable": ({"Library/greet": _exec(decorators='@descends_to("Record")')}, {}, "Library/greet", '@descends_to("Record") names a context declared occupiable = False'),
    "max_items on a phase with no over": ({}, {"Library/inspect_book": _phased("    @phase(max_items=3)\n" + _FIND + _RESULT)}, "Library/inspect_book", "phase 'find' of PlanGenerator: @phase(max_items=) is for a fan-out phase, which runs once per item: add over=..., or leave max_items out"),
    "until on a phase with no over": ({}, {"Library/inspect_book": _phased("    @phase(until=lambda find: True)\n" + _FIND + _RESULT)}, "Library/inspect_book", "@phase(until=) is for a fan-out phase"),
    "bind that is not a bool": ({}, {"Library/inspect_book": _phased(_FIND + '''
    @phase(over="find.book_uids", bind="yes")
    def book(self, uid):
        return [run("find_book", query=uid)]
''' + _RESULT)}, "Library/inspect_book", "@phase(bind=...) must be True or False, got 'yes'"),
    "over a single value": ({}, {"Library/inspect_book": _phased(_FIND + '''
    @phase(over="find.total")
    def book(self, total):
        return [run("find_book", query=total)]
''' + _RESULT)}, "Library/inspect_book", "over='find.total' reads one value (int), not a list, so the fan-out would run once"),
    "a phase returning Output or None": ({}, {"Library/inspect_book": _phased(_FIND.replace("-> FindBook.Output:", "-> FindBook.Output | None:") + _RESULT)}, "Library/inspect_book", "its return annotation Output | None includes None"),
    "run named= on a command that requires more": ({"Library/lend": _LEND}, {"Library/inspect_book": _plan(body='''    def steps(self, input):
        return [run("lend", named=input.query), "say what was lent"]''')}, "Library/inspect_book", "a step opens a row of 'lend' by named=, but 'lend' also requires days, which such a step cannot pass"),
    "do_not_run naming no command": ({}, {"Library/inspect_book": _phased(_FIND.replace('query=input.query)]', 'query=input.query), do_not_run("find_bok")]') + _RESULT)}, "Library/inspect_book", "a step names 'find_bok', which is not a command of the workflow"),
    "an option on a facts field": ({}, {"Library/inspect_book": _phased('''    def find(self, input) -> Facts:
        return [run("find_book", query=input.query)]
''' + _RESULT, before='''
class Facts(BaseModel):
    found: FindBook.Output = Field(json_schema_extra={"copied_from_command_text": True})
''')}, "Library/inspect_book", "Facts: field 'found': copied_from_command_text is an option of a command's Input or Output field; a facts field takes none"),
    # the retired copy key on a facts field typed with a command's Output (bd fix-ftdl.9.4): it was ignored silently
    "a retired from key on a typed facts field": ({}, {"Library/inspect_book": _phased('''    def find(self, input) -> Facts:
        return [run("find_book", query=input.query)]
''' + _RESULT, before='''
class Facts(BaseModel):
    found: FindBook.Output = Field(json_schema_extra={"from": "find_book"})
''')}, "Library/inspect_book", 'Facts: field \'found\': from is removed: copies are no longer read (fact_from and "from" are removed); the field\'s type, a command\'s Output, says what it holds: leave the key out'),
    "a json_schema_extra that is a function": ({"Library/greet": _exec(input_extra=", json_schema_extra=lambda schema: None")}, {}, "Library/greet", "field 'name': json_schema_extra must be a dict of the field's options"),
    "@writes on a PlanGenerator": ({}, {"Library/inspect_book": _plan(decorators="@writes")}, "Library/inspect_book", "a planning command cannot carry @writes"),
    # a declared default that is the framework's not-given marker: it could never be told from a missing value
    "NOT_FOUND default on an execution Input": ({"Library/greet": _exec(input_extra=', default="NOT_FOUND"')}, {}, "Library/greet", "Signature.Input.name: default 'NOT_FOUND' is the framework's not-given marker, so a value could never be told from a missing one. Declare a required field with no default (`name: str = Field(description=...)`, asked for when missing), or `name: str | None = None` for an optional one"),
    "the NOT_FOUND constant as a planning Input default": ({}, {"Library/inspect_book": _plan(imports="from fastworkflow.parameters.signatures import NOT_FOUND", input_extra=", default=NOT_FOUND")}, "Library/inspect_book", "Signature.Input.query: default 'NOT_FOUND' is the framework's not-given marker"),
    "the int not-given marker as a default": ({"Library/greet": _exec(input_extra=", default=INVALID_INT_VALUE").replace("name: str = Field(", "name: int = Field(").replace("from pydantic import", "from fastworkflow.parameters.signatures import INVALID_INT_VALUE\nfrom pydantic import")}, {}, "Library/greet", "Signature.Input.name: default -9223372036854775807 is the framework's not-given marker"),
    # the settled list of Signature members (planning_commands.SIGNATURE_MEMBERS): anything else public fails the command
    "a misspelt Signature member": ({"Library/greet": _exec().replace("    plain_utterances = [", "    plain_utterance = [")}, {}, "Library/greet", "Signature.plain_utterance is not a Signature member; did you mean 'plain_utterances'? The members are Input, Output, plain_utterances, generate_utterances, template_utterances (and template on a planning command); a helper's name starts with an underscore"),
    "a stray nested class on Signature": ({"Library/greet": _exec().replace("    plain_utterances = [", "    class Inputs(BaseModel):\n        pass\n\n    plain_utterances = [")}, {}, "Library/greet", "Signature.Inputs is not a Signature member; did you mean 'Input'?"),
    "template on an execution Signature": ({"Library/greet": _exec().replace("    plain_utterances = [", "    template = ['greet']\n    plain_utterances = [")}, {}, "Library/greet", "Signature.template is read only on a planning command"),
    "a removed hook on Signature": ({"Library/greet": _exec().replace("    plain_utterances = [", "    def process_extracted_parameters(self, workflow, command, cmd_parameters):\n        pass\n\n    plain_utterances = [")}, {}, "Library/greet", "Signature.process_extracted_parameters is not a Signature member"),
    "SlotInput on an application Signature": ({}, {"Library/inspect_book": _plan().replace("    plain_utterances = [", "    class SlotInput(BaseModel):\n        slot: str\n\n    plain_utterances = [")}, "Library/inspect_book", "Signature.SlotInput is not a Signature member"),
    "an unknown member on a planning Signature": ({}, {"Library/inspect_book": _plan().replace("    plain_utterances = [", "    utterances = ['inspect it']\n    plain_utterances = [")}, "Library/inspect_book", "Signature.utterances is not a Signature member; did you mean 'plain_utterances'?"),
    "a member inherited from the Signature's base": ({"Library/greet": _exec().replace("\n\nclass Signature:", "\n\nclass _Base:\n    greeting_word = 'hello'\n\n\nclass Signature(_Base):")}, {}, "Library/greet", "Signature.greeting_word is not a Signature member"),
    "@writes on Signature": ({"Library/greet": _exec().replace("from fastworkflow import descends_to", "from fastworkflow import writes, descends_to").replace("\n\nclass Signature:", "\n\n@writes\nclass Signature:")}, {}, "Library/greet", "@writes is on Signature: a declaration goes on the ResponseGenerator or PlanGenerator class"),
}


@pytest.mark.parametrize("case", sorted(LOCAL))
def test_a_mistake_in_one_command_file_fails_that_command_only_and_is_listed(tmp_path, case):
    commands, planning, failing, text = LOCAL[case]
    folder = _build_workflow(tmp_path, "wf", extra_commands=commands, extra_planning_commands=planning)
    problems = command_preview.load_problems(folder)
    mine = [p for p in problems if p.startswith(f"{failing}: ")]
    assert len(mine) == 1 and text in mine[0], problems
    assert f"{failing.rsplit('/', 1)[-1]}.py" in mine[0]                       # the message names the file
    assert all(p.split(": ", 1)[0] in {failing, "pong"} for p in problems), problems   # nothing else
    definition = RoutingRegistry.get_definition(folder)                       # the workflow loads
    if failing in planning:
        assert failing not in workflow_agent.load_planning_command_map(folder)
        assert "hours_overview" in workflow_agent.load_planning_command_map(folder)
    elif failing in definition.command_directory.map_command_2_metadata:      # classes read: every fetch raises
        for module_type in (ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS, ModuleType.RESPONSE_GENERATION_INFERENCE):
            with pytest.raises(CommandLoadError):
                definition.get_command_class(failing, module_type)
    else:                                                                     # the file cannot be read: unregistered
        assert failing in definition.command_directory.load_errors
        assert failing not in definition.get_command_names("Library")
    assert definition.get_command_class("Library/find_book", ModuleType.INPUT_FOR_PARAM_EXTRACTION_CLASS) is not None


def test_a_signature_with_helpers_and_inherited_members_loads(tmp_path):
    """A name that starts with an underscore is the author's, and the listed members may come from a base class."""
    greet = _exec().replace("\n\nclass Signature:", "\n\nclass _Base:\n    plain_utterances = ['greet Ann']\n\n\n"
                            "class Signature(_Base):").replace(
        "    plain_utterances = [\"say hello to Ann\"]", "    _helper_words = ('hi', 'hello')\n\n    @staticmethod\n"
        "    def _shout(text):\n        return text.upper()")
    folder = _build_workflow(tmp_path, "wf_sig_ok", extra_commands={"Library/greet": greet})
    assert command_preview.load_problems(folder) == []


def test_slot_input_is_a_signature_member_of_the_framework_workflows_only():
    from pydantic import BaseModel

    class Signature:
        """Edit a proposal."""

        class SlotInput(BaseModel):
            slot: str

        plain_utterances = ["edit it"]

    assert pc.signature_member_problems(Signature, is_planning=False, framework=True) == []
    assert "Signature.SlotInput is not a Signature member" in \
        pc.signature_member_problems(Signature, is_planning=False, framework=False)[0]
    assert pc.is_framework_command_folder(fastworkflow.get_internal_workflow_path("plan_metadata_extraction"))
    assert not pc.is_framework_command_folder(os.path.join(ROOT, "fastworkflow", "examples", "library"))


def test_a_command_with_a_not_found_default_is_in_no_command_set(tmp_path):
    """A NOT_FOUND default fails the command's load, so it is in no scope's command set: the MCP server, which lists
    the scope's commands as tools, never sees one (it no longer special-cases that default)."""
    folder = _build_workflow(tmp_path, "wf_not_found", extra_commands={
        "Library/greet": _exec(input_extra=', default="NOT_FOUND"')})
    definition = RoutingRegistry.get_definition(folder)
    assert "Library/greet" in definition.not_loading()
    assert "Library/greet" not in definition.get_command_names("Library")
    assert "Library/find_book" in definition.get_command_names("Library")


def test_a_badge_on_a_number_loads(tmp_path):
    """A score drawn as a badge is a presentation choice, not a mistake."""
    folder = _build_workflow(tmp_path, "wf_badge", extra_commands={"Library/greet": _exec(
        output_extra=', json_schema_extra={"ui": {"format": "badge"}}').replace(
        "text: str = Field(description=\"the greeting\"", "text: float = Field(description=\"the greeting\"").replace(
        'Signature.Output(text="hello")', "Signature.Output(text=0.5)")})
    assert command_preview.load_problems(folder) == []


def test_a_good_approval_template_and_a_real_descends_to_target_load(tmp_path):
    folder = _build_workflow(tmp_path, "wf_fine", extra_commands={"Library/greet": _exec(
        decorators='@user_approval_for_execution(UserApproval.REQUIRED_ALWAYS)\n@descends_to("Book")',
        attributes='    approval_template = "Greet {name}"')})
    assert command_preview.load_problems(folder) == []


# ------------------------------------------------------------------ the workflow-global class

def _edit(folder: str, relative: str, old: str, new: str) -> None:
    path = Path(folder) / relative
    text = path.read_text(encoding="utf-8")
    assert text.count(old) == 1
    path.write_text(text.replace(old, new), encoding="utf-8")


def _one_global_problem(folder: str, file_label: str) -> str:
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and file_label in problems[0], problems
    return problems[0]


def test_an_unknown_base_stops_the_workflow_naming_the_file_and_the_referring_context(tmp_path):
    folder = _build_workflow(tmp_path, "wf_base", extra_inheritance={"Member": {"base": ["Recrd"]}})
    problem = _one_global_problem(folder, "_commands/Member/_Member.py")
    assert "inherits names 'Recrd', which is no context; did you mean 'Record'?" in problem
    with pytest.raises(Exception, match="inherits names 'Recrd', which is no context"):
        fastworkflow.command_context_model.CommandContextModel.load(folder)     # at load, not at first use


@pytest.mark.parametrize("old,new,text", [
    ("from typing import Any", "import no_such_module_b2\nfrom typing import Any", "No module named 'no_such_module_b2'"),
    ("    state_version = 1", "    state_version = 1\n\n    @classmethod\n    def instance_label(cls, record):\n"
                             "        return 'x'", "instance_label is removed"),
    ("    state_version = 1", "    state_version = 1\n    enter_command = 'show_book'", "Context.enter_command is no longer read"),
], ids=["does_not_import", "instance_label", "enter_command"])
def test_a_context_class_file_mistake_stops_the_workflow_naming_the_file(tmp_path, old, new, text):
    folder = _build_workflow(tmp_path, "wf_ctx")
    _edit(folder, "_commands/Book/_Book.py", old, new)
    problem = _one_global_problem(folder, "_commands/Book/_Book.py")
    assert text in problem
    with pytest.raises(Exception):
        RoutingRegistry.get_definition(folder)


def test_a_startup_action_naming_no_command_is_a_workflow_problem(tmp_path):
    folder = _build_workflow(tmp_path, "wf_startup")
    (Path(folder) / "startup_action.json").write_text('{"command_name": "strtup"}', encoding="utf-8")
    problem = _one_global_problem(folder, "startup_action.json")
    assert "command_name 'strtup' is no command of this workflow" in problem


def test_a_startup_command_that_does_not_load_is_a_workflow_problem_too(tmp_path):
    folder = _build_workflow(tmp_path, "wf_startup_broken")
    _edit(folder, "_commands/startup.py", "import fastworkflow", "import fastworkflow\nimport no_such_module_b2")
    problems = command_preview.load_problems(folder)
    assert [p.split(": ", 1)[0] for p in problems] == ["startup", "startup_action.json"], problems
    assert "its command 'startup' does not load" in problems[1]


def test_a_workflow_settings_file_that_does_not_import_is_a_workflow_problem_naming_the_file(tmp_path):
    folder = _build_workflow(tmp_path, "wf_split")
    (Path(folder) / "_commands" / "_workflow.py").write_text("import no_such_module_b2\n", encoding="utf-8")
    problem = _one_global_problem(folder, "_commands/_workflow.py")
    assert "could not import the workflow settings" in problem


# ------------------------------------------------------------------ the load-check sweep, workflow-global

def test_a_context_class_in_a_misnamed_file_stops_the_workflow_naming_the_right_file(tmp_path):
    folder = _build_workflow(tmp_path, "wf_ctx_name")
    book = Path(folder) / "_commands" / "Book"
    (book / "_Book.py").rename(book / "_book.py")
    problem = _one_global_problem(folder, "_commands/Book/_book.py")
    assert "defines class Context, but the context class file of Book is _Book.py: rename _book.py to _Book.py" \
        in problem


def test_the_removed_inheritance_file_stops_the_workflow(tmp_path):
    """context_inheritance_model.json is a removed spelling (bd fix-qjne.36): whatever it holds (here a key that is
    no folder, once its own mistake), its presence is one workflow-wide problem naming the file and the
    declarations that replace it."""
    folder = _build_workflow(tmp_path, "wf_inh_key")
    (Path(folder) / "_commands" / "context_inheritance_model.json").write_text(
        json.dumps({"Membr": {"base": ["Record"]}}), encoding="utf-8")
    problem = _one_global_problem(folder, "context_inheritance_model.json")
    assert problem.workflow_wide
    assert "context_inheritance_model.json is removed and not read" in problem
    assert "inherits = [" in problem and "fastworkflow migrate-contexts" in problem
    with pytest.raises(Exception, match="context_inheritance_model.json is removed and not read"):
        fastworkflow.command_context_model.CommandContextModel.load(folder)


_A_CONTEXT_CLASS = """class Context:
    \"\"\"A paperback book.\"\"\"
"""


def _context_folder(folder: str, name: str, files: dict[str, str]) -> None:
    path = Path(folder) / "_commands" / name
    path.mkdir(parents=True, exist_ok=True)
    for file_name, text in files.items():
        (path / file_name).write_text(text, encoding="utf-8")


def test_a_folder_with_no_command_file_that_inherits_one_loads_with_its_context_class(tmp_path, caplog):
    """A context whose commands are all inherited is a folder with no command file of its own; its _<Context>.py is
    read, and nothing is logged about it (it is not a mistake)."""
    folder = _build_workflow(tmp_path, "wf_inheriting")
    _context_folder(folder, "Paperback", {"_Paperback.py": _A_CONTEXT_CLASS})
    declare_inherits(folder, {"Paperback": ["Book"]})
    with caplog.at_level("DEBUG"):
        assert command_preview.load_problems(folder) == []
    assert not [r for r in caplog.records if r.levelname in ("WARNING", "ERROR") and "Paperback" in r.getMessage()]
    model = fastworkflow.command_context_model.CommandContextModel.load(folder)
    assert set(model.commands("Book")) <= set(model.commands("Paperback")) and model.commands("Paperback")
    definition = RoutingRegistry.get_definition(folder)
    assert "Paperback" in definition.command_directory.map_context_2_metadata
    assert set(definition.get_planning_command_names("Book")) <= set(definition.get_planning_command_names("Paperback"))


def test_a_folder_that_inherits_from_a_base_workflow_folder_loads(tmp_path):
    """A base workflow's context folders are context folders of the derived workflow: a key naming one is no
    mistake, and a commandless folder of the derived workflow may inherit from it."""
    base = _build_workflow(tmp_path, "wf_base_workflow")
    derived = tmp_path / "wf_derived"
    (derived / "_commands").mkdir(parents=True)
    (derived / "workflow_inheritance_model.json").write_text(json.dumps({"base": [base]}), encoding="utf-8")
    _context_folder(str(derived), "Paperback", {"_Paperback.py": _A_CONTEXT_CLASS})
    declare_inherits(derived, {"Paperback": ["Book"]})
    model = fastworkflow.command_context_model.CommandContextModel.load(str(derived))
    assert "Book/show_book" in model.commands("Paperback")


@pytest.mark.parametrize("extra_inheritance,folders,empty", [
    ({}, {"Orphan": {"_Orphan.py": _A_CONTEXT_CLASS}}, ["Orphan"]),
    ({}, {"shared": {"__init__.py": "def helper():\n    return 1\n"}}, ["shared"]),
    ({"Paperback": {"base": ["Hollow"]}, "Hollow": {"base": []}},
     {"Paperback": {"_Paperback.py": _A_CONTEXT_CLASS}, "Hollow": {"_Hollow.py": _A_CONTEXT_CLASS}},
     ["Hollow", "Paperback"]),
], ids=["no_key_no_commands", "helper_folder_without_underscore", "inherits_only_empty_contexts"])
def test_a_folder_with_no_command_file_that_inherits_none_stops_the_workflow(tmp_path, extra_inheritance, folders,
                                                                             empty):
    folder = _build_workflow(tmp_path, "wf_empty")
    for name, files in folders.items():
        _context_folder(folder, name, files)
    declare_inherits(folder, {name: entry["base"] for name, entry in extra_inheritance.items()})
    problems = command_preview.load_problems(folder)
    assert len(problems) == len(empty), problems
    for name, problem in zip(empty, problems):
        assert problem.workflow_wide
        assert f"_commands/{name}: the context folder {name!r} holds no command file and inherits none" in problem
        assert "add a command" in problem and "move helper modules to _commands/_shared/" in problem
    with pytest.raises(Exception, match="holds no command file and inherits none"):
        fastworkflow.command_context_model.CommandContextModel.load(folder)


def test_a_folder_whose_only_command_does_not_load_is_that_commands_problem_only(tmp_path):
    """Command files count, not commands that load: a folder whose one command fails has that command's line only."""
    folder = _build_workflow(tmp_path, "wf_broken_only")
    _context_folder(folder, "Paperback", {
        "_Paperback.py": _A_CONTEXT_CLASS,
        "price.py": _exec().replace("import fastworkflow", "import fastworkflow\nimport no_such_module_b2", 1)})
    problems = command_preview.load_problems(folder)
    assert len(problems) == 1 and not problems[0].workflow_wide, problems
    assert problems[0].startswith("Paperback/price: ") and "no_such_module_b2" in problems[0]


@pytest.mark.parametrize("old,new,name", [
    ("    @classmethod\n    def get_parent(", "    @classmethod\n    def get_parnet(", "get_parnet"),
    ("    state_version = 1", "    state_version = 1\n    instance_label_atr = 'title'", "instance_label_atr"),
], ids=["get_parnet", "instance_label_atr"])
def test_an_unknown_public_member_of_a_context_class_stops_the_workflow(tmp_path, old, new, name):
    folder = _build_workflow(tmp_path, "wf_ctx_member")
    _edit(folder, "_commands/Book/_Book.py", old, new)
    problem = _one_global_problem(folder, "_commands/Book/_Book.py")
    assert f"Context.{name} is not a context hook" in problem and "get_parent" in problem
    assert "a helper starts with _" in problem


def test_a_misspelt_key_in_startup_action_json_stops_the_workflow(tmp_path):
    folder = _build_workflow(tmp_path, "wf_startup_key")
    (Path(folder) / "startup_action.json").write_text('{"command_name": "startup", "paramters": {}}',
                                                       encoding="utf-8")
    problem = _one_global_problem(folder, "startup_action.json")
    assert "unknown key 'paramters'" in problem and "did you mean 'parameters'?" in problem


def test_a_step_that_runs_a_command_that_does_not_load_says_so(tmp_path):
    """The step's command exists: it does not load, and the planning command's line points at that command's own
    problem instead of calling it no command of the workflow."""
    broken = _exec(input_extra=', json_schema_extra={"x_note": 1}').replace('"""Say hello."""', '"""Find a book."""')
    folder = _build_workflow(tmp_path, "wf_step_broken", extra_commands={"Library/find_book": broken},
                             extra_planning_commands={"Library/inspect_book": _plan(body='''    def steps(self, input):
        return [run("find_book", name=input.query), "say what was found"]''')})
    problems = command_preview.load_problems(folder)
    mine = [p for p in problems if p.startswith("Library/inspect_book: ")]
    assert len(mine) == 1 and "a step runs 'find_book', which does not load: see its own problem" in mine[0], problems
    assert any(p.startswith("Library/find_book: ") and "x_note" in p for p in problems), problems


_BOOK_PHASES = '''from fastworkflow.planning import run


class BookPhases:
    def find(self, input):
        return [run("find_book", query=input.query), "say what was found"]

    def describe(self):
        return "a helper someone forgot to name with _"
'''


@pytest.mark.parametrize("template", [False, True], ids=["no_template", "declared_template"])
def test_a_public_helper_on_a_shared_base_names_the_base_and_the_underscore_fix(tmp_path, template):
    """The fix is the helper's name, not Signature.template; and declaring a template does not hide the mistake."""
    plan = _plan(imports="from .._shared.book_phases import BookPhases", bases="(BookPhases)", body="    pass")
    if template:
        plan = plan.replace('    plain_utterances = ["inspect a book"]',
                            '    template = ["find the book", "say what was found"]\n'
                            '    plain_utterances = ["inspect a book"]')
    folder = _build_workflow(tmp_path, "wf_helper", extra_planning_commands={"Library/inspect_book": plan})
    (Path(folder) / "_commands" / "_shared").mkdir()
    (Path(folder) / "_commands" / "_shared" / "book_phases.py").write_text(_BOOK_PHASES, encoding="utf-8")
    problems = command_preview.load_problems(folder)
    mine = [p for p in problems if p.startswith("Library/inspect_book: ")]
    assert len(mine) == 1, problems
    assert ("every public method of a PlanGenerator and of its bases is a phase, and describe is defined on the "
            "base BookPhases (book_phases.py): if describe is a helper, name it _describe") in mine[0]
    assert "declare Signature.template" not in mine[0]


def test_an_import_error_in_a_helper_names_the_helper_file_and_line(tmp_path):
    folder = _build_workflow(tmp_path, "wf_helper_import",
                             extra_commands={"Library/greet": "from .._shared.greet_helpers import X\n" + _exec()})
    (Path(folder) / "_commands" / "_shared").mkdir()
    (Path(folder) / "_commands" / "_shared" / "greet_helpers.py").write_text(
        "X = 1\nimport no_such_module_b3\n", encoding="utf-8")
    problems = command_preview.load_problems(folder)
    assert problems == ["Library/greet: _commands/Library/greet.py: ModuleNotFoundError: No module named "
                        "'no_such_module_b3' (in _commands/_shared/greet_helpers.py, line 2)"], problems
