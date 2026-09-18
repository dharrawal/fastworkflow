"""Auto-navigation (ido-8ps.9): the rule, the three cases, and the refusal.

R1 (ido-8ps.8) made a foreign known name loud; 74c348a made the hint say where
the command lives. This suite covers what happens next: the framework composes
[enter the owning context; run the original command] when the context model says
how, and BLOCKS with a clarification when it does not.

THE RULE, which every test here exists to hold: every automatic action is a pure
function of the UTTERANCE and the CONTEXT MODEL, never of history. The property
test in `TestNeverFromHistory` is the load-bearing one -- it parameterises over
several prior histories and asserts the decision cannot move.

Offline throughout: no model, no backend, no trained artifact, no live run. The
pure decision function is exercised directly; the two runtime seams (the CME
wildcard command and `CommandExecutor._auto_navigate`) are exercised with the
same doubles `test_intent_detection_foreign_context.py` uses.
"""

import json
import os
import shutil
from pathlib import Path

import pytest

import fastworkflow
from fastworkflow import auto_navigation
from fastworkflow.auto_navigation import (
    CLARIFY,
    DISPATCH,
    NONE,
    RULE_EXPLICIT_HANDLE,
    RULE_STATELESS,
    RULE_UTTERANCE_PARAMETERS,
    ContextEntry,
    EntryContract,
    candidate_values,
    clarification_text,
    decide,
    entry_command_name,
    validate_entry_contracts,
)
from fastworkflow._workflows.command_metadata_extraction import intent_detection
from fastworkflow._workflows.command_metadata_extraction._commands import (
    wildcard as wildcard_command,
)

INVENTORY_FIXTURE = Path(__file__).parent / "fixtures" / "ido_routing_inventory.json"
INTENT_DETECTION = fastworkflow.NLUPipelineStage.INTENT_DETECTION


# ---------------------------------------------------------------------------
# Contracts, written out rather than loaded, so the decision's inputs are visible
# ---------------------------------------------------------------------------

#: IDO's shape: an Account is entered by a uid nobody can guess.
ACCOUNT = EntryContract(
    context="Account",
    declaration="open_account_by_uid <account_uid>",
    command_name="open_account_by_uid",
    qualified_command_name="DirectoryExplorer/open_account_by_uid",
    owner_contexts=("DirectoryExplorer",),
    required_parameters=("account_uid",),
)
#: A workspace: one bare command enters it, so there is nothing to supply.
CONTROLS_MONITOR = EntryContract(
    context="ControlsMonitor",
    declaration="open_controls_monitor",
    command_name="open_controls_monitor",
    qualified_command_name="Directory/open_controls_monitor",
    owner_contexts=("Directory",),
)
#: Two required parameters, to show rule 2 is all-or-nothing.
REPORT = EntryContract(
    context="Report",
    declaration="open_report <report_uid> <revision>",
    command_name="open_report",
    qualified_command_name="Directory/open_report",
    owner_contexts=("Directory",),
    required_parameters=("report_uid", "revision"),
)


def entry(sequence=1, context="Account", command_name="open_account_by_uid",
          parameters=None, alias="O4"):
    return ContextEntry(
        sequence=sequence, context=context, command_name=command_name,
        parameters=parameters if parameters is not None else {"account_uid": "3f2a"},
        alias=alias,
    )


# ---------------------------------------------------------------------------
# The declaration
# ---------------------------------------------------------------------------

class TestTheDeclaration:
    @pytest.mark.parametrize("declaration,expected", [
        ("open_controls_monitor", "open_controls_monitor"),
        ("open_account_by_uid <account_uid>", "open_account_by_uid"),
        ("  open_account_by_uid  <account_uid>ACC</account_uid> ", "open_account_by_uid"),
        ("open_account_by_uid(account_uid)", "open_account_by_uid"),
        ("", None),
        ("<account_uid>", None),
        ("3f2a", None),
        (None, None),
    ])
    def test_only_the_leading_command_name_is_load_bearing(self, declaration, expected):
        """The tail is hint text. A declaration that carried real VALUES would
        make navigation a function of the declaration, not of the utterance."""
        assert entry_command_name(declaration) == expected

    def test_a_context_with_no_required_parameters_is_stateless(self):
        assert CONTROLS_MONITOR.stateless
        assert not ACCOUNT.stateless


# ---------------------------------------------------------------------------
# The three rules
# ---------------------------------------------------------------------------

class TestRuleOneStateless:
    def test_a_stateless_context_is_entered_with_its_bare_command(self):
        decision = decide(
            command_name="list_findings",
            utterance="list_findings",
            owner_contexts=["ControlsMonitor"],
            contracts={"ControlsMonitor": CONTROLS_MONITOR},
        )
        assert decision.kind == DISPATCH
        assert decision.rule == RULE_STATELESS
        assert decision.entered_context == "ControlsMonitor"
        assert decision.entry_utterance == "open_controls_monitor"
        assert decision.entry_parameters == {}

    def test_it_needs_no_registry_and_no_parameters_in_the_utterance(self):
        """Rule 1 is the case where there is nothing at all to be wrong about."""
        bare = decide(
            command_name="list_findings", utterance="list_findings",
            owner_contexts=["ControlsMonitor"],
            contracts={"ControlsMonitor": CONTROLS_MONITOR}, entries=(),
        )
        loaded = decide(
            command_name="list_findings", utterance="list_findings",
            owner_contexts=["ControlsMonitor"],
            contracts={"ControlsMonitor": CONTROLS_MONITOR},
            entries=(entry(), entry(sequence=2, alias="O9")),
        )
        assert bare == loaded


class TestRuleTwoParametersInTheUtterance:
    def test_the_value_the_agent_wrote_is_the_value_that_enters(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions <account_uid>3f2a</account_uid>",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        assert decision.kind == DISPATCH
        assert decision.rule == RULE_UTTERANCE_PARAMETERS
        assert decision.entry_utterance == (
            "open_account_by_uid <account_uid>3f2a</account_uid>")

    def test_it_reads_the_utterance_with_the_extractor_s_own_grammar(self):
        """The dispatched entry command is re-extracted by the real extractor,
        so a value this rule can see and that one cannot would dispatch a step
        that then fails to bind its own parameter."""
        from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
            ParameterExtraction,
        )
        from pydantic import BaseModel

        class Input(BaseModel):
            account_uid: str

        utterance = "open_account_by_uid <account_uid>3f2a</account_uid>"
        extracted = ParameterExtraction._extract_parameters_from_xml(utterance, Input)
        assert extracted.account_uid == auto_navigation.xml_parameter(
            utterance, "account_uid")

    def test_all_required_parameters_or_none(self):
        """One of two is not "nearly enough": the missing one would have to be
        guessed, and guessing is the thing the rule forbids."""
        decision = decide(
            command_name="show_findings",
            utterance="show_findings <report_uid>R-1</report_uid>",
            owner_contexts=["Report"], contracts={"Report": REPORT},
        )
        assert decision.kind == CLARIFY
        assert decision.missing_parameters == ("revision",)

    def test_a_value_in_the_utterance_beats_an_absent_registry(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions <account_uid>3f2a</account_uid>",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT}, entries=(),
        )
        assert decision.rule == RULE_UTTERANCE_PARAMETERS


class TestRuleThreeAnExplicitHandle:
    def test_an_o_alias_the_agent_wrote_resolves_to_the_entry_it_denotes(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions O4",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(entry(alias="O4", parameters={"account_uid": "3f2a"}),),
        )
        assert decision.kind == DISPATCH
        assert decision.rule == RULE_EXPLICIT_HANDLE
        assert decision.handle == "O4"
        assert decision.entry_utterance == (
            "open_account_by_uid <account_uid>3f2a</account_uid>")

    def test_a_uid_the_agent_wrote_resolves_the_same_way(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions <uid>3f2a</uid>",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(entry(parameters={"account_uid": "3f2a"}),),
        )
        assert decision.rule == RULE_EXPLICIT_HANDLE
        assert decision.handle == "3f2a"

    def test_a_handle_that_denotes_another_context_is_not_used(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions O4",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(entry(context="Identity", alias="O4"),),
        )
        assert decision.kind == CLARIFY

    def test_an_unmentioned_entry_is_never_used(self):
        """The registry is a lookup table, not a history. An entry nothing in
        the utterance names may not be selected because it is the only one."""
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(entry(alias="O4", parameters={"account_uid": "3f2a"}),),
        )
        assert decision.kind == CLARIFY
        assert decision.missing_parameters == ("account_uid",)

    def test_a_handle_matching_two_different_entries_blocks(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions O4 O9",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(
                entry(alias="O4", parameters={"account_uid": "3f2a"}),
                entry(sequence=2, alias="O9", parameters={"account_uid": "91bb"}),
            ),
        )
        assert decision.kind == CLARIFY
        assert decision.reason == auto_navigation.REASON_AMBIGUOUS_HANDLE

    def test_an_entry_missing_the_required_parameter_does_not_resolve(self):
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions O4",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(entry(alias="O4", parameters={"identity_uid": "3f2a"}),),
        )
        assert decision.kind == CLARIFY


#: The shape the recorder actually captures: `model_dump()` of the entry
#: command's Input model, so the identifying uid AND every optional parameter
#: that took its default. Built as the registry holds it, so this reproduction
#: does not depend on the recorder having been fixed too.
def recorded_entry(sequence=1, alias="O7", account_uid="3f2a9c1d7b8e4f60",
                   page_size="25"):
    return ContextEntry(
        sequence=sequence, context="Account", command_name="open_account_by_uid",
        parameters={
            "account_uid": account_uid,
            "name_filter": "admin",
            "page_size": page_size,
            "include_disabled": "True",
        },
        alias=alias,
    )


class TestOnlyIdentifiersAreHandles:
    """F31 (ido-nx6). Rule 3 resolves an ``O`` alias or a value the agent wrote
    to DENOTE an instance -- which is what the docs say and what the code did
    not do. Every recorded parameter value was a handle, defaults included, so a
    foreign command carrying the literal ``25`` for its own unrelated limit --
    or prose that merely contained the number -- navigated into an account the
    agent had never named."""

    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    def _decide(self, utterance):
        return decide(
            command_name="list_permissions", utterance=utterance,
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(recorded_entry(),),
        )

    @pytest.mark.parametrize("utterance", [
        "list_permissions <limit>25</limit>",
        "list_permissions 25",
        "list_permissions <include_disabled>True</include_disabled>",
        "list_permissions <name_filter>admin</name_filter>",
    ])
    def test_a_foreign_value_that_is_not_an_identifier_does_not_navigate(
        self, utterance
    ):
        decision = self._decide(utterance)
        assert decision.kind == CLARIFY
        assert decision.handle is None

    def test_prose_carrying_a_recorded_number_does_not_navigate(self):
        """The confirmed report: 'for the top 25 rights' dispatched into an
        account because 25 was some listing's default page size."""
        decision = self._decide("list_permissions for the top 25 rights")
        assert decision.kind == CLARIFY
        assert decision.entry_utterance is None

    def test_the_observation_alias_still_resolves(self):
        decision = self._decide("list_permissions O7")
        assert decision.kind == DISPATCH
        assert decision.rule == RULE_EXPLICIT_HANDLE
        assert decision.handle == "O7"
        assert decision.entry_utterance == (
            "open_account_by_uid <account_uid>3f2a9c1d7b8e4f60</account_uid>")

    def test_a_required_parameter_value_still_resolves(self):
        decision = self._decide("list_permissions 3f2a9c1d7b8e4f60")
        assert decision.kind == DISPATCH
        assert decision.handle == "3f2a9c1d7b8e4f60"

    def test_an_alias_after_a_newline_is_still_a_handle(self):
        """Tokenising on a single space made `list_permissions\nO7` yield no
        tokens at all, so the agent's own alias was invisible."""
        decision = self._decide("list_permissions\nO7")
        assert decision.kind == DISPATCH
        assert decision.handle == "O7"

    @pytest.mark.parametrize("value,is_handle", [
        ("3f2a9c1d7b8e4f60", True),
        ("acct-25", True),
        ("1234567", True),
        ("25", False),
        ("0", False),
        ("-1", False),
        ("3.5", False),
        ("True", False),
        ("false", False),
        ("none", False),
        ("   ", False),
    ])
    def test_what_a_value_has_to_look_like_to_be_a_handle(self, value, is_handle):
        assert auto_navigation.is_handle_value(value) is is_handle

    def test_two_entries_differing_only_in_a_default_are_not_ambiguous(self):
        """Identity is the values that NAME the instance. A second entry into
        the same account with a different page size denotes the same account."""
        first = recorded_entry(alias="O7")
        second = recorded_entry(sequence=2, alias="O9", page_size="50")
        decision = decide(
            command_name="list_permissions", utterance="list_permissions O7 O9",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=(first, second),
        )
        assert decision.kind == DISPATCH
        assert decision.entry_utterance == (
            "open_account_by_uid <account_uid>3f2a9c1d7b8e4f60</account_uid>")

    def test_an_empty_tag_is_not_a_value_the_agent_supplied(self):
        """The smaller half of the same finding: rule 2 accepted an empty tag
        and composed an entry command with an empty tag, which can only fail."""
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions <account_uid> </account_uid>",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT}, entries=(),
        )
        assert decision.kind == CLARIFY
        assert decision.entry_utterance is None
        assert decision.missing_parameters == ("account_uid",)


# ---------------------------------------------------------------------------
# When the framework declines to act at all
# ---------------------------------------------------------------------------

class TestWhenNothingIsDispatched:
    def test_a_context_with_no_entry_declaration_leaves_the_r1_hint_alone(self):
        """ido-pyw.1: the declaration is the switch. A workflow that declares no
        `enter_command` gets the declaration-and-hint behaviour the flag used to
        pin, and gets it because there is nothing to dispatch."""
        decision = decide(
            command_name="list_findings", utterance="list_findings",
            owner_contexts=["ControlsMonitor"], contracts={},
        )
        assert decision.kind == NONE
        assert decision.reason == auto_navigation.REASON_NO_ENTRY_DECLARATION

    def test_a_name_no_context_owns_is_not_this_mechanism_s_business(self):
        decision = decide(
            command_name="list_holders", utterance="list_holders",
            owner_contexts=[], contracts={},
        )
        assert decision.kind == NONE
        assert decision.reason == auto_navigation.REASON_NOT_A_KNOWN_NAME

    def test_two_owning_contexts_are_never_chosen_between(self):
        decision = decide(
            command_name="list_accounts", utterance="list_accounts",
            owner_contexts=["Application", "Identity", "Repository"],
            contracts={"Identity": ACCOUNT},
        )
        assert decision.kind == NONE
        assert decision.reason == auto_navigation.REASON_SEVERAL_OWNING_CONTEXTS

    def test_an_undeclared_context_keeps_the_hint_only_behaviour(self):
        """The 74c348a predecessor: the hint names the context and stops."""
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={},
        )
        assert decision.kind == NONE
        assert decision.reason == auto_navigation.REASON_NO_ENTRY_DECLARATION


# ---------------------------------------------------------------------------
# The property the rule exists for
# ---------------------------------------------------------------------------

#: Histories the decision must be blind to. Each is a plausible turn-so-far: an
#: action log, the observations of earlier execute steps, and a registry holding
#: entries the utterance does not name. None of them may move a decision.
HISTORIES = [
    pytest.param((), id="no-history"),
    pytest.param(
        (entry(alias="O2", parameters={"account_uid": "aaaa"}),),
        id="one-account-entered-and-left"),
    pytest.param(
        (
            entry(alias="O2", parameters={"account_uid": "aaaa"}),
            entry(sequence=2, alias="O5", parameters={"account_uid": "bbbb"}),
            entry(sequence=3, context="Identity", command_name="open_identity_by_uid",
                  parameters={"identity_uid": "cccc"}, alias="O7"),
        ),
        id="three-entries-two-contexts"),
    pytest.param(
        (entry(sequence=9, alias="O31", parameters={"account_uid": "3f2a"}),),
        id="the-answer-is-in-the-log-but-unnamed"),
]

#: The same utterances, decided against each history above.
UTTERANCES = [
    # rule 1: nothing to supply
    ("list_findings", "list_findings", ["ControlsMonitor"],
     {"ControlsMonitor": CONTROLS_MONITOR}),
    # rule 2: the value is in the utterance
    ("list_permissions", "list_permissions <account_uid>3f2a</account_uid>",
     ["Account"], {"Account": ACCOUNT}),
    # the blocking case: the value exists in the turn but the utterance never
    # names it. THIS is the row the whole rule is about -- the last history
    # holds exactly the uid the agent needs, and it still must not be used.
    ("list_permissions", "list_permissions", ["Account"], {"Account": ACCOUNT}),
    # undeclared
    ("list_permissions", "list_permissions", ["Account"], {}),
]


class TestNeverFromHistory:
    @pytest.mark.parametrize("entries", HISTORIES)
    @pytest.mark.parametrize(
        "command_name,utterance,owners,contracts", UTTERANCES,
        ids=[u.split()[0] + "/" + ("declared" if c else "undeclared")
             + ("/with-params" if "<" in u else "")
             for _, u, _, c in UTTERANCES],
    )
    def test_the_same_utterance_and_model_decide_the_same_way(
        self, entries, command_name, utterance, owners, contracts
    ):
        decision = decide(
            command_name=command_name, utterance=utterance,
            owner_contexts=owners, contracts=contracts, entries=entries,
        )
        reference = decide(
            command_name=command_name, utterance=utterance,
            owner_contexts=owners, contracts=contracts, entries=(),
        )
        assert decision == reference

    @pytest.mark.parametrize("entries", HISTORIES)
    def test_a_value_present_in_the_turn_but_unnamed_still_blocks(self, entries):
        """The ido-8ps.6.1 shape: `list_permissions` from Identity, with the
        account uid sitting in the preceding observation and in the agent's own
        plan. 20 of 20 live occurrences. It blocks, every time."""
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=entries,
        )
        assert decision.kind == CLARIFY
        assert decision.missing_parameters == ("account_uid",)
        assert decision.entry_utterance is None

    @pytest.mark.parametrize("observations", [
        [],
        ["O7\n3f2a  Alan Cooper\n91bb  Brandon Miller"],
        ["O7\n3f2a  Alan Cooper", "O8\nNo permissions found."],
        ["O1\n" + "\n".join(f"uid{n}  Person {n}" for n in range(40))],
    ])
    def test_the_candidate_list_never_reaches_the_decision(self, observations):
        """Candidates are composed AFTER the decision, from observations the
        decision function is never given. They change the message; they cannot
        change what the framework does."""
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        text = clarification_text(decision, candidate_values(observations))
        assert decision.kind == CLARIFY
        assert decision.entry_utterance is None
        assert "open_account_by_uid" in text


# ---------------------------------------------------------------------------
# The clarification and its candidates
# ---------------------------------------------------------------------------

class TestTheClarification:
    def test_it_names_the_context_the_entry_command_and_the_missing_parameter(self):
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        text = clarification_text(decision)
        assert "Account context" in text
        assert "open_account_by_uid" in text
        assert "account_uid" in text
        assert auto_navigation.missing_information_errmsg() in text
        assert text.count("account_uid") >= 2

    def test_candidates_are_listed_and_said_to_be_listed(self):
        observations = ["O7 listing\n3f2a  Alan Cooper\n91bb  Brandon Miller"]
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        text = clarification_text(decision, candidate_values(observations))
        assert "3f2a" in text and "91bb" in text
        assert "listed, not chosen" in text

    def test_candidates_come_off_the_framework_s_own_row_format_newest_first(self):
        observations = [
            "O1\naaaa  Old Person",
            "O2\nbbbb  New Person\ncccc  Other Person",
        ]
        assert candidate_values(observations, limit=10) == ["bbbb", "cccc", "aaaa"]

    def test_the_candidate_list_is_bounded(self):
        rows = "\n".join(f"uid{n}  Person {n}" for n in range(50))
        assert len(candidate_values([rows], limit=4)) == 4

    def test_no_observations_means_no_candidates_and_still_a_clarification(self):
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        text = clarification_text(decision, candidate_values([]))
        assert "listed, not chosen" not in text
        assert "open_account_by_uid" in text


# ---------------------------------------------------------------------------
# The turn-scoped registry
# ---------------------------------------------------------------------------

class TestTheRegistry:
    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    def test_an_entry_is_recorded_under_its_scope_and_nowhere_else(self):
        auto_navigation.record_context_entry(
            "turn-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "3f2a"}, alias="O4")
        assert len(auto_navigation.context_entries("turn-a")) == 1
        assert auto_navigation.context_entries("turn-b") == ()

    def test_empty_values_are_not_handles(self):
        recorded = auto_navigation.record_context_entry(
            "turn-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "3f2a", "note": "  ", "absent": None},
            alias="O4", required_parameters=("account_uid",))
        assert dict(recorded.parameters) == {"account_uid": "3f2a"}
        assert set(recorded.handles()) == {"O4", "3f2a"}

    def test_an_entry_recorded_without_a_contract_publishes_its_alias_only(self):
        """ido-nx6. Which values NAME the instance is the contract's answer. An
        entry recorded without one may not guess that every value it was handed
        does, because the dump it is handed includes the defaults."""
        recorded = auto_navigation.record_context_entry(
            "turn-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "3f2a", "page_size": 25}, alias="O4")
        assert set(recorded.handles()) == {"O4"}

    def test_nothing_survives_the_turn(self):
        auto_navigation.record_context_entry(
            "turn-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "3f2a"})
        auto_navigation.reset_auto_navigation_state()
        assert auto_navigation.context_entries("turn-a") == ()

    def test_the_offloading_reset_takes_the_registry_with_it(self):
        """One turn boundary, not two: a handle written in one turn must never
        resolve to a context instance another turn entered."""
        from fastworkflow.observation_offloading.state import reset_runtime_state

        auto_navigation.record_context_entry(
            "turn-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "3f2a"})
        reset_runtime_state()
        assert auto_navigation.context_entries("turn-a") == ()


class TestTheProcessDefaultScopeResolvesNothing:
    """ido-bhf (F37): a scope that cannot tell turn 1 from turn 2.

    Every scope a host binds is minted per turn -- the agent's
    ``continuation_scope`` carries the turn key its observations are archived
    under, and a trace host's is rebuilt from the claim ``_begin_turn`` mints.
    The process default is one object per process: the same id for every turn,
    while the ``O`` aliases the agent writes restart at ``O1`` each trajectory.
    ``O3`` in turn 2 would therefore resolve to whatever turn 1 entered under
    ``O3`` and dispatch to the wrong instance.
    """

    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    @property
    def _default_scope_id(self):
        from fastworkflow.observation_offloading.state import default_scope

        return str(default_scope().scope_id)

    def test_the_default_scope_and_the_unbound_fallback_are_recognised(self):
        assert auto_navigation.is_process_default_scope(self._default_scope_id)
        assert auto_navigation.is_process_default_scope(
            auto_navigation.UNBOUND_SCOPE_ID)
        assert not auto_navigation.is_process_default_scope("turn-a")

    def test_a_handle_recorded_under_the_default_scope_does_not_resolve(self):
        """The c9 script's shape: turn 1 entered an account under O3, turn 2
        writes O3 meaning something else entirely."""
        scope_id = self._default_scope_id
        auto_navigation.record_context_entry(
            scope_id, context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "AAAA000000000001"}, alias="O3",
            required_parameters=("account_uid",))

        assert auto_navigation.context_entries(scope_id) == ()
        decision = decide(
            command_name="list_permissions", utterance="list_permissions O3",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=auto_navigation.context_entries(scope_id))
        assert decision.kind == CLARIFY
        assert decision.reason == auto_navigation.REASON_MISSING_ENTRY_PARAMETERS

    def test_the_entry_is_still_recorded_and_still_reclaimed(self):
        """Resolution is refused, not recording: the residency reclamation and
        the durable copy both still see what the turn did."""
        scope_id = self._default_scope_id
        auto_navigation.record_context_entry(
            scope_id, context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "AAAA000000000001"}, alias="O3")
        assert len(auto_navigation._entries[scope_id]) == 1
        auto_navigation.forget_scope(scope_id)
        assert scope_id not in auto_navigation._entries

    def test_a_turn_scope_still_resolves_its_own_handles(self):
        """The refusal is the default scope's alone. A real host mints a turn
        key per turn, and rule 3 is exactly as it was there."""
        auto_navigation.record_context_entry(
            "turn-2026-a", context="Account", command_name="open_account_by_uid",
            parameters={"account_uid": "AAAA000000000001"}, alias="O3",
            required_parameters=("account_uid",))
        decision = decide(
            command_name="list_permissions", utterance="list_permissions O3",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=auto_navigation.context_entries("turn-2026-a"))
        assert decision.kind == DISPATCH
        assert decision.rule == auto_navigation.RULE_EXPLICIT_HANDLE

    def test_rules_1_and_2_are_untouched_under_the_default_scope(self):
        """They read the utterance, which no scope can distort."""
        decision = decide(
            command_name="list_permissions",
            utterance="list_permissions <account_uid>AAAA000000000002</account_uid>",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
            entries=auto_navigation.context_entries(self._default_scope_id))
        assert decision.kind == DISPATCH
        assert decision.rule == auto_navigation.RULE_UTTERANCE_PARAMETERS


# ---------------------------------------------------------------------------
# The validator (part a), offline
# ---------------------------------------------------------------------------

def _declare_enter_command(workflow_path, context_name: str, declaration) -> None:
    """Write an `enter_command` declaration into a COPY of the test workflow."""
    path = workflow_path / "_commands" / context_name / f"_{context_name}.py"
    source = path.read_text()
    marker = "class Context:\n"
    assert marker in source, path
    path.write_text(source.replace(
        marker, f"class Context:\n    enter_command = {declaration!r}\n", 1))


@pytest.fixture
def workflow_copy(tmp_path):
    """A writable copy of the real todo_list_workflow, with real classes."""
    workflow_path = tmp_path / "todo_list_workflow"
    shutil.copytree(
        os.path.join(os.path.dirname(__file__), "todo_list_workflow"),
        workflow_path,
        ignore=shutil.ignore_patterns(
            "___command_info", "___workflow_contexts", "___convo_info", "__pycache__",
        ),
    )
    return workflow_path


class TestTheValidator:
    def test_a_workflow_that_declares_nothing_is_valid_and_says_so(
        self, workflow_copy, setup_test_environment
    ):
        report = validate_entry_contracts(str(workflow_copy))
        assert report.ok
        assert report.contracts == {}
        assert "no context declares enter_command" in report.render()

    def test_a_parameterised_entry_is_classified_by_its_required_parameters(
        self, workflow_copy, setup_test_environment
    ):
        """`get_todo_list <id>` is owned by TodoListManager, the PARENT of
        TodoList -- reachable, and parameterised because `id` is required."""
        _declare_enter_command(workflow_copy, "TodoList", "get_todo_list <id>")
        report = validate_entry_contracts(str(workflow_copy))

        assert report.ok, report.render()
        assert report.parameterised_contexts == {"TodoList": ("id",)}
        assert report.stateless_contexts == ()
        assert report.contracts["TodoList"].command_name == "get_todo_list"
        assert report.contracts["TodoList"].owner_contexts == ("TodoListManager",)

    def test_an_entry_command_with_no_parameters_is_stateless(
        self, workflow_copy, setup_test_environment
    ):
        _declare_enter_command(workflow_copy, "TodoListManager", "list_todo_lists")
        report = validate_entry_contracts(str(workflow_copy))

        assert report.ok, report.render()
        assert report.stateless_contexts == ("TodoListManager",)

    def test_a_command_the_workflow_does_not_own_is_an_error(
        self, workflow_copy, setup_test_environment
    ):
        _declare_enter_command(workflow_copy, "TodoList", "open_todo_list <id>")
        report = validate_entry_contracts(str(workflow_copy))

        assert not report.ok
        assert [i.code for i in report.issues] == [
            auto_navigation.ISSUE_UNKNOWN_COMMAND]
        assert "open_todo_list" in report.render()

    def test_an_owner_the_declaring_context_cannot_reach_is_an_error(
        self, workflow_copy, setup_test_environment
    ):
        """`mark_completed` is a TodoList command. TodoListManager is TodoList's
        PARENT, so the entry command would have to be run from inside the very
        context it is supposed to enter."""
        _declare_enter_command(workflow_copy, "TodoListManager", "mark_completed")
        report = validate_entry_contracts(str(workflow_copy))

        assert not report.ok
        assert [i.code for i in report.issues] == [
            auto_navigation.ISSUE_UNREACHABLE_OWNER]

    def test_a_declaration_that_does_not_parse_is_an_error(
        self, workflow_copy, setup_test_environment
    ):
        _declare_enter_command(workflow_copy, "TodoList", "<id>")
        report = validate_entry_contracts(str(workflow_copy))

        assert not report.ok
        assert [i.code for i in report.issues] == [auto_navigation.ISSUE_UNPARSEABLE]

    def test_two_entry_commands_are_refused_rather_than_chosen_between(
        self, workflow_copy, setup_test_environment
    ):
        _declare_enter_command(
            workflow_copy, "TodoList", ["get_todo_list <id>", "get_child_by_id <id>"])
        report = validate_entry_contracts(str(workflow_copy))

        assert not report.ok
        assert [i.code for i in report.issues] == [
            auto_navigation.ISSUE_SEVERAL_DECLARATIONS]
        assert report.contracts == {}

    def test_the_runtime_reads_the_same_contract_the_validator_reports(
        self, workflow_copy, setup_test_environment
    ):
        """One source. A validator that agreed with nothing at runtime would be
        a second opinion, which is what the single-declaration rule exists to
        prevent."""
        _declare_enter_command(workflow_copy, "TodoList", "get_todo_list <id>")
        report = validate_entry_contracts(str(workflow_copy))
        live = auto_navigation.entry_contract_for(str(workflow_copy), "TodoList")
        assert live == report.contracts["TodoList"]

    def test_a_refused_declaration_yields_no_runtime_contract_either(
        self, workflow_copy, setup_test_environment
    ):
        _declare_enter_command(
            workflow_copy, "TodoList", ["get_todo_list <id>", "get_child_by_id <id>"])
        assert auto_navigation.entry_contract_for(str(workflow_copy), "TodoList") is None


# ---------------------------------------------------------------------------
# The CME seam: what the wildcard command does with the decision
# ---------------------------------------------------------------------------

class _AppWorkflow:
    """The parent chain, as `wildcard.ResponseGenerator` walks it."""

    def __init__(self, chain, folderpath=""):
        self._chain = list(chain)
        self.folderpath = folderpath
        self.current_command_context = self._chain[0]
        self.command_context_for_response_generation = self._chain[0]
        self.current_command_context_name = self._chain[0]
        self.context = {"run_as_agent": True}

    @property
    def is_command_context_for_response_generation_root(self):
        return self.command_context_for_response_generation == self._chain[-1]

    def get_parent(self, context_object):
        index = self._chain.index(context_object)
        return self._chain[min(index + 1, len(self._chain) - 1)]


class _CmeWorkflow:
    def __init__(self, app_workflow):
        self._context = {"app_workflow": app_workflow}

    @property
    def context(self):
        return self._context

    @context.setter
    def context(self, value):
        self._context = value

    def end_command_processing(self):
        self._context["NLU_Pipeline_Stage"] = INTENT_DETECTION


def _predictor_declining(hint, owners):
    class Predictor:
        def __init__(self, cme_workflow):
            pass

        def predict(self, context_name, command, stage):
            from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
                CommandNamePrediction,
            )
            return CommandNamePrediction.Output(
                command_name=None, known_name_owner_contexts=list(owners),
                routing_hint=hint,
            )
    return Predictor


@pytest.fixture
def declining_wildcard(monkeypatch):
    """The wildcard command with a predictor that declines everywhere."""
    def run(utterance, owners, *, hint="hint", folderpath=""):
        monkeypatch.setattr(
            wildcard_command, "CommandNamePrediction",
            _predictor_declining(hint, owners))
        monkeypatch.setattr(
            fastworkflow.Workflow, "get_command_context_name",
            staticmethod(lambda context_object: context_object))
        monkeypatch.setattr(
            wildcard_command.CommandExecutor, "perform_action",
            staticmethod(lambda workflow, action: fastworkflow.CommandOutput(
                command_response=fastworkflow.CommandResponse(
                    response="I couldn't determine which available command "
                             "matches your request.",
                    success=False))))
        app_workflow = _AppWorkflow(["Identity", "*"], folderpath=folderpath)
        return wildcard_command.ResponseGenerator()(
            _CmeWorkflow(app_workflow), utterance)
    return run


class TestTheWildcardSeam:
    def setup_method(self):
        auto_navigation.reset_auto_navigation_state()

    def teardown_method(self):
        auto_navigation.reset_auto_navigation_state()

    def test_a_dispatchable_name_leaves_a_plan_for_the_executor(
        self, declining_wildcard, monkeypatch, setup_test_environment
    ):
        monkeypatch.setattr(
            auto_navigation, "entry_contract_for",
            lambda folderpath, context: CONTROLS_MONITOR if context == "ControlsMonitor" else None)

        output = declining_wildcard("list_findings", ["ControlsMonitor"])
        plan = output.command_response.artifacts[auto_navigation.AUTO_NAVIGATION_ARTIFACT]

        assert output.command_response.artifacts["command_handled"] is True
        assert plan["rule"] == RULE_STATELESS
        assert plan["entry_utterance"] == "open_controls_monitor"
        assert plan["original_utterance"] == "list_findings"
        assert plan["entered_context"] == "ControlsMonitor"

    def test_a_blocking_case_says_what_it_needs_instead_of_planning(
        self, declining_wildcard, monkeypatch, setup_test_environment
    ):
        monkeypatch.setattr(
            auto_navigation, "entry_contract_for",
            lambda folderpath, context: ACCOUNT if context == "Account" else None)

        output = declining_wildcard("list_permissions", ["Account"])
        response = output.command_response.response

        assert auto_navigation.AUTO_NAVIGATION_ARTIFACT not in output.command_response.artifacts
        assert auto_navigation.missing_information_errmsg() in response
        assert "account_uid" in response
        assert "open_account_by_uid" in response
        assert output.success is False

    def test_a_context_with_no_declaration_keeps_the_r1_hint_exactly(
        self, declining_wildcard, monkeypatch, setup_test_environment
    ):
        """ido-pyw.1. `entry_contract_for` finds nothing, so the dispatcher
        declines and the R1 hint is what the agent is told, byte for byte."""
        monkeypatch.setattr(
            auto_navigation, "entry_contract_for", lambda folderpath, context: None)
        hint = intent_detection.foreign_context_hint(
            "list_findings", ["ControlsMonitor"], ["open_controls_monitor"])

        output = declining_wildcard("list_findings", ["ControlsMonitor"], hint=hint)

        assert auto_navigation.AUTO_NAVIGATION_ARTIFACT not in output.command_response.artifacts
        assert output.command_response.response.endswith(hint)

    def test_free_text_is_untouched(
        self, declining_wildcard, monkeypatch, setup_test_environment
    ):
        """A name the workflow does not own has no owners, so there is nothing
        to navigate to and the ordinary message stands."""
        output = declining_wildcard(
            "which identities hold the compliance officer permission", [], hint=None)
        assert output.command_response.response == (
            "I couldn't determine which available command matches your request.")


class TestARootCommandIsUnaffected:
    """`fetch_result_page` is a `*` command: the walk reaches it, so the
    dispatcher is never consulted. ido-8ps.6.1's own case."""

    @pytest.fixture
    def ido_contexts(self):
        return json.loads(INVENTORY_FIXTURE.read_text())["contexts"]

    def test_the_owner_is_on_the_walk_so_the_walk_resolves_it(
        self, ido_contexts, setup_test_environment
    ):
        assert "fetch_result_page" in ido_contexts["*"]
        # The guard declines it in Permission, the walk carries it to '*', and
        # '*' owns it -- so `command_name` is set before the post-walk block the
        # dispatcher lives in is ever reached.
        decision = decide(
            command_name="fetch_result_page",
            utterance="fetch_result_page <handle>O9</handle> <contains>Alan Cooper</contains>",
            owner_contexts=["*"], contracts={},
        )
        assert decision.kind == NONE

    def test_the_wildcard_never_plans_when_the_walk_resolved(
        self, monkeypatch, setup_test_environment
    ):
        """A resolved prediction leaves the post-walk block unreached; the
        response carries no plan."""

        class Resolving:
            def __init__(self, cme_workflow):
                pass

            def predict(self, context_name, command, stage):
                from fastworkflow._workflows.command_metadata_extraction.intent_detection import (
                    CommandNamePrediction,
                )
                return CommandNamePrediction.Output(command_name="fetch_result_page")

        monkeypatch.setattr(wildcard_command, "CommandNamePrediction", Resolving)
        monkeypatch.setattr(
            fastworkflow.Workflow, "get_command_context_name",
            staticmethod(lambda context_object: context_object))

        class Extractor:
            def __init__(self, *args, **kwargs):
                pass

            def extract(self):
                from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import (
                    ParameterExtraction,
                )
                return ParameterExtraction.Output(
                    parameters_are_valid=True, cmd_parameters=None)

        monkeypatch.setattr(wildcard_command, "ParameterExtraction", Extractor)
        app_workflow = _AppWorkflow(["Permission", "*"])
        output = wildcard_command.ResponseGenerator()(
            _CmeWorkflow(app_workflow),
            "fetch_result_page <handle>O9</handle>")

        assert auto_navigation.AUTO_NAVIGATION_ARTIFACT not in output.command_response.artifacts
        assert output.command_response.artifacts["command_name"] == "fetch_result_page"


# ---------------------------------------------------------------------------
# The executor seam: two real steps, in order
# ---------------------------------------------------------------------------

def _output(response, success=True):
    return fastworkflow.CommandOutput(
        command_response=fastworkflow.CommandResponse(
            response=response, success=success))


class _Session:
    """A session whose current context the entry step is expected to move.

    `_auto_navigate` reads the context name either side of the entry step,
    because whether a declared entry command really ENTERS is a runtime fact the
    offline validator cannot check.
    """

    def __init__(self, contexts):
        self._contexts = list(contexts)

    def advance(self):
        if len(self._contexts) > 1:
            self._contexts.pop(0)

    def get_active_workflow(self):
        session = self

        class Workflow:
            current_command_context_name = session._contexts[0]

        return Workflow


class TestTheTwoStepDispatch:
    PLAN = {
        "rule": RULE_UTTERANCE_PARAMETERS,
        "entered_context": "Account",
        "entry_command": "open_account_by_uid",
        "entry_utterance": "open_account_by_uid <account_uid>3f2a</account_uid>",
        "original_utterance": "list_permissions <account_uid>3f2a</account_uid>",
        "command_name": "list_permissions",
        "handle": None,
    }

    def _recorder(self, monkeypatch, outputs):
        from fastworkflow.command_executor import CommandExecutor

        calls = []

        def invoke_command(chat_session, command, *, auto_navigation_step=None):
            calls.append((command, auto_navigation_step))
            advance = getattr(chat_session, "advance", None)
            if advance is not None:
                advance()
            return outputs[len(calls) - 1]

        monkeypatch.setattr(
            CommandExecutor, "invoke_command", classmethod(
                lambda cls, chat_session, command, *, auto_navigation_step=None:
                    invoke_command(chat_session, command,
                                   auto_navigation_step=auto_navigation_step)))
        return calls

    def test_the_entry_command_runs_first_then_the_original(
        self, monkeypatch, setup_test_environment
    ):
        from fastworkflow.command_executor import CommandExecutor

        calls = self._recorder(monkeypatch, [
            _output("You are now in the Account context."),
            _output("2 permissions."),
        ])
        output = CommandExecutor._auto_navigate(
            _Session(["Identity", "Account"]), self.PLAN)

        assert [command for command, _ in calls] == [
            self.PLAN["entry_utterance"], self.PLAN["original_utterance"]]
        assert "You are now in the Account context." in output.command_response.response
        assert "2 permissions." in output.command_response.response

    def test_both_steps_carry_the_rule_that_fired(
        self, monkeypatch, setup_test_environment
    ):
        """Each is an ordinary execute step -- own span, own observation, own
        execution record (A1/A2). These three attributes are the only thing that
        says the agent did not type it."""
        from fastworkflow.command_executor import CommandExecutor

        calls = self._recorder(monkeypatch, [_output("entered"), _output("done")])
        CommandExecutor._auto_navigate(
            _Session(["Identity", "Account"]), self.PLAN)

        for _, marks in calls:
            assert marks[auto_navigation.ATTR_AUTO_NAVIGATED] is True
            assert marks[auto_navigation.ATTR_AUTO_NAVIGATION_RULE] == (
                RULE_UTTERANCE_PARAMETERS)
            assert marks[auto_navigation.ATTR_ENTERED_CONTEXT] == "Account"
        assert [marks[auto_navigation.ATTR_AUTO_NAVIGATION_STEP] for _, marks in calls] == [
            auto_navigation.STEP_ENTRY, auto_navigation.STEP_ORIGINAL]

    def test_a_failed_entry_stops_the_dispatch(
        self, monkeypatch, setup_test_environment
    ):
        """Running the original command anyway would be acting on a guess about
        where the failed entry left the context."""
        from fastworkflow.command_executor import CommandExecutor

        calls = self._recorder(monkeypatch, [
            _output("No account with that uid.", success=False),
            _output("must not run"),
        ])
        output = CommandExecutor._auto_navigate(
            _Session(["Identity", "Account"]), self.PLAN)

        assert len(calls) == 1
        assert "was not run" in output.command_response.response
        assert "No account with that uid." in output.command_response.response
        assert output.success is False

    def test_the_inner_steps_do_not_dispatch_again(
        self, monkeypatch, setup_test_environment
    ):
        """The rule composes two steps, not a search."""
        from fastworkflow.command_executor import CommandExecutor

        depths = []

        def invoke_command(cls, chat_session, command, *, auto_navigation_step=None):
            depths.append(auto_navigation.dispatch_in_flight())
            chat_session.advance()
            return _output("ok")

        monkeypatch.setattr(
            CommandExecutor, "invoke_command", classmethod(invoke_command))
        CommandExecutor._auto_navigate(
            _Session(["Identity", "Account"]), self.PLAN)

        assert depths == [True, True]
        assert auto_navigation.dispatch_in_flight() is False

    def test_an_entry_that_did_not_move_the_context_stops_the_dispatch(
        self, monkeypatch, setup_test_environment
    ):
        """IDO's `open_finding <finding_uid>`: the parameter has a default, so
        the command succeeds by LISTING findings rather than entering one. The
        validator classifies it stateless and rule 1 fires; this is the runtime
        check that keeps the second step from running in the context that had
        already declined it."""
        from fastworkflow.command_executor import CommandExecutor

        calls = self._recorder(monkeypatch, [
            _output("3 findings: ..."), _output("must not run")])
        output = CommandExecutor._auto_navigate(
            _Session(["ControlsMonitor"]), self.PLAN)

        assert len(calls) == 1
        assert "did not enter" in output.command_response.response
        assert "was not run" in output.command_response.response
        assert output.success is False

    def test_a_session_that_cannot_name_its_context_does_not_block_the_dispatch(
        self, monkeypatch, setup_test_environment
    ):
        """Not knowing is not evidence that nothing moved."""
        from fastworkflow.command_executor import CommandExecutor

        calls = self._recorder(monkeypatch, [_output("entered"), _output("done")])
        CommandExecutor._auto_navigate(object(), self.PLAN)
        assert len(calls) == 2

    def test_a_plan_is_not_made_while_one_is_in_flight(self, setup_test_environment):
        with auto_navigation.dispatching():
            decision = auto_navigation.plan(
                "", command_name="list_findings", utterance="list_findings",
                owner_contexts=["ControlsMonitor"])
        assert decision.kind == NONE
        assert decision.reason == auto_navigation.REASON_DISPATCH_IN_FLIGHT


# ---------------------------------------------------------------------------
# The routing event
# ---------------------------------------------------------------------------

class TestTheRoutingEvent:
    def test_the_event_names_the_decision_and_its_reason(self):
        decision = decide(
            command_name="list_findings", utterance="list_findings",
            owner_contexts=["ControlsMonitor"],
            contracts={"ControlsMonitor": CONTROLS_MONITOR},
        )
        event = decision.event()
        assert event[auto_navigation.ATTR_AUTO_NAVIGATION_DECISION] == decision.kind
        assert event[auto_navigation.ATTR_AUTO_NAVIGATION_RULE] == RULE_STATELESS
        assert event[auto_navigation.ATTR_ENTERED_CONTEXT] == "ControlsMonitor"
        assert event["reason"] == auto_navigation.REASON_STATELESS

    def test_the_attribute_names_match_the_declared_span_contract(self):
        """The emission sites write literal keys, because the span-contract scan
        reads them statically; these constants are what every consumer reads.
        Two spellings of one name is one name plus a drift, so they are pinned
        against the declaration itself."""
        from fastworkflow import tracing

        declared = tracing.SPAN_CONTRACTS[tracing.SPAN_COMMAND_EXECUTE].attributes
        for constant in (
            auto_navigation.ATTR_AUTO_NAVIGATED,
            auto_navigation.ATTR_AUTO_NAVIGATION_RULE,
            auto_navigation.ATTR_ENTERED_CONTEXT,
            auto_navigation.ATTR_AUTO_NAVIGATION_STEP,
        ):
            assert constant in declared
        # ido-pyw.1: the flag attribute is gone from the NLU span with the flag.
        assert "auto_navigation_enabled" not in (
            tracing.SPAN_CONTRACTS[tracing.SPAN_NLU_INTENT].attributes)

    def test_a_clarification_event_names_what_was_missing(self):
        decision = decide(
            command_name="list_permissions", utterance="list_permissions",
            owner_contexts=["Account"], contracts={"Account": ACCOUNT},
        )
        event = decision.event()
        assert event[auto_navigation.ATTR_AUTO_NAVIGATION_DECISION] == CLARIFY
        assert event["missing_parameters"] == ["account_uid"]
        assert event["entry_command"] == "open_account_by_uid"
        assert event[auto_navigation.ATTR_AUTO_NAVIGATION_RULE] is None
