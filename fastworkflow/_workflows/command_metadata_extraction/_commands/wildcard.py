import fastworkflow
from fastworkflow import Action, CommandOutput, CommandResponse, NLUPipelineStage
from fastworkflow import auto_navigation
from fastworkflow.command_executor import CommandExecutor
from fastworkflow.nlu_labels import PARAMETER_VALUE_PLACEHOLDERS

from ..intent_detection import CommandNamePrediction
from ..parameter_extraction import ParameterExtraction

#: The artifact key the plan travels under; owned by `auto_navigation` so the
#: executor can read it without importing this command module.
AUTO_NAVIGATION_ARTIFACT = auto_navigation.AUTO_NAVIGATION_ARTIFACT


def _record_auto_navigation(decision) -> None:
    """File the routing event for a declined KNOWN name.

    Only for a name the workflow really owns: ordinary free text that no context
    could route is not an auto-navigation opportunity, and an event per
    unroutable sentence would bury the ones that are. Best effort -- a measure
    must never fail a turn.
    """
    if not decision.owner_contexts:
        return
    try:
        from fastworkflow.observation_offloading.state import record_event

        record_event(decision.event())
    except Exception:  # noqa: BLE001 - a measure must never fail a turn
        pass


class Signature:
    # These are the PARAMETER_EXTRACTION stage's bare-value literals. They belong
    # to nlu_labels.PARAMETER_VALUE_LABEL, not to this command's INTENT_DETECTION
    # escalation class; see fastworkflow/nlu_labels.py for the stage split.
    plain_utterances = list(PARAMETER_VALUE_PLACEHOLDERS)

    @staticmethod
    def generate_utterances(workflow: fastworkflow.Workflow, command_name: str) -> list[str]:
        # Only the humanised command name. The placeholders above are excluded on
        # purpose: training them into the 'wildcard' label taught the escalation
        # classifier that a bare value like "france" means "escalate to my
        # parent". The trainer labels them under PARAMETER_VALUE_LABEL instead.
        return [
            command_name.split('/')[-1].lower().replace('_', ' ')
        ]


class ResponseGenerator:
    def __call__(
        self, 
        workflow: fastworkflow.Workflow, 
        command: str,
    ) -> CommandOutput:  # sourcery skip: hoist-if-from-if
        app_workflow = workflow.context["app_workflow"]   # type: fastworkflow.Workflow
        cmd_ctxt_obj_name = app_workflow.current_command_context_name
        nlu_pipeline_stage = workflow.context.get(
            "NLU_Pipeline_Stage", 
            NLUPipelineStage.INTENT_DETECTION)

        predictor = CommandNamePrediction(workflow)           
        cnp_output = predictor.predict(cmd_ctxt_obj_name, command, nlu_pipeline_stage)

        if cnp_output.error_msg:
            workflow_context = workflow.context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION
            workflow_context["command"] = command
            workflow.context = workflow_context
            return CommandOutput(
                command_response=
                    CommandResponse(
                        response=(
                            f"Ambiguous intent error for command '{command}'\n"
                            f"{cnp_output.error_msg}"
                        ),
                        success=False
                    )
            )
        else:
            if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION and \
                cnp_output.command_name != 'ErrorCorrection/you_misunderstood':
                workflow_context = workflow.context
                workflow_context["command"] = command
                workflow.context = workflow_context
        
        if cnp_output.is_cme_command:
            workflow_context = workflow.context
            if cnp_output.command_name == 'ErrorCorrection/you_misunderstood':
                workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
                workflow_context["command"] = command
            elif (
                nlu_pipeline_stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION or
                cnp_output.command_name == 'ErrorCorrection/abort'
            ):
                workflow.end_command_processing()
            workflow.context = workflow_context

            startup_action = Action(
                command_name=cnp_output.command_name,
                command=command,
            )
            command_output = CommandExecutor.perform_action(workflow, startup_action)
            if (
                nlu_pipeline_stage == fastworkflow.NLUPipelineStage.INTENT_DETECTION or
                cnp_output.command_name == 'ErrorCorrection/abort'
            ):
                command_output.command_response.artifacts["command_handled"] = True     
                # Set the additional attributes
                command_output.command_name = cnp_output.command_name
            return command_output
        
        if nlu_pipeline_stage in {
                NLUPipelineStage.INTENT_DETECTION,
                NLUPipelineStage.INTENT_AMBIGUITY_CLARIFICATION,
                NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
            }:
            app_workflow.command_context_for_response_generation = \
                app_workflow.current_command_context

            if cnp_output.command_name is None:
                # R1 (ido-8ps.8): the hint the first declining context composed.
                # Every context on the chain declines the same token for the
                # same reason, so the first one is the whole story; it is kept
                # rather than recomputed because the walk overwrites cnp_output.
                routing_hint = cnp_output.routing_hint
                # ido-8ps.9: the owners travel with the hint, for the same
                # reason -- the dispatcher below needs the context model's
                # answer to "who owns this name", and only the first declining
                # context still has it once the walk has moved on.
                owner_contexts = cnp_output.known_name_owner_contexts
                while not cnp_output.command_name and \
                    app_workflow.command_context_for_response_generation is not None and \
                        not app_workflow.is_command_context_for_response_generation_root:
                    app_workflow.command_context_for_response_generation = \
                        app_workflow.get_parent(app_workflow.command_context_for_response_generation)
                    cnp_output = predictor.predict(
                        fastworkflow.Workflow.get_command_context_name(app_workflow.command_context_for_response_generation), 
                        command, nlu_pipeline_stage)
                    routing_hint = routing_hint or cnp_output.routing_hint
                    owner_contexts = owner_contexts or cnp_output.known_name_owner_contexts
            
                if cnp_output.command_name is None:
                    if nlu_pipeline_stage == NLUPipelineStage.INTENT_DETECTION:
                        # ido-8ps.9: the walk is exhausted, so the owning context
                        # is genuinely unreachable from here. THIS is the only
                        # point at which two-step dispatch may be considered --
                        # a root ('*') command, or any name an ancestor owns, has
                        # already been resolved by the walk above and never gets
                        # here.
                        #
                        # The decision is a pure function of this utterance and
                        # the context model (`auto_navigation.decide`); the
                        # registry it consults answers only "what does the handle
                        # the agent just wrote denote". Nothing below reads the
                        # action log.
                        decision = auto_navigation.plan(
                            getattr(app_workflow, 'folderpath', ''),
                            command_name=command.split(" ", 1)[0].split("(", 1)[0].lower(),
                            utterance=command,
                            owner_contexts=owner_contexts or [],
                        )
                        _record_auto_navigation(decision)

                        if decision.dispatches:
                            # The two steps run through the ordinary command
                            # path, one at a time, in `CommandExecutor` -- the
                            # CME cannot run them itself (it has the workflow,
                            # not the session), and running them anywhere but
                            # the normal step path would cost them their spans
                            # and their observations. The plan travels as an
                            # artifact; `invoke_command` executes it.
                            workflow.end_command_processing()
                            return CommandOutput(
                                command_response=CommandResponse(
                                    response="",
                                    artifacts={
                                        "command_handled": True,
                                        AUTO_NAVIGATION_ARTIFACT: {
                                            "rule": decision.rule,
                                            "entered_context": decision.entered_context,
                                            "entry_command": decision.entry_command,
                                            "entry_utterance": decision.entry_utterance,
                                            "original_utterance": command,
                                            "command_name": decision.command_name,
                                            "handle": decision.handle,
                                        },
                                    },
                                )
                            )

                        if decision.kind == auto_navigation.CLARIFY:
                            # Blocking, by the rule: the framework says what it
                            # needs and acts only on what comes back. Candidates
                            # are read off recent observations and LISTED -- the
                            # decision above was made without them.
                            routing_hint = auto_navigation.clarification_text(
                                decision,
                                auto_navigation.candidate_values(
                                    auto_navigation.recent_execute_observations()
                                ),
                            )

                        # out of scope commands
                        workflow_context = workflow.context
                        workflow_context["NLU_Pipeline_Stage"] = \
                            NLUPipelineStage.INTENT_MISUNDERSTANDING_CLARIFICATION
                        workflow_context["command"] = command
                        workflow.context = workflow_context

                        startup_action = Action(
                            command_name='ErrorCorrection/you_misunderstood',
                            command=command,
                        )
                        command_output = CommandExecutor.perform_action(workflow, startup_action)
                        # The name IS a command of this workflow; the walk simply
                        # never passed a context that owns it. "Nothing matched"
                        # is true and useless here, so say where it lives and how
                        # to get there. A hint only: nothing below navigates, and
                        # navigating on a guess about what was meant would change
                        # the workflow's state on the strength of that guess
                        # (ido-8ps.8, owner-approved scope addition).
                        if routing_hint:
                            response = command_output.command_response
                            response.response = f"{response.response}\n\n{routing_hint}"
                        command_output.command_response.artifacts["command_handled"] = True
                        return command_output

                    return CommandOutput(
                        command_response=
                            CommandResponse(
                                response=cnp_output.error_msg,
                                success=False
                            )
                    )

            # move to the parameter extraction stage
            workflow_context = workflow.context
            workflow_context["NLU_Pipeline_Stage"] = NLUPipelineStage.PARAMETER_EXTRACTION
            workflow.context = workflow_context

        if nlu_pipeline_stage == NLUPipelineStage.PARAMETER_EXTRACTION:
            cnp_output.command_name = workflow.context["command_name"]
        else:
            workflow_context = workflow.context
            workflow_context["command_name"] = cnp_output.command_name
            workflow.context = workflow_context

        command_name = cnp_output.command_name
        # Use the preserved original command (with parameters) if available
        preserved_command = f'{command_name}: {workflow.context.get("command", command)}'
        extractor = ParameterExtraction(workflow, app_workflow, command_name, preserved_command)
        pe_output = extractor.extract()
        if not pe_output.parameters_are_valid:
            return CommandOutput(
                command_name = command_name,
                command_response=
                    CommandResponse(
                        response=(
                            f"PARAMETER EXTRACTION ERROR FOR COMMAND '{command_name}'\n"
                            f"{pe_output.error_msg}"
                        ),
                        success=False
                    )
            )

        workflow.end_command_processing()

        return CommandOutput(
            command_response=
                CommandResponse(
                    response="",
                    artifacts={
                        "command": preserved_command,
                        "command_name": command_name,
                        "cmd_parameters": pe_output.cmd_parameters,
                    },
                )
        ) 