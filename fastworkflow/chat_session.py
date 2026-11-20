from enum import Enum
from queue import Empty, Queue
from threading import Thread
from typing import Optional
import contextlib
from pathlib import Path

import litellm  # noqa: F401  (must precede dspy; see fastworkflow/utils/dspy_utils.py)
import dspy
import litellm

import fastworkflow
from fastworkflow import active_workflow
from fastworkflow import waiting as waiting_module
from fastworkflow.workflow_execution_context import CommandCancelledError, WorkflowExecutionContext
from fastworkflow.utils.logging import logger
from fastworkflow.utils.startup_progress import StartupProgress
from fastworkflow.turn_gate import GatedQueue, MessageQueue, TurnGate


class SessionStatus(Enum):
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"

class ChatWorker(Thread):
    def __init__(self, chat_session: "ChatSession"):
        super().__init__()
        self.chat_session = chat_session
        self.daemon = True
        
    def run(self):
        """Process messages for the root workflow"""
        try:
            self.chat_session._status = SessionStatus.RUNNING
            workflow = self.chat_session._current_workflow
            if workflow:
                logger.debug(f"Started root workflow {workflow.id}")
            
            # Run the workflow loop
            self.chat_session._run_workflow_loop()
            
        finally:
            self.chat_session._status = SessionStatus.STOPPED
            # Ensure workflow is popped if thread terminates unexpectedly
            if self.chat_session.get_active_workflow() is not None:
                self.chat_session.pop_active_workflow()

class ChatSession:
    def get_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        """Get the currently active workflow.

        Returns the top of the context-local stack when set (e.g. during a
        message turn or explicit push); otherwise falls back to the bound app
        workflow so callers on other threads see the session's workflow.
        """
        workflow = active_workflow.get_active_workflow()
        if workflow is not None:
            return workflow
        return self._core.app_workflow if self._core else None

    def push_active_workflow(self, workflow: fastworkflow.Workflow) -> None:
        """Push a workflow onto the context-local stack."""
        active_workflow.push_active_workflow(workflow)

    def pop_active_workflow(self) -> Optional[fastworkflow.Workflow]:
        """Pop a workflow from the context-local stack."""
        return active_workflow.pop_active_workflow()

    def clear_workflow_stack(self) -> None:
        """Clear the entire workflow stack for this context."""
        active_workflow.clear_workflow_stack()

    def stop_workflow(self) -> None:
        """
        Stop the current workflow and clear the workflow stack.
        This method is called when starting a new root workflow to ensure
        the previous workflow is properly stopped and resources are cleaned up.
        """
        # Set status to stopping to signal the workflow loop to exit
        self._status = SessionStatus.STOPPING
        
        # Wait for the chat worker thread to finish if it exists
        if self._chat_worker and self._chat_worker.is_alive():
            self._chat_worker.join(timeout=5.0)  # Wait up to 5 seconds
            if self._chat_worker.is_alive():
                logger.warning("Chat worker thread did not terminate within timeout")
        
        # Clear the workflow stack
        self.clear_workflow_stack()
        
        # Reset status to stopped
        self._status = SessionStatus.STOPPED
        
        # Clear current workflow reference
        self._current_workflow = None
        
        logger.debug("Workflow stopped and workflow stack cleared")

    def __init__(
        self,
        run_as_agent: bool = False,
        generate_insights: bool = False,
        detect_no_agent_mode: bool = False,
    ):
        """
        Initialize a chat session.

        Args:
            run_as_agent: If True, use agent mode (DSPy-based tool selection).
                         If False (default), use traditional command execution.
            generate_insights: If True, enable teacher/student insights distillation
                         on each agent turn (Topology A / CLI only).
            detect_no_agent_mode: If True, an unset agent model (``LLM_AGENT``) puts the session in no-agent mode
                         (``fastworkflow.no_agent_mode``). ``fastworkflow run`` sets it.

        A chat session can run multiple workflows that share the same message queues.
        Use start_workflow() to start a specific workflow within this session.
        ChatSession is Topology A: ask_user blocks on the queue until the human answers.
        """
        self._core = WorkflowExecutionContext(
            run_as_agent=run_as_agent,
            generate_insights=generate_insights,
            detect_no_agent_mode=detect_no_agent_mode,
        )
        # CLI identity [R17]: a synthetic per-session channel so CLI turns are
        # attributable in the observability store (conversation ids are minted
        # by the store from Phase 2).
        from datetime import datetime, timezone
        self._core.bind_observability_identity(
            channel_id=f"cli:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%f')}Z"
        )

        # Create queues for user messages and command outputs (CLI transport)
        # Gated queues: the terminal can stop waiting on a turn and have its output dropped (see turn_gate).
        self._gate = TurnGate()
        self._user_message_queue = MessageQueue(self._gate)
        self._command_output_queue = GatedQueue(self._gate)
        self._command_trace_queue = GatedQueue(self._gate)
        self._core.set_transport_queues(
            user_message_queue=self._user_message_queue,
            command_output_queue=self._command_output_queue,
            command_trace_queue=self._command_trace_queue,
        )

        self._status = SessionStatus.STOPPED
        self._chat_worker = None
        self._current_workflow = None
        self._keep_alive = False

        from fastworkflow.command_executor import CommandExecutor
        self._CommandExecutor = CommandExecutor

        # this intializes the conversation traces file name also
        # which is necessary when starting a brand new chat session
        self.clear_conversation_history()

    def start_workflow(self,
        workflow_folderpath: str, 
        workflow_id_str: Optional[str] = None, 
        parent_workflow_id: Optional[int] = None, 
        workflow_context: dict = None, 
        startup_command: str = "", 
        startup_action: Optional[fastworkflow.Action] = None, 
        keep_alive: bool = False,
        project_folderpath: Optional[str] = None
        ) -> Optional[fastworkflow.CommandOutput]:
        """
        Create and start a workflow within this chat session.
        
        Args:
            workflow_folderpath: The folder containing the fastworkflow Workflow
            workflow_id_str: Arbitrary key used to persist the workflow state
            parent_workflow_id: Persist this workflow under a parent workflow
            workflow_context: The starting context for the workflow.
            startup_command: Optional command to execute on startup
            startup_action: Optional action to execute on startup
            keep_alive: Whether to keep the chat session alive after workflow completion
            
        Returns:
            CommandOutput for non-keep_alive workflows, None otherwise
        """
        if startup_command and startup_action:
            raise ValueError("Cannot provide both startup_command and startup_action")

        # Phase 9 step 4: artifacts trained before the sparse "not here" construction are refused
        # at load, for the app and for the internal CME workflow whose heads it also loads.
        # Training never comes through here, so `fastworkflow train` can still retrain them.
        from fastworkflow.train.artifact_versioning import require_construction_record
        require_construction_record(workflow_folderpath)
        require_construction_record(
            fastworkflow.get_internal_workflow_path("command_metadata_extraction"))

        litellm.drop_params = True  # See https://docs.litellm.ai/docs/completion/drop_params

        # Create the workflow
        workflow = fastworkflow.Workflow.create(
            workflow_folderpath,
            workflow_id_str=workflow_id_str,
            parent_workflow_id=parent_workflow_id,
            workflow_context=workflow_context,
            project_folderpath=project_folderpath
        )
        
        self._current_workflow = workflow
        self._status = SessionStatus.STOPPED
        self._startup_command = startup_command

        if startup_action and startup_action.workflow_id is None:
            startup_action.workflow_id = workflow.id
        self._startup_action = startup_action
        self._keep_alive = False if parent_workflow_id else keep_alive

        # Check if we need to stop the current workflow
        # Stop if this is a new root workflow (no parent, keep_alive=True)
        current_workflow = self.get_active_workflow()
        if (current_workflow and 
            parent_workflow_id is None and 
            self._keep_alive):
            logger.info(f"Stopping current workflow {current_workflow.id} to start new root workflow {workflow.id}")
            self.stop_workflow()

        # ------------------------------------------------------------
        # Eager warm-up of CommandRouter / ModelPipeline
        # ------------------------------------------------------------
        # Loading transformer checkpoints and moving them to device is
        # expensive (~1 s).  We do it here *once* for every model artifact
        # directory so that the first user message does not pay the cost.
        # Import is deferred so `import fastworkflow` / ChatSession does not
        # pull transformers/torch/sklearn at package import time.
        try:
            from fastworkflow.model_pipeline_training import CommandRouter

            command_info_root = Path(workflow.folderpath) / "___command_info"
            if command_info_root.is_dir():
                subdirs = [d for d in command_info_root.iterdir() if d.is_dir()]

                # Tell the progress bar how many extra steps we are going to
                # perform (one per directory plus one for the wildcard "*").
                StartupProgress.add_total(len(subdirs) + 1)

                for subdir in subdirs:
                    # Instantiating CommandRouter triggers ModelPipeline
                    # construction and caches it process-wide.
                    with contextlib.suppress(Exception):
                        CommandRouter(str(subdir))
                    StartupProgress.advance(f"Warm-up {subdir.name}")

                # Also warm-up the global-context artefacts, which live in a
                # pseudo-folder named '*' in some workflows.
                with contextlib.suppress(Exception):
                    CommandRouter(str(command_info_root / '*'))
                StartupProgress.advance("Warm-up global")
        except Exception as warm_err:  # pragma: no cover – warm-up must never fail
            logger.debug(f"Model warm-up skipped due to error: {warm_err}")

        # Planning commands and command utterances: read on the first request otherwise (~2 s).
        # Keyed by the folder path string, so the same one the turn path passes.
        try:
            from fastworkflow.workflow_agent import warm_planning_commands
            warm_planning_commands(workflow.folderpath)
        except Exception as warm_err:  # pragma: no cover – warm-up must never fail
            logger.debug(f"Planning command warm-up skipped due to error: {warm_err}")

        self._core.bind_app_workflow(workflow)
        self._core.keep_alive = self._keep_alive

        # Start the workflow
        if self._status != SessionStatus.STOPPED:
            raise RuntimeError("Workflow already started")
        
        self._status = SessionStatus.STARTING
        
        # Agent + MCP tooling initialize lazily on first agent message (after contextvar push)
        
        command_output = None
        if self._keep_alive:
            # Root workflow gets a worker thread
            self._chat_worker = ChatWorker(self)
            self._chat_worker.start()
        else:
            # Child workflows run their loop in the current thread
            self._status = SessionStatus.RUNNING
            command_output = self._run_workflow_loop()

        return command_output

    @property
    def workflow_tool_agent(self):
        """Get the workflow tool agent for agent mode."""
        return self._core.workflow_tool_agent

    @property
    def cme_workflow(self) -> fastworkflow.Workflow:
        """Get the command metadata extraction workflow."""
        return self._core.cme_workflow

    @property
    def app_workflow(self) -> Optional[fastworkflow.Workflow]:
        """Get the bound application workflow."""
        return self._core.app_workflow
    
    @property
    def run_as_agent(self) -> bool:
        """Check if running in agent mode."""
        return self._core.run_as_agent

    @property
    def no_agent_mode(self) -> bool:
        """True when the agent model (``LLM_AGENT``) is not specified and this session detects the mode."""
        return self._core.no_agent_mode

    @property
    def _distillation_insights_count(self) -> int:
        """Number of insights extracted so far in insights-distillation mode."""
        return self._core._distillation_insights_count

    @property
    def user_message_queue(self) -> Queue:
        return self._user_message_queue

    @property
    def command_output_queue(self) -> Queue:
        return self._command_output_queue

    @property
    def command_trace_queue(self) -> Queue:
        return self._command_trace_queue

    @property
    def interrupted_turn_running(self) -> bool:
        """True while a turn the terminal stopped waiting on is still running (its output is being dropped)."""
        return self._gate.muted

    def abandon_turn(self) -> bool:
        """Stop waiting on the running turn (Ctrl-C or a deadline at the terminal).

        The engine cannot abort a turn in flight, so the turn runs to its end in the worker thread: its commands'
        side effects still happen. Guaranteed: nothing it outputs reaches the queues (what it already put is
        cleared, the rest is dropped), it can never block on ``ask_user`` (that read cancels the turn through the
        engine's ``CommandCancelledError``), and the next message runs only after it finishes, in order.
        False when the turn had already ended (its sentinel is queued): nothing is abandoned.
        """
        return self._gate.abandon(self._command_trace_queue, self._command_output_queue)

    def _deliver_if_unsent(self, output) -> None:
        """Give the terminal a turn's output when the turn put none on the queues.

        A reply to an approval or review, and the approval management messages (``/approvals``), are answered in
        ``_execute_message`` without the queue puts a command's run makes, so the terminal waited for a sentinel that
        never came. A turn that put its sentinel (every command path does) is left alone.
        """
        if self._gate.sentinels == 0 and output is not None:
            self._core._maybe_enqueue_output(output)
            self._core._maybe_enqueue_trace_sentinel()

    def _pending_cancel_reply(self) -> Optional[tuple[str, str]]:
        """What is pending on the session once no turn is running, as ``(label, the engine reply that cancels it)``.

        The one classification ``//cancel`` and the cleanup of an abandoned turn share: the CANCEL event of
        ``waiting.step`` on the core's one ``waiting`` record, whose kind table gives the reply that ends it as typing it
        does: ``no`` for an approval or a plan review, ``cancel`` for a plan's which-command or value question and a
        paused decision step, ``abort`` for the CME's open command. None when nothing is pending, and for an
        ``ask_user`` question: a blocked read is ended by ``MessageQueue.post_cancel`` (a ChatSession never holds a
        suspended one, it blocks on its queue).
        """
        move = waiting_module.step(self._core.waiting, waiting_module.Event(waiting_module.CANCEL))
        return (move.label, move.text) if move.action == waiting_module.ACT_REPLY else None

    def cancel_pending_prompt(self) -> Optional[str]:
        """Cancel what the session is waiting on, the way the chat page's Cancel does; the kind, or None.

        * ``"question"``: a blocked ``ask_user`` read ends through the engine's ``CommandCancelledError``
          (the turn's "Command cancelled" output, suspension cleared).
        * ``"review"``: an open approval or plan review is answered ``no``.
        * ``"clarification"``: any other open question (a which-command or value question of a plan or of the open
          command, a paused decision step) is answered with its kind's cancel reply (``fastworkflow.waiting.TABLE``).

        The reply goes through the user message queue, so the turn that results is read like any other.
        None when nothing is pending, or when an interrupted turn is still finishing (it has no prompt).
        """
        if self._gate.muted:
            return None
        if self._user_message_queue.post_cancel():
            return "question"
        if (pending := self._pending_cancel_reply()) is None:
            return None
        self._user_message_queue.put(pending[1])
        return pending[0]

    def _cancel_unseen_prompts(self) -> None:
        """An abandoned turn just ended. What it left pending was never shown to the user, so the next message must
        not be read as its answer (a ``yes`` would approve an unseen write): cancel it here, in the worker, before the
        next queued message is taken, with the replies ``//cancel`` sends. Its output is dropped with the turn's."""
        for _ in range(3):
            if (pending := self._pending_cancel_reply()) is None:
                return
            self._gate.cancelled_unseen = True
            self._gate.begin_turn()
            try:
                output = self._core._execute_message(pending[1])
                self._core.finalize_turn_for_observability(output)
            except Exception:  # noqa: BLE001 - cleanup must never kill the worker
                logger.warning("could not cancel the prompt an interrupted turn left pending", exc_info=True)
                return

    def take_cancelled_unseen_notice(self) -> bool:
        """True once after a question of an interrupted turn was cancelled unseen (for the one-line notice)."""
        with self._gate.lock:
            seen, self._gate.cancelled_unseen = self._gate.cancelled_unseen, False
        return seen

    @property
    def workflow_is_complete(self) -> bool:
        workflow = self._core.app_workflow
        return workflow.is_complete if workflow else True
    
    @workflow_is_complete.setter
    def workflow_is_complete(self, value: bool) -> None:
        if workflow := self._core.app_workflow:
            workflow.is_complete = value
    
    def append_action_log(self, record: dict) -> None:
        self._core.append_action_log(record)

    def clear_action_log(self) -> None:
        self._core.clear_action_log()

    def append_turn_output(self, command_output: fastworkflow.CommandOutput) -> None:
        """Delegate turn-accumulator capture to the core (duck-typed by workflow_agent)."""
        self._core.append_turn_output(command_output)

    def append_ask_user_entry(self, question: str) -> fastworkflow.CommandOutput:
        """Delegate ask_user entry creation to the core (duck-typed by workflow_agent)."""
        return self._core.append_ask_user_entry(question)

    def run_planning_command(self, qualified_name: str, params, step_text: str) -> fastworkflow.CommandOutput:
        """Delegate a planning command's run to the core (command_executor dispatches to it)."""
        return self._core.run_planning_command(qualified_name, params, step_text)

    def complete_ask_user_entry(self, answer: str) -> None:
        """Delegate ask_user answer fill to the core (duck-typed by workflow_agent)."""
        self._core.complete_ask_user_entry(answer)

    @property
    def conversation_history(self) -> dspy.History:
        """Return the conversation history."""
        return self._core.conversation_history

    @property
    def _conversation_history(self) -> dspy.History:
        return self._core.conversation_history

    @_conversation_history.setter
    def _conversation_history(self, value: dspy.History) -> None:
        self._core._conversation_history = value

    # def clear_conversation_history(self, trace_filename_suffix: Optional[str] = None) -> None:
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        This resets the conversation history to an empty state.
        """
        self._core.clear_conversation_history()
        # Filename for conversation traces
        # if trace_filename_suffix:
        #     self._conversation_traces_file_name: str = (
        #         f"conversation_traces_{trace_filename_suffix}"
        #     )
        # else:
        #     self._conversation_traces_file_name: str = (
        #         f"conversation_traces_{datetime.now().strftime('%m_%d_%Y:%H_%M_%S')}.jsonl"
        #     )

    def _run_workflow_loop(self) -> Optional[fastworkflow.CommandOutput]:
        """
        Run the workflow message processing loop.
        For child workflows (keep_alive=False):
        - Returns final CommandOutput when workflow completes
        - All outputs (success or failure) are sent to queue during processing
        """
        last_output = None
        workflow = self._current_workflow

        try:
            # Handle startup command/action
            if self._startup_command:
                last_output = self._core._execute_message(self._startup_command)
                self._core.finalize_turn_for_observability(last_output)
            elif self._startup_action:
                last_output = self._core.process_action(self._startup_action)
                self._core.finalize_turn_for_observability(last_output)

            while (
                not self.workflow_is_complete or self._keep_alive
            ) and self._status != SessionStatus.STOPPING:
                try:
                    message = self.user_message_queue.idle_get()

                    self._gate.begin_turn()
                    try:
                        if isinstance(message, fastworkflow.Action):
                            try:
                                last_output = self._core.process_action(message)
                            except CommandCancelledError as exc:
                                # An action has no message path to convert this (``_execute_message`` does): a
                                # cancelled question must end the action, not the worker thread.
                                self._core._reset_agent_suspension()
                                last_output = self._core._command_cancelled_output(str(exc))
                        else:
                            last_output = self._core._execute_message(message)
                        # Emit the turn record/root-span close for observability;
                        # the CLI transport itself still rides the queues.
                        self._deliver_if_unsent(last_output)
                        self._core.finalize_turn_for_observability(last_output)
                    finally:
                        try:
                            if self._gate.muted:
                                self._cancel_unseen_prompts()
                        finally:
                            # An abandoned turn's output was dropped while it ran; with it over, drop no more.
                            self._gate.turn_finished()

                except Empty:
                    continue

            # Return final output for child workflows, regardless of success/failure
            if not self._keep_alive:
                return last_output

        finally:
            self._status = SessionStatus.STOPPED
            if self.get_active_workflow() is not None:
                self.clear_workflow_stack()
            logger.debug(f"Workflow {workflow.id if workflow else 'unknown'} completed")

        return None
    
    # def _is_mcp_tool_call(self, message: str) -> bool:
    #     """Detect if message is an MCP tool call JSON"""
    #     try:
    #         data = json.loads(message)
    #         return data.get("type") == "mcp_tool_call"
    #     except (json.JSONDecodeError, AttributeError):
    #         return False
    
    # def _process_mcp_tool_call(self, message: str) -> fastworkflow.CommandOutput:
    #     # sourcery skip: class-extract-method, extract-method
    #     """Process an MCP tool call message"""
    #     workflow = self.get_active_workflow()
        
    #     try:
    #         # Parse JSON message
    #         data = json.loads(message)
    #         tool_call_data = data["tool_call"]
            
    #         # Create MCPToolCall object
    #         tool_call = fastworkflow.MCPToolCall(
    #             name=tool_call_data["name"],
    #             arguments=tool_call_data["arguments"]
    #         )
            
    #         # Execute via command executor
    #         mcp_result = self._CommandExecutor.perform_mcp_tool_call(
    #             workflow, 
    #             tool_call, 
    #             command_context=workflow._current_command_context_name
    #         )
            
    #         # Convert MCPToolResult back to CommandOutput for consistency
    #         command_output = self._convert_mcp_result_to_command_output(mcp_result)
            
    #         # Put in output queue if needed
    #         if (not command_output.success or self._keep_alive) and self.command_output_queue:
    #             self.command_output_queue.put(command_output)

    #         # Flush on successful or failed tool call – state may have changed.
    #         if workflow := self.get_active_workflow():
    #             workflow.flush()
            
    #         return command_output
            
    #     except Exception as e:
    #         logger.error(f"Error processing MCP tool call: {e}. Tool call content: {message}")
    #         return self._process_message(message)  # process as a message
    
    # def _convert_mcp_result_to_command_output(self, mcp_result: fastworkflow.MCPToolResult) -> fastworkflow.CommandOutput:
    #     """Convert MCPToolResult to CommandOutput for compatibility"""
    #     command_response = fastworkflow.CommandResponse(
    #         response=mcp_result.content[0].text if mcp_result.content else "No response",
    #         success=not mcp_result.isError
    #     )
        
    #     command_output = fastworkflow.CommandOutput(command_response=command_response)
    #     command_output._mcp_source = mcp_result  # Mark for special formatting
    #     return command_output
    
