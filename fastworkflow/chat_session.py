from enum import Enum
from queue import Empty, Queue
from threading import Thread
from typing import Optional
import contextlib
from pathlib import Path
import os
import time

import dspy
import litellm

import fastworkflow
from fastworkflow import active_workflow
from fastworkflow.typed_failure import (
    CODE_WORKER_FAILED,
    TypedFailure,
    classify_exception,
)
from fastworkflow.worker_health import (
    TurnRequest,
    WorkerDeadError,
    WorkerHealth,
    WorkerState,
    unwrap_request,
)
from fastworkflow.workflow_execution_context import WorkflowExecutionContext
from fastworkflow.utils.logging import logger
from fastworkflow.utils.startup_progress import StartupProgress


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
        """Process messages for the root workflow, under supervision.

        Arch §13.3, FW-REQ-008 clause 3. This used to be a bare try/finally, so
        an exception escaping the message loop terminated the only worker
        silently: the thread died, `_status` went to STOPPED, and every caller
        blocked on `command_output_queue.get()` waited for a reply nobody was
        left to send. The loop below catches the outer-loop terminal failure,
        classifies it, records it on health, and *delivers* it — a failure
        CommandOutput and its trace sentinel — so a waiting caller is released
        with an answer instead of a timeout.
        """
        session = self.chat_session
        session._health.beat(WorkerState.RUNNING)
        try:
            session._status = SessionStatus.RUNNING
            workflow = session._current_workflow
            if workflow:
                logger.debug(f"Started root workflow {workflow.id}")

            session._run_workflow_loop()
        except BaseException as exc:  # noqa: BLE001 - terminal supervision point
            failure = classify_exception(exc)
            failure = TypedFailure(
                disposition=failure.disposition,
                code=CODE_WORKER_FAILED,
                detail=f"worker loop terminated: {failure.detail}",
            )
            session._health.record_worker_failure(
                failure, turn_key=session._core.current_turn_key
            )
            logger.critical(
                "Chat worker terminated: %s", failure.as_observation(), exc_info=True
            )
            session._deliver_worker_failure(failure)
            # Not re-raised: the thread is ending either way, and a traceback on
            # a daemon thread is not a delivery mechanism. The classification is
            # on health and in the output queue, which is where a caller looks.
        finally:
            session._status = SessionStatus.STOPPED
            # `set_state` refuses to un-poison, so a worker that FAILED above is
            # not relabelled STOPPED here (§13.3: the poisoning stays visible).
            session._health.set_state(WorkerState.STOPPED)
            session._fail_queued_requests(session._health.rejection_failure())
            # Ensure workflow is popped if thread terminates unexpectedly
            if session.get_active_workflow() is not None:
                session.pop_active_workflow()

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
        self._health.set_state(WorkerState.STOPPING)

        # Wait for the chat worker thread to finish if it exists
        if self._chat_worker and self._chat_worker.is_alive():
            self._chat_worker.join(timeout=5.0)  # Wait up to 5 seconds
            if self._chat_worker.is_alive():
                # Arch §13.3: a timed-out join() does not clear ownership and
                # does not claim termination. The thread may still be inside a
                # call; clearing the stack and reporting STOPPED — which is what
                # this used to do unconditionally — hands the next workflow a
                # session whose predecessor is still running in it.
                detail = "chat worker did not terminate within 5s of stop_workflow"
                logger.error("%s; ownership retained and worker marked stuck", detail)
                self._health.mark_stuck(detail)
                return

        # Clear the workflow stack
        self.clear_workflow_stack()

        # Reset status to stopped
        self._status = SessionStatus.STOPPED
        self._health.set_state(WorkerState.STOPPED)

        # Clear current workflow reference
        self._current_workflow = None

        logger.debug("Workflow stopped and workflow stack cleared")

    def __init__(
        self,
        run_as_agent: bool = False,
        generate_insights: bool = False,
    ):
        """
        Initialize a chat session.

        Args:
            run_as_agent: If True, use agent mode (DSPy-based tool selection).
                         If False (default), use traditional command execution.
            generate_insights: If True, enable teacher/student insights distillation
                         on each agent turn (Topology A / CLI only).

        A chat session can run multiple workflows that share the same message queues.
        Use start_workflow() to start a specific workflow within this session.
        ChatSession is Topology A: ask_user blocks on the queue until the human answers.
        """
        self._core = WorkflowExecutionContext(
            run_as_agent=run_as_agent,
            generate_insights=generate_insights,
        )
        # CLI identity [R17]: a synthetic per-session channel so CLI turns are
        # attributable in the observability store (conversation ids are minted
        # by the store from Phase 2).
        from datetime import datetime, timezone
        self._core.bind_observability_identity(
            channel_id=f"cli:{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%f')}Z"
        )

        # Create queues for user messages and command outputs (CLI transport)
        self._user_message_queue = Queue()
        self._command_output_queue = Queue()
        self._command_trace_queue = Queue()
        self._core.set_transport_queues(
            user_message_queue=self._user_message_queue,
            command_output_queue=self._command_output_queue,
            command_trace_queue=self._command_trace_queue,
        )

        self._status = SessionStatus.STOPPED
        self._chat_worker = None
        self._current_workflow = None
        self._keep_alive = False
        # Worker health as a value (FW-REQ-008 clause 4): observable without
        # reading a thread stack, and the thing `submit`/`receive_turn` consult
        # so a caller fails promptly instead of blocking on a dead worker.
        self._health = WorkerHealth()

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

        self._core.bind_app_workflow(workflow)
        self._core.keep_alive = self._keep_alive

        # Start the workflow
        if self._status != SessionStatus.STOPPED:
            raise RuntimeError("Workflow already started")
        
        self._status = SessionStatus.STARTING
        
        # Agent + MCP tooling initialize lazily on first agent message (after contextvar push)
        
        command_output = None
        if self._keep_alive:
            # Root workflow gets a worker thread. Refuse to start one over a
            # poisoned predecessor: its ownership was retained deliberately.
            if self._health.is_poisoned:
                raise WorkerDeadError(self._health)
            self._health = WorkerHealth(state=WorkerState.STARTING)
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
    def intent_clarification_agent(self):
        """Get the intent clarification agent for agent mode."""
        return self._core.intent_clarification_agent

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

    # ------------------------------------------------------------------
    # Worker supervision (arch §13.3, FW-REQ-008 clauses 3-5)
    # ------------------------------------------------------------------

    @property
    def health(self) -> WorkerHealth:
        """Observable worker health — state, heartbeat, last classified failure."""
        return self._health

    def _failure_output(
        self, failure: TypedFailure
    ) -> fastworkflow.CommandOutput:
        """A CommandOutput that says a turn failed, and says how.

        `success` is left to the response's own failure flag rather than being
        described in prose: a caller reading the transport must be able to tell
        a failed turn from a successful one without parsing text.
        """
        active = self.get_active_workflow()
        response = fastworkflow.CommandResponse(
            response=failure.as_observation(), success=False
        )
        response.artifacts["failure"] = failure.to_state()
        output = fastworkflow.CommandOutput(command_response=response)
        if active is not None:
            output.workflow_name = active.folderpath.split("/")[-1]
        return output

    def _publish_failure(self, output: fastworkflow.CommandOutput) -> None:
        """Put a failure on the transport, output before sentinel.

        The ordering is the existing transport contract (see `_ask_user_tool`):
        the payload must be visible before the sentinel releases the reader.
        """
        if self._command_output_queue is not None:
            self._command_output_queue.put(output)
        if self._command_trace_queue is not None:
            self._command_trace_queue.put(None)

    def _deliver_worker_failure(self, failure: TypedFailure) -> None:
        """Release whoever is waiting, with the classification."""
        with contextlib.suppress(Exception):
            self._publish_failure(self._failure_output(failure))

    def _fail_queued_requests(self, failure: TypedFailure) -> int:
        """Fail every envelope still queued for a worker that is gone.

        Raw (non-envelope) submissions are drained too — there is nowhere to
        deliver their failure, which is exactly the gap envelopes close — and
        the count is logged so the loss is visible rather than silent.
        """
        failed = raw = 0
        while True:
            try:
                item = self._user_message_queue.get_nowait()
            except Empty:
                break
            _payload, request = unwrap_request(item)
            if request is not None:
                request.fail(failure)
                failed += 1
            else:
                raw += 1
        if raw:
            logger.warning(
                "Dropped %d queued message(s) submitted without an envelope; "
                "they have no failure delivery path", raw
            )
        return failed

    def submit(self, message, *, envelope: bool = True):
        """Submit a message to the worker, failing fast when it cannot run it.

        FW-REQ-008 clause 5. The raw `user_message_queue.put()` path still works
        and is what `envelope=False` reproduces; it just cannot tell the caller
        that nobody will ever read it.
        """
        if self._keep_alive and not self._health.is_alive:
            raise WorkerDeadError(self._health)
        if not envelope:
            self._user_message_queue.put(message)
            return None
        request = TurnRequest(payload=message, request_id=str(time.time_ns()))
        self._user_message_queue.put(request)
        return request

    def receive_turn(self, timeout: Optional[float] = None):
        """Health-aware replacement for `command_output_queue.get()`.

        Arch §13.3: supported callers migrate off raw queue polling. The
        difference that matters is at the bottom of this loop — when the queue
        is empty *and* the worker is gone, this raises instead of waiting out a
        timeout that cannot end in an answer.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            try:
                return self._command_output_queue.get(timeout=0.1)
            except Empty:
                pass
            if not self._health.is_alive:
                raise WorkerDeadError(self._health)
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("no turn output within the requested window")

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
                request = None
                try:
                    self._health.beat(WorkerState.RUNNING)
                    message, request = unwrap_request(self.user_message_queue.get())

                    if isinstance(message, fastworkflow.Action):
                        last_output = self._core.process_action(message)
                    else:
                        last_output = self._core._execute_message(message)
                    # Emit the turn record/root-span close for observability;
                    # the CLI transport itself still rides the queues.
                    self._core.finalize_turn_for_observability(last_output)
                    if request is not None:
                        request.complete(last_output)
                    self._health.beat()

                except Empty:
                    continue
                except Exception as exc:
                    # FW-REQ-008 clause 3: an unhandled turn exception is a
                    # terminal FAILED TURN, not a dead worker. The turn's
                    # failure is classified, delivered on the transport, and
                    # the loop goes back for the next message — which is what
                    # makes the acceptance criterion "a following independent
                    # turn can run" true rather than aspirational.
                    #
                    # `except Exception`, deliberately not BaseException: the
                    # control signals (CommandCancelledError, AskUserSuspend)
                    # and thread-level signals subclass BaseException and must
                    # keep their existing handling, which is above this frame.
                    failure = classify_exception(exc)
                    self._health.record_turn_failure(
                        failure, turn_key=self._core.current_turn_key
                    )
                    logger.error(
                        "Turn failed: %s", failure.as_observation(), exc_info=True
                    )
                    last_output = self._failure_output(failure)
                    self._publish_failure(last_output)
                    if request is not None:
                        request.fail(failure)
                    self._health.beat()
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
    #             command_context=workflow.current_command_context_name
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
    
    def _process_message(self, message: str) -> fastworkflow.CommandOutput:
        """Back-compat shim: delegate single-message execution to the core."""
        return self._core._execute_message(message)

    def _process_agent_message(self, message: str) -> fastworkflow.CommandOutput:
        """Back-compat shim: delegate agent-message execution to the core."""
        return self._core._execute_message(message)

    def _process_action(self, action: fastworkflow.Action) -> fastworkflow.CommandOutput:
        """Back-compat shim: delegate action execution to the core."""
        return self._core.process_action(action)

    def profile_invoke_command(self, message: str):
        """
        Profile the invoke_command method with detailed focus on performance issues.
        
        Args:
            message: The message to process
            output_file: Name of the profile output file
            
        Returns:
            The result of the invoke_command call
        """
        from datetime import datetime
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"invoke_command_{timestamp}.prof"        

        import cProfile
        import pstats
        import io
        import time
        
        # Create a Profile object
        profiler = cProfile.Profile()
        
        # Enable profiling
        profiler.enable()
        
        # Execute invoke_command and time it
        start_time = time.time()
        if self._core.app_workflow:
            self.push_active_workflow(self._core.app_workflow)
        try:
            result = self._CommandExecutor.invoke_command(self, message)
        finally:
            if self._core.app_workflow:
                self.pop_active_workflow()
        elapsed = time.time() - start_time
        
        # Disable profiling
        profiler.disable()
        
        # Save profile results to file
        profiler.dump_stats(output_file)
        print(f"\nProfile data saved to {os.path.abspath(output_file)}")
        print(f"invoke_command execution took {elapsed:.4f} seconds")
        
        # Create summary report
        report_file = f"{os.path.splitext(output_file)[0]}_report.txt"
        with open(report_file, "w") as f:
            # Overall summary by cumulative time
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('cumulative').print_stats(30)
            f.write(f"=== CUMULATIVE TIME SUMMARY (TOP 30) === Execution time: {elapsed:.4f}s\n")
            f.write(s.getvalue())
            f.write("\n\n")
            
            # Internal time summary
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('time').print_stats(30)
            f.write("=== INTERNAL TIME SUMMARY (TOP 30) ===\n")
            f.write(s.getvalue())
            f.write("\n\n")
            
            # Most called functions
            s = io.StringIO()
            ps = pstats.Stats(profiler, stream=s)
            ps.sort_stats('calls').print_stats(30)
            f.write("=== MOST CALLED FUNCTIONS (TOP 30) ===\n")
            f.write(s.getvalue())
            
            # Focus areas for issues 3-7
            focus_areas = [
                ('lock_contention', ['lock', 'acquire', 'release'], 'time'),
                ('model_operations', ['torch', 'nn', 'model'], 'cumulative'),
                ('command_extraction', ['wildcard.py', 'extract', 'predict'], 'cumulative'),
                ('file_io', ['_get_sessiondb_folderpath', '_load', '_save'], 'cumulative'),
                ('frequent_operations', ['startswith', 'isinstance', 'get'], 'calls')
            ]
            
            for name, patterns, sort_by in focus_areas:
                f.write(f"\n\n=== {name.upper()} ===\n")
                for pattern in patterns:
                    s = io.StringIO()
                    ps = pstats.Stats(profiler, stream=s)
                    ps.sort_stats(sort_by).print_stats(pattern, 10)
                    f.write(f"\nPattern: '{pattern}'\n")
                    f.write(s.getvalue())
        
        print(f"Detailed report saved to {os.path.abspath(report_file)}")
        
        return result
