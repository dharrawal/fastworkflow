import argparse
import json
import os
import queue
from typing import Optional
import contextlib
import time

from fastworkflow.utils.env import read_env_files

# Instantiate a global console for consistent styling
console = None

def run_main(args):
    # Third-party CLI prettification libraries
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.console import Group
    from prompt_toolkit import PromptSession
    from prompt_toolkit.patch_stdout import patch_stdout
    from prompt_toolkit.formatted_text import HTML

    import fastworkflow
    from fastworkflow.utils.logging import logger
    from fastworkflow.core.turn_progress import ProgressEvent
    from fastworkflow.run_cli.turn_wait import WaitOutcome, interrupted_message, timed_out_message, wait_for_turn

    # Progress bar helper
    from fastworkflow.utils.startup_progress import StartupProgress

    # Instantiate a global console for consistent styling
    global console
    console = Console()
    prompt_session = PromptSession(HTML('<b>User ></b> '))

    def _build_artifact_table(artifacts: dict[str, str]) -> Table:
        """Return a rich.Table representation for artifact key-value pairs."""
        table = Table(show_header=True, header_style="bold cyan", box=None)
        table.add_column("Name", style="cyan", overflow="fold")
        table.add_column("Value", style="white", overflow="fold")
        for name, value in artifacts.items():
            table.add_row(str(name), str(value))
        return table

    def print_command_output(command_output):
        """Pretty-print workflow output using rich panels and tables."""
        # Use workflow_name in panel title
        panel_title = f"[bold yellow]Workflow - {command_output.workflow_name}[/bold yellow]"
        
        # Collect all body renderables
        all_renderables = []
        
        # Add command info section first (only once)
        command_info_table = Table(show_header=False, box=None)
        if command_output.context:
            command_info_table.add_row("Context:", Text(command_output.context, style="yellow"))
        if command_output.command_name:
            command_info_table.add_row("Command:", Text(command_output.command_name, style="yellow"))
        if command_output.command_parameters:
            command_info_table.add_row("Parameters:", Text(str(command_output.command_parameters.model_dump()), style="yellow"))
        
        # Add command info section if we have any rows
        if command_info_table.row_count > 0:
            all_renderables.extend(
                (Text("Command Information", style="bold yellow"), command_info_table)
            )
            
            # Add a horizontal separator after command information
            from rich.rule import Rule
            all_renderables.append(Rule(style="dim"))
        
        # Process the command response
        command_response = command_output.command_response
        if command_response.response:
            all_renderables.append(Text(command_response.response, style="green"))

        if command_response.artifacts:
            all_renderables.extend(
                (
                    Text("Artifacts", style="bold cyan"),
                    _build_artifact_table(command_response.artifacts),
                )
            )
        if command_response.next_actions:
            actions_table = Table(show_header=False, box=None)
            for act in command_response.next_actions:
                actions_table.add_row(Text(str(act), style="blue"))
            all_renderables.extend(
                (Text("Next Actions", style="bold blue"), actions_table)
            )
        if command_response.recommendations:
            rec_table = Table(show_header=False, box=None)
            for rec in command_response.recommendations:
                rec_table.add_row(Text(str(rec), style="magenta"))
            all_renderables.extend(
                (Text("Recommendations", style="bold magenta"), rec_table)
            )

        # Group all renderables together
        group = Group(*all_renderables)
        # Use the group in the panel
        panel = Panel.fit(group, title=panel_title, border_style="green")
        console.print(panel)


    """Main function to run the workflow."""
    if not os.path.isdir(args.workflow_path):
        console.print(f"[bold red]Error:[/bold red] The specified workflow path '{args.workflow_path}' is not a valid directory.")
        exit(1)

    commands_dir = os.path.join(args.workflow_path, "_commands")
    if not os.path.isdir(commands_dir):
        logger.info(f"No _commands directory found at {args.workflow_path}, existing...")
        return

    env_vars = read_env_files(args.env_file_path, args.passwords_file_path)
    if not os.path.isfile(args.env_file_path):
        raise ValueError(f"Env file not found: {args.env_file_path}")
    if not env_vars:
        raise ValueError(f"Env file {args.env_file_path} loaded no variables; path may be incorrect")

    if args.startup_command and args.startup_action:
        raise ValueError("Cannot provide both startup_command and startup_action")

    console.print(Panel(f"Running fastWorkflow: [bold]{args.workflow_path}[/bold]", title="[bold green]fastworkflow[/bold green]", border_style="green"))
    console.print(
        "[bold green]Tips:[/bold green] Type '//exit' to quit the application. Type '//new' to start a new conversation. " 
        "[bold green]Tips:[/bold green] Prefix natural language commands with a single '/' to execute them in deterministic (non-agentic) mode. "
        "Type '//cancel' to cancel a pending question, review or approval. Ctrl-C stops waiting for a running turn.")

    # Command files that do not load are skipped; say so before the first prompt (not only in the log). A workflow-wide
    # problem (a shared file such as startup_action.json or _workflow.py) stops the run, naming the file.
    from fastworkflow.planning.command_preview import load_problems

    problems = load_problems(args.workflow_path)
    for problem in problems:
        console.print(f"[bold red]Load problem:[/bold red] {problem}")
    if any(problem.workflow_wide for problem in problems):
        console.print("[bold red]Error:[/bold red] the workflow does not load: fix the shared file named above.")
        exit(1)

    # ------------------------------------------------------------------
    # Startup progress bar ------------------------------------------------
    # ------------------------------------------------------------------

    # 3 coarse CLI steps + per-directory warm-up (handled inside ChatSession) + 1 global warm-up
    StartupProgress.begin(total=3)

    # Heavy import – counted as first step once completed
    StartupProgress.advance("Imported fastworkflow modules")

    fastworkflow.init(env_vars=env_vars)
    StartupProgress.advance("fastworkflow.init complete")

    # Fail fast if the workflow has not been trained. Otherwise intent detection
    # crashes on the first command when CommandRouter cannot find the model
    # artifacts (e.g. ___command_info/global/threshold.json).
    from fastworkflow.intent.router import is_workflow_trained
    trained, missing_contexts = is_workflow_trained(args.workflow_path)
    # A context with no trained local model may run on the model matcher (JEV_API_KEY):
    # intent_detector.startup_report is the one decision, and it says in one line what the chain is.
    from fastworkflow.intent import intent_detector
    engine_problems, engine_notices = intent_detector.startup_report(
        args.workflow_path, missing_contexts, build=True)
    for notice in engine_notices:
        console.print(f"[bold yellow]Note:[/bold yellow] {notice}")
    if not trained and engine_problems:
        StartupProgress.end()
        workflow_name = os.path.basename(os.path.normpath(args.workflow_path))
        console.print(
            f"[bold red]Error:[/bold red] Workflow '{workflow_name}' is not trained "
            f"(missing model artifacts for context(s): {', '.join(missing_contexts)})."
        )
        console.print(
            "Train it first with:\n"
            f"  [bold]fastworkflow train {args.workflow_path} "
            f"{args.env_file_path} {args.passwords_file_path}[/bold]"
        )
        exit(1)

    # Phase 9 step 4: artifacts built before the sparse "not here" construction are refused,
    # for the app and for the internal CME workflow.
    from fastworkflow.train.artifact_versioning import construction_record_problem
    cme_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
    problem = (construction_record_problem(args.workflow_path)
               or construction_record_problem(cme_path))
    if problem:
        StartupProgress.end()
        console.print(f"[bold red]Error:[/bold red] {problem}")
        exit(1)

    # --startup_action / --startup_command win; with neither, the workflow folder's startup_action.json runs.
    startup_action = fastworkflow.resolve_startup_action(
        args.workflow_path, args.startup_command, args.startup_action)
    if startup_action is None and not args.startup_command:
        # A command with no Signature runs only as the startup action: with none, nothing would run it.
        from fastworkflow.planning.command_preview import missing_startup_action_problem

        if problem := missing_startup_action_problem(args.workflow_path):
            StartupProgress.end()
            console.print(f"[bold red]Startup problem:[/bold red] {problem}")
            exit(1)

    context_dict = None
    if args.context_file_path:
        with open(args.context_file_path, 'r') as file:
            context_dict = json.load(file)

    # No agent model (LLM_AGENT unset): deterministic mode, said plainly (fastworkflow.core.no_agent_mode).
    from fastworkflow.core import no_agent_mode

    run_as_agent, fallback_banner = no_agent_mode.run_mode(args.assistant)
    # (`load_problems` above ran before `init` loaded the env files, so the startup problem is read here.)
    if startup_problems := no_agent_mode.runtime_problems():
        # Neither the agent model nor Jev is set: stop, with no fallback to assistant mode.
        StartupProgress.end()
        for problem in startup_problems:
            console.print(f"[bold red]Startup problem:[/bold red] {problem}")
        exit(1)
    fastworkflow.chat_session = fastworkflow.ChatSession(
        run_as_agent=run_as_agent,
        generate_insights=getattr(args, "generate_insights", False),
        detect_no_agent_mode=True,
    )

    # Observability [R4]: recording is always on, so the SQLite sink is opened
    # here for the CLI's turns. The CLI channel is the
    # synthetic cli:<start> id ChatSession bound; each CLI session starts a
    # fresh conversation, and //new mints another [R17].
    from fastworkflow.observability.store import get_observability_sink
    obs_sink = get_observability_sink(args.workflow_path)

    def _bind_fresh_cli_conversation() -> None:
        """Mint a new CLI conversation id and bind it — never fails the CLI.

        A wedged/corrupt observability DB degrades to conversation-less turn
        records. The chatbot groups its rail by conversation, so those rows are
        reachable only through the channel's conversation-less group [R17].
        """
        core = fastworkflow.chat_session._core
        try:
            core.bind_observability_identity(
                conversation_id=obs_sink.store.mint_conversation_id(
                    core.observability_channel_id
                )
            )
        except Exception as exc:
            logger.warning(
                f"Could not mint a CLI conversation id ({type(exc).__name__}: "
                f"{exc}); this session's turns are recorded without a conversation"
            )

    if obs_sink is not None:
        fastworkflow.chat_session._core.set_trace_sink(obs_sink)
        _bind_fresh_cli_conversation()

    # Start the workflow within the chat session
    fastworkflow.chat_session.start_workflow(
        args.workflow_path, 
        workflow_context=context_dict,
        startup_command=args.startup_command, 
        startup_action=startup_action, 
        keep_alive=args.keep_alive,
        project_folderpath=args.project_folderpath
    )

    StartupProgress.advance("ChatSession ready")
    StartupProgress.end()
    if fallback_banner:
        console.print(f"[bold yellow]{fallback_banner}[/bold yellow]")
   
    with contextlib.suppress(queue.Empty):
        timeout: float = 0.000001
        if args.startup_command or startup_action is not None:
            timeout=180.0
        if command_output := fastworkflow.chat_session.command_output_queue.get(
            timeout=timeout
        ):
            print_command_output(command_output)

    # Drain any trace events (and sentinel) produced during startup
    while True:
        try:
            evt = fastworkflow.chat_session.command_trace_queue.get_nowait()
            if evt is None:
                break
        except queue.Empty:
            break

    while not fastworkflow.chat_session.workflow_is_complete or args.keep_alive:
        with patch_stdout():
            user_command = prompt_session.prompt()
        if user_command.startswith("//exit"):
            if getattr(args, "generate_insights", False):
                count = getattr(fastworkflow.chat_session, "_distillation_insights_count", 0)
                console.print(
                    f"\n[bold cyan]Insights generation complete. New insights extracted: {count}[/bold cyan]"
                )
            break
        if user_command.startswith("//new"):
            fastworkflow.chat_session.clear_conversation_history()
            if obs_sink is not None:
                # //new starts a new logical conversation [R17]
                _bind_fresh_cli_conversation()
            console.print("[bold]Agent >[/bold] New conversation started!\n", end="")
            user_command = prompt_session.prompt()

        if user_command.strip() == "//cancel":
            # The same engine replies the chat page's Cancel sends (see ChatSession.cancel_pending_prompt).
            if fastworkflow.chat_session.cancel_pending_prompt() is None:
                console.print(
                    "[bold]Nothing to cancel.[/bold]"
                    if not fastworkflow.chat_session.interrupted_turn_running
                    else "[bold]The interrupted turn is still finishing; there is nothing pending to cancel.[/bold]"
                )
                continue
        else:
            if fastworkflow.chat_session.interrupted_turn_running:
                console.print("[bold yellow]The interrupted turn is still finishing; your message runs after it.[/bold yellow]")
            fastworkflow.chat_session.user_message_queue.put(user_command)

        def _cancelled_unseen_note() -> None:
            if fastworkflow.chat_session.take_cancelled_unseen_notice():
                console.print("[bold yellow]A question from the interrupted turn was cancelled; "
                              "your message ran against a clean state.[/bold yellow]")

        def _show_event(evt) -> None:
            if isinstance(evt, ProgressEvent):
                status.update(f"[bold cyan]{evt.label}...[/bold cyan]")
                return

            # Choose styles based on success
            info_style = "dim orange3" if (evt.success is False) else "dim yellow"
            resp_style = "dim orange3" if (evt.success is False) else "dim green"

            if evt.direction == fastworkflow.CommandTraceEventDirection.AGENT_TO_WORKFLOW:
                console.print(f'[bold]Agent >[/bold] {evt.raw_command}', style=info_style)
            else:
                header = f"{evt.command_name or ''}"
                if evt.parameters:
                    header += f", {evt.parameters}"
                if header.strip():
                    console.print(f'[bold]Workflow >[/bold] {header}', style=info_style)
                console.print(f'[bold]Workflow >[/bold] {evt.response_text}', style=resp_style)

        def _show_tick(counter: int) -> None:
            if counter % 10 == 0:
                status.update(f"[bold cyan]Processing command... ({counter//10}s)[/bold cyan]")

        # Show spinner while draining trace events until sentinel signals completion.
        # Ctrl-C or --timeout_seconds stops the wait (the turn is abandoned; see turn_wait).
        with console.status("[bold cyan]Processing command...[/bold cyan]", spinner="dots") as status:
            outcome = wait_for_turn(
                fastworkflow.chat_session, _show_event,
                timeout_seconds=getattr(args, "timeout_seconds", None), on_tick=_show_tick,
            )

        _cancelled_unseen_note()
        if outcome is not WaitOutcome.DONE:
            console.print(
                "[bold yellow]"
                + (interrupted_message() if outcome is WaitOutcome.INTERRUPTED
                   else timed_out_message(args.timeout_seconds))
                + "[/bold yellow]"
            )
            continue

        # Get command output (already queued before the sentinel)
        try:
            command_output = fastworkflow.chat_session.command_output_queue.get_nowait()
        except queue.Empty:
            command_output = None

        if command_output:
            print_command_output(command_output)

    # Flush the observability writer so the session's last turn is never lost
    # [R7] (atexit also covers abnormal exits).
    if obs_sink is not None:
        obs_sink.close()

if __name__ == "__main__":
    print("Loading fastWorkflow...\n")

    parser = argparse.ArgumentParser(description="AI Assistant for workflow processing")
    parser.add_argument("workflow_path", help="Path to the workflow folder")
    parser.add_argument("env_file_path", help="Path to the environment file")
    parser.add_argument("passwords_file_path", help="Path to the passwords file")
    parser.add_argument(
        "--context_file_path", help="Optional context file path", default=""
    )
    parser.add_argument(
        "--startup_command", help="Optional startup command", default=""
    )
    parser.add_argument(
        "--startup_action", help="Optional startup action: a path to a JSON file, or an inline JSON object starting with '{'", default=""
    )
    parser.add_argument(
        "--keep_alive", help="Optional keep_alive", default=True
    )
    parser.add_argument(
        "--project_folderpath", help="Optional path to project folder containing application code", default=None
    )
    parser.add_argument(
        "--assistant", action="store_true", default=False,
        help="Run in assistant (non-agentic) mode. Default is agentic mode.",
    )
    parser.add_argument(
        "--timeout_seconds", type=float, default=None,
        help="Optional per-turn limit in seconds (default: none). Past it the terminal stops waiting and returns to the prompt; "
             "the turn finishes in the background and its result is discarded.",
    )
    parser.add_argument(
        "--generate_insights", action="store_true", default=False,
        help="Enable insights-distillation mode: compare teacher and student agents, extract planning and execution insights on divergence.",
    )
    args = parser.parse_args()
    run_main(args)
