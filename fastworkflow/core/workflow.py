import contextlib
import contextvars
import os
import sys
import threading
import weakref
from functools import wraps
from typing import Optional

import fastworkflow
from fastworkflow.session.kvstore import KVStore
from fastworkflow.utils.logging import logger

#: How many commands are running in this context (``command_executor.call_response_generator`` counts them). While
#: one runs, ``workflow.record`` is the record it was invoked on; outside every command it is the current position.
_RUNNING_COMMANDS: contextvars.ContextVar[int] = contextvars.ContextVar("fw_running_commands", default=0)


def command_running() -> bool:
    """Whether a command's code is running in this context (``workflow.record`` is then its invoked record)."""
    return _RUNNING_COMMANDS.get() > 0


#: ``(workflow, record)`` a command's checks act on in this context, bound by ``running_command(workflow, record)``, or
#: None: the record the dispatch bound on the workflow (``_command_context_for_response_generation``). A context-local
#: binding because the plan path checks the drafts of several parts at once (``subtask_pool``), each on its own record.
_BOUND_RECORD: contextvars.ContextVar[Optional[tuple]] = contextvars.ContextVar("fw_bound_record", default=None)


@contextlib.contextmanager
def running_command(workflow: Optional["Workflow"] = None, record: Optional[object] = None):
    """Mark a command's code as running for its duration (nested calls count).

    With *workflow*, ``workflow.record`` is *record* in this context while it runs: the binding of a command's checks
    made outside a dispatch (the plan path validates a part's arguments on the record its command will run on,
    ``intent_loop.on_owner_record``). Without it, the record is the one the dispatch bound on the workflow; a dispatch
    nested in a bound check reads its own."""
    token = _RUNNING_COMMANDS.set(_RUNNING_COMMANDS.get() + 1)
    bound = _BOUND_RECORD.set((workflow, record) if workflow is not None else None)
    try:
        yield
    finally:
        _BOUND_RECORD.reset(bound)
        _RUNNING_COMMANDS.reset(token)


# ----------------------------------------------------------------------
# Process-global, in-memory session-state registry.
#
# This replaces the two per-session speedict (RocksDB) backends that used to
# live on the hot path: the per-workflow snapshot DB and the global
# ``workflowid_2_sessiondata_map``.  ``Workflow.create`` previously paid ~6
# RocksDB open/close cycles per new session (snapshot + map, x2 for the cme and
# app workflows) even though the payload is trivial and all heavy definitions
# are already cached per folderpath (RoutingRegistry / command_directory).
#
# Session state is mutable but cheap and inherently per-process, so the live
# Workflow object *is* the store entry.  The registry holds only **weak**
# references: once the owning embedder (e.g. a WorkflowExecutionContext /
# ChannelRuntime) drops its reference and the Workflow is garbage-collected,
# its registry entry is reclaimed automatically.  This means abandoned sessions
# (e.g. a unique channel_id per request) do not accumulate — no manual cleanup
# or close() is required to avoid a leak.  Parent/child topology is stored on
# the live objects themselves (``_children``) so it is reclaimed with them.
#
# Durable cross-process state for the FastAPI service is owned separately by
# SessionStateStore (suspended turns) and the per-workflow observability DB
# (conversations, turn records, feedback); the CLI no longer resumes workflow
# context across process restarts (accepted trade-off).
#
# Remaining on-disk caches (enablecache below and the NLU suggested-commands state)
# use stdlib sqlite3 via :class:`fastworkflow.session.kvstore.KVStore` — JSON
# values, WAL mode, no process-exclusive LOCK.
# ----------------------------------------------------------------------
_STATE_LOCK = threading.RLock()
# workflow_id -> live Workflow (weak, so abandoned sessions auto-evict)
_WORKFLOW_REGISTRY: "weakref.WeakValueDictionary[int, Workflow]" = (
    weakref.WeakValueDictionary()
)


# implements the enablecache decorator
def enablecache(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # Create a cache key based on the function arguments
        key = str(args) + str(kwargs)

        # Get the cache database (folder path historically fed RocksDB; SQLite
        # needs a file inside that folder).
        cache_db_folder = self.get_cachedb_folderpath(func.__name__)
        with KVStore(os.path.join(cache_db_folder, "cache.sqlite3")) as cache_db:
            if key not in cache_db:
                # If the result is not in the cache, call the function and store the result
                result = func(self, *args, **kwargs)
                try:
                    cache_db[key] = result
                except TypeError as exc:
                    raise TypeError(
                        f"@enablecache requires a JSON-serialisable return value; "
                        f"{func.__qualname__} returned {type(result).__name__}"
                    ) from exc
            else:
                result = cache_db[key]

        return result

    return wrapper

# Attribute names removed from Workflow, each mapped to its replacement. Reading or
# assigning one raises AttributeError naming the replacement (Workflow.__getattr__ / __setattr__).
_RETIRED_CONTEXT_ATTRIBUTES = {
    "command_context_for_response_generation": "is removed: use workflow.record",
    "current_command_context": "is removed: use workflow.record (assigning it moves the workflow)",
    "root_command_context": "is removed: use workflow.root",
    "is_command_context_for_response_generation_root": "is removed: use workflow.is_record_root",
    "current_command_context_name": "is internal: read the record with workflow.record and its context class's get_displayname",
    "current_command_context_displayname": "is internal: read the record with workflow.record and its context class's get_displayname",
    "is_current_command_context_root": "is internal: read the record with workflow.record and compare it with workflow.root",
}


class MissingGetParentError(AttributeError):
    """A workflow with a root stands on a record that is not the root, and the record's context does not say what
    encloses it: no ``_commands/<Context>/_<Context>.py`` or no ``get_parent`` in its ``class Context``; or its
    ``get_parent`` returned a record of a context its declared ``parents`` do not name. Raised on the
    climb (``Workflow.get_parent``); a command that moves onto such a record fails instead (``command_executor``)."""


class Workflow:
    """Workflow class"""
    @classmethod
    def create(
        cls, 
        workflow_folderpath: str, 
        workflow_id_str: Optional[str] = None, 
        parent_workflow_id: Optional[int] = None, 
        workflow_context: dict = None,
        project_folderpath: Optional[str] = None
    ) -> "Workflow":
        if workflow_id_str is not None and parent_workflow_id is not None:
            raise ValueError("workflow_id_str and parent_workflow_id cannot both be provided")

        if not os.path.exists(workflow_folderpath):
            raise ValueError(f"The folder path {workflow_folderpath} does not exist")

        if not os.path.isdir(workflow_folderpath):
            raise ValueError(f"{workflow_folderpath} must be a directory")

        if workflow_id_str:
            workflow_id = fastworkflow.get_workflow_id(workflow_id_str)
        else:
            workflow_id = cls.generate_child_workflow_id(workflow_folderpath, parent_workflow_id)

        python_project_path = project_folderpath or workflow_folderpath
        if python_project_path not in sys.path:
            # THIS IS IMPORTANT: it allows relative import of modules in the code inside workflow_folderpath
            sys.path.insert(0, python_project_path)

        if workflow := cls.get_workflow(workflow_id):
            if workflow_context is not None:
                workflow.context = workflow_context
            return workflow

        # Resolve the workflow path to ensure consistent cache keys
        from pathlib import Path
        resolved_workflow_path = str(Path(workflow_folderpath).resolve())
        
        workflow_snapshot = {
            "workflow_id": workflow_id,
            "workflow_folderpath": resolved_workflow_path,
            "workflow_context": workflow_context or {},
            "parent_workflow_id": parent_workflow_id,
            "is_complete": False
        }
        workflow = Workflow(cls.__create_key, workflow_snapshot)

        with _STATE_LOCK:
            if workflow.parent_id:
                parent = _WORKFLOW_REGISTRY.get(workflow.parent_id)
                if parent is not None and workflow.id not in parent._children:
                    parent._children.append(workflow.id)

        return workflow

    @classmethod
    def get_workflow(cls, workflow_id: int) -> Optional["Workflow"]:
        """return the live workflow from the in-memory registry (or None)"""
        with _STATE_LOCK:
            return _WORKFLOW_REGISTRY.get(workflow_id)

    @classmethod
    def generate_child_workflow_id(cls, workflow_folderpath: str, parent_workflow_id: Optional[int] = None) -> int:
        """generate a child workflow id"""
        workflow_type = os.path.basename(workflow_folderpath).rstrip("/")
        workflow_id_str = f"{parent_workflow_id}{workflow_type}" \
                         if parent_workflow_id \
                         else f"{workflow_type}"

        return fastworkflow.get_workflow_id(workflow_id_str)

    # enforce workflow creation exclusively using Workflow.create
    # https://stackoverflow.com/questions/8212053/private-constructor-in-python
    __create_key = object()
   
    def __init__(self, create_key, workflow_snapshot: dict[str, str|int|bool]):
        """initialize the Workflow class"""
        if create_key is not Workflow.__create_key:
            raise ValueError("Workflow objects must be created using Workflow.create")

        from pathlib import Path
        self._id = workflow_snapshot["workflow_id"]
        # Always resolve the workflow path to ensure consistent cache keys
        self._folderpath = str(Path(workflow_snapshot["workflow_folderpath"]).resolve())
        self._parent_id = workflow_snapshot.get("parent_workflow_id")
        self._is_complete = workflow_snapshot.get("is_complete", False)
        self._context = workflow_snapshot.get("workflow_context", {}) 

        self._root_command_context = None
        self._current_command_context = None
        self._command_context_for_response_generation = None
        # How many times ``workflow.record`` was assigned: a command moved the session when this changed while it ran
        # (``command_executor._check_move``). Setting ``workflow.root`` is not a record move and does not count.
        self._record_assignments = 0

        # Observers invoked when the current position changes (workflow.record = x). Kept on
        # the workflow because the setter is the single chokepoint every context switch
        # (set_current_*, go_up, reset_context) passes through.
        self._context_change_listeners: list = []

        # Child workflow ids (parent/child topology lives on the live object so
        # it is reclaimed when the object is garbage-collected).
        self._children: list[int] = []

        # ------------------------------------------------------------------
        # Registration control
        # ------------------------------------------------------------------
        # ``_dirty`` tracks whether state has changed since the last flush.
        # We register the freshly-constructed workflow immediately so that
        # get_workflow() can find it, then mark it clean.
        self._dirty: bool = False
        self._save()
        self._dirty = False

    @property
    def _current_command_context_name(self) -> str:
        return Workflow.get_command_context_name(self._current_command_context)

    @property
    def _current_command_context_displayname(self) -> str:
        return self._displayname_of(self._current_command_context)

    @property
    def record_displayname(self) -> str:
        """The display name of ``workflow.record``, as a command prints it."""
        return self._displayname_of(self.record)

    def _displayname_of(self, record: Optional[object]) -> str:
        crd = fastworkflow.CommandContextModel.load(self._folderpath)
        context_class = crd.get_context_class(
                Workflow.get_command_context_name(record),
                fastworkflow.ModuleType.CONTEXT_CLASS
        )
        if context_class and hasattr(context_class, 'get_displayname'):
            return context_class.get_displayname(record)

        return Workflow.get_command_context_name(record, for_display=True)


    @property
    def _is_current_command_context_root(self) -> bool:
        return self._current_command_context is self._root_command_context

    def add_context_change_listener(self, listener) -> None:
        """Register a zero-arg callable invoked whenever the current context changes.

        Bound methods are stored as ``weakref.WeakMethod`` so a short-lived
        WorkflowExecutionContext (or similar) can be GC'd without an explicit
        unsubscribe. Plain callables (including closures) are stored strongly;
        callers that register those should call ``remove_context_change_listener``.
        """
        # Bound methods: WeakMethod so WEC/agent GC is not blocked by the list.
        if getattr(listener, "__self__", None) is not None and hasattr(listener, "__func__"):
            try:
                self._context_change_listeners.append(weakref.WeakMethod(listener))
                return
            except TypeError:
                pass
        self._context_change_listeners.append(listener)

    def remove_context_change_listener(self, listener) -> None:
        """Unregister a previously added context-change listener (no-op if absent)."""
        remaining = []
        for entry in self._context_change_listeners:
            if isinstance(entry, weakref.WeakMethod):
                current = entry()
                if current is None:
                    continue
                if current is listener or current == listener:
                    continue
                remaining.append(entry)
            elif entry is listener or entry == listener:
                continue
            else:
                remaining.append(entry)
        self._context_change_listeners = remaining

    def _notify_context_change(self) -> None:
        alive = []
        for entry in list(self._context_change_listeners):
            if isinstance(entry, weakref.WeakMethod):
                listener = entry()
                if listener is None:
                    continue
            else:
                listener = entry
            alive.append(entry)
            try:
                listener()
            except Exception:  # a listener must never break context switching
                logger.warning("context-change listener failed", exc_info=True)
        self._context_change_listeners = alive

    @property
    def record(self) -> object:
        """The record the running command acts on: the one it was invoked on; outside a command, the current position.

        While a command runs it differs from the current position when the command was found on an
        enclosing record through ``get_parent`` (or is a global command, run on the root). It keeps returning
        the invoked record until the command returns, so a command that moves the workflow reads the record
        once into a local, then assigns ``workflow.record = child`` to move. Assigning it sets the current
        position, which is what listings and navigation read, and what a read outside a command returns.
        """
        if command_running():
            bound = _BOUND_RECORD.get()
            return bound[1] if bound is not None and bound[0] is self else self._command_context_for_response_generation
        return self._current_command_context

    @record.setter
    def record(self, value: Optional[object]) -> None:
        self._record_assignments += 1
        changed = value is not self._current_command_context
        self._current_command_context = value
        if changed:
            self._notify_context_change()

    @property
    def root(self) -> object:
        """The root record of this workflow, set once in the startup command."""
        return self._root_command_context

    @root.setter
    def root(self, value: Optional[object]) -> None:
        if self._root_command_context is not None:
            raise ValueError("workflow.root can only be set once per Workflow")

        self._root_command_context = value
        self._current_command_context = value

    def get_parent(self, record: Optional[object] = None) -> Optional[object]:
        """The record that encloses *record*: one rule keyed on the root.

        The root (and None) answers the root. With no root, a context with no ``_<Context>.py``, no ``get_parent`` or
        a ``get_parent`` that returns None all mean "no enclosing record: straight to the global commands". With a
        root, a record that is not the root needs its context's answer: ``return None`` is the explicit "go to the
        global commands", and a context with no file or no ``get_parent`` raises ``MissingGetParentError``. A record
        whose class is no context of the workflow has no enclosing record (a command cannot move onto one).
        """
        if record is None or record is self._root_command_context:
            return self._root_command_context
        context_class, problem = self._enclosing_rule(record)
        if problem is not None:
            raise MissingGetParentError(problem)
        if context_class is None or not hasattr(context_class, "get_parent"):
            return None
        parent = context_class.get_parent(record)
        if parent is not None and (problem := self._wrong_parent(record, parent)):
            raise MissingGetParentError(problem)
        return parent

    def _wrong_parent(self, record: object, parent: object) -> Optional[str]:
        """Why *parent*, what *record*'s ``get_parent`` returned, cannot enclose it, or None: its context is neither
        one of the context's declared ``parents`` nor a context that inherits one (a PremiumUser is a User)."""
        model = fastworkflow.CommandContextModel.load(self._folderpath)
        context_name = Workflow.get_command_context_name(record)
        declared = model.parents(context_name)
        if declared is None:
            return None   # a get_parent with no parents is a load error of the context class
        parent_context = Workflow.get_command_context_name(parent)
        if {parent_context, *model.inherited_base_contexts(parent_context)} & set(declared):
            return None
        return (f"the get_parent of context {context_name} returned a {parent_context} record, but {context_name} "
                f"declares parents = {declared!r} in _commands/{context_name}/_{context_name}.py: a {context_name} "
                f"record sits only in those contexts (or one that inherits one). Return the enclosing record, or add "
                f"{parent_context!r} to parents")

    def missing_get_parent(self, record: Optional[object]) -> Optional[str]:
        """Why the climb from *record* cannot go on, or None: the workflow has a root, *record* is not it, its class is
        a context of the workflow, and that context has no ``get_parent`` (no ``_<Context>.py``, or none in its
        ``class Context``; one inherited from a Python base class counts). The text names the context, the file to
        write and the method."""
        if record is None or record is self._root_command_context:
            return None
        return self._enclosing_rule(record)[1]

    def is_context(self, context_name: str, model=None) -> bool:
        """Whether *context_name* is a context of this workflow: a folder ``_commands/<context_name>/`` holding a
        command, an entry of its inheritance file, or a context with a registered ``_<context_name>.py``."""
        from fastworkflow.core.command_directory import get_cached_command_directory

        model = model or fastworkflow.CommandContextModel.load(self._folderpath)
        return (context_name in model._command_contexts
                or context_name in get_cached_command_directory(self._folderpath).map_context_2_metadata)

    def _enclosing_rule(self, record: object) -> tuple[object, Optional[str]]:
        """``(context class or None, the MissingGetParentError text or None)`` for a record that is not the root."""
        model = fastworkflow.CommandContextModel.load(self._folderpath)
        context_name = Workflow.get_command_context_name(record)
        context_class = model.get_context_class(context_name, fastworkflow.ModuleType.CONTEXT_CLASS)
        if (self._root_command_context is None or hasattr(context_class, "get_parent")
                or (context_class is None and not self.is_context(context_name, model))):
            return context_class, None
        where = f"_commands/{context_name}/_{context_name}.py"
        missing = (f"its class Context in {where} has no get_parent" if context_class is not None
                   else f"there is no {where}")
        return context_class, (
            f"the context {context_name} has no get_parent ({missing}): the workflow has a root and a {context_name} "
            f"record is not it, so the framework asks its context for the record that encloses it. Write "
            f"'@classmethod def get_parent(cls, record)' in the class Context of {where}, returning the enclosing "
            f"record (return None to go straight to the global commands)")

    @property
    def is_record_root(self) -> bool:
        """Whether ``workflow.record`` is the root record (see ``workflow.root``)."""
        return self.record is self._root_command_context

    @staticmethod
    def get_command_context_name(record: Optional[object], for_display = False) -> str:
        if record is not None:
            return record.__class__.__name__
        return 'global' if for_display else '*'

    def __getattr__(self, name: str):
        # Only reached when normal lookup fails, so it catches exactly the retired
        # spellings. A getattr(workflow, name, default) call would otherwise hide the
        # removal, which is why the message names the replacement.
        replacement = _RETIRED_CONTEXT_ATTRIBUTES.get(name)
        if replacement is not None:
            raise AttributeError(f"Workflow.{name} {replacement}")
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

    def __setattr__(self, name: str, value) -> None:
        # Without this, ``workflow.current_command_context = x`` would silently create
        # a plain attribute and leave the workflow where it was.
        replacement = _RETIRED_CONTEXT_ATTRIBUTES.get(name)
        if replacement is not None:
            raise AttributeError(f"Workflow.{name} {replacement}")
        object.__setattr__(self, name, value)

    @property
    def id(self) -> int:
        """get the workflow id"""
        return self._id
    
    @property
    def parent_id(self) -> Optional[int]:
        """get the parent workflow id"""
        return self._parent_id

    @property
    def folderpath(self) -> str:
        return self._folderpath

    @folderpath.setter
    def folderpath(self, value: str) -> None:
        self._folderpath = value
        self._mark_dirty()

    @property
    def context(self) -> dict:
        return self._context
    
    @context.setter
    def context(self, value: dict) -> None:
        self._context = value
        self._mark_dirty()

    @property
    def is_complete(self) -> bool:
        return self._is_complete
    
    @is_complete.setter
    def is_complete(self, value: bool) -> None:
        self._is_complete = value
        self._mark_dirty()
    
    def end_command_processing(self) -> None:
        """Process the end of a command"""
        # important to clear the current command from the workflow context
        if "command" in self._context:
            del self._context["command"]

        # important to clear the pending question (parameter extraction error state, a which-command question): it lives
        # in the turn's waiting record, reached through the port (a context with none holds nothing of its own)
        from fastworkflow.core import waiting

        if (port := self._context.get(waiting.PORT_KEY)) is not None:
            port.clear()

        self._mark_dirty()

    def close(self) -> bool:
        """close the session: explicitly evict it (and its descendants) from the registry.

        Eviction is also automatic once all strong references to a Workflow are
        dropped (the registry holds only weak references); close() is the
        explicit, immediate path and additionally tidies sys.path.
        """
        if self.parent_id:
            raise ValueError("close should only be called on the root session")

        with _STATE_LOCK:
            # collect all (still-live) descendants
            descendant_list = []
            to_process = list(self._children)
            while to_process:
                current_id = to_process.pop()
                descendant_list.append(current_id)
                child = _WORKFLOW_REGISTRY.get(current_id)
                if child is None:
                    # Already garbage-collected (and thus auto-evicted).
                    continue
                if child._folderpath in sys.path:
                    sys.path.remove(child._folderpath)  # remove the workflow folderpath from sys.path
                # Add any children to the processing queue
                to_process.extend(child._children)

            # process all descendants
            for descendant_workflow_id in descendant_list:
                _WORKFLOW_REGISTRY.pop(descendant_workflow_id, None)

            if self._folderpath in sys.path:
                sys.path.remove(self._folderpath)
            # remove ourselves from the registry
            _WORKFLOW_REGISTRY.pop(self.id, None)

        return True

    @classmethod
    def _get_session_data(cls, workflow_id: int) -> tuple[int, Optional[str], list, str]:
        """get the parent id, (legacy, now None) session db folder path, the children list, and the workflow folderpath"""
        with _STATE_LOCK:
            workflow = _WORKFLOW_REGISTRY.get(workflow_id)

        if workflow is None:
            raise ValueError(f"Workflow {workflow_id} not found")

        return (
            workflow._parent_id,
            None,
            workflow._children,
            workflow._folderpath
        )

    def get_cachedb_folderpath(self, function_name: str) -> str:
        """Get the cache database folder path for a specific function"""
        from fastworkflow.session import state_paths
        return state_paths.function_cache_dir(self._folderpath, function_name)

    @classmethod
    def _load(cls, workflow_id: int) -> dict[str, str|int|bool]:
        """load the workflow snapshot from the in-memory registry"""
        with _STATE_LOCK:
            workflow = _WORKFLOW_REGISTRY.get(workflow_id)
            if workflow is None:
                raise ValueError(
                    f"Workflow {workflow_id} not found in session-state registry"
                )
            return workflow._to_dict()

    def _save(self) -> None:
        """register the live workflow in the in-memory registry (idempotent)"""
        with _STATE_LOCK:
            _WORKFLOW_REGISTRY[self._id] = self

    def _to_dict(self) -> dict[str, str|int|bool]:
        """Return a JSON-serialisable representation of the workflow."""
        return {
            "workflow_id": self._id,
            "workflow_folderpath": self._folderpath,
            "parent_workflow_id": self._parent_id,
            "is_complete": self._is_complete,
            "workflow_context": self._context
        }

    # ------------------------------------------------------------------
    # Deferred-save helpers
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Flag that the workflow state has changed since the last flush."""
        self._dirty = True

    def flush(self) -> None:
        """Re-affirm registration if state changed.

        Mutable state lives on the live object itself, so this is effectively a
        no-op (besides re-registering); it is retained for API compatibility and
        so the registry entry survives even if a key was evicted mid-lifetime.
        """
        if self._dirty:
            self._save()
            self._dirty = False
