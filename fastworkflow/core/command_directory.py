import ast
import contextlib
import functools
import hashlib
import os
import re
from typing import Optional
from pathlib import Path
from functools import lru_cache

from pydantic import BaseModel, field_validator, model_serializer, model_validator, ConfigDict, Field, ValidationInfo

from fastworkflow.utils import python_utils
from fastworkflow.utils.logging import logger
from fastworkflow.workflow_inheritance_model import WorkflowInheritanceModel


#: The framework-provided planning commands (global scope, lowest precedence).
FRAMEWORK_PLANNING_COMMANDS_DIR = Path(__file__).resolve().parent / "review" / "commands"

class CommandMetadata(BaseModel):
    """Command metadata"""
    # parameter extraction
    workflow_folderpath: Optional[str] = None
    parameter_extraction_signature_module_path: Optional[str] = None
    input_for_param_extraction_class: Optional[str] = None
    command_parameters_class: Optional[str] = None
    # response generation
    response_generation_module_path: str
    response_generation_class_name: str
    is_planning: bool = False
    # A framework-provided command (fastworkflow/review/commands, fix-3rsq.plan-review.1): system machinery.
    # Loaded and callable by name through the executor and loader, but never listed to the planner, the agent or
    # the user, never a head label, and never counted by the artifact construction check.
    framework: bool = False
    # The command's ``@user_approval_for_execution`` value ("required_always" | "required_until_delegated"),
    # set from its class when the routing definition loads (approval_annotations). None is agent discretion
    # and is not serialised.
    approval: Optional[str] = None

    @model_serializer(mode="wrap")
    def _omit_absent_approval(self, handler):
        data = handler(self)
        if isinstance(data, dict) and data.get("approval") is None:
            data.pop("approval", None)
        return data

    @model_validator(mode="after")
    def validate_parameter_extraction_fields(self) -> "CommandMetadata":
        rg_fields = [
            self.response_generation_module_path,
            self.response_generation_class_name,
        ]

        if not (all(rg_fields)):
            raise ValueError("Response generation modules must be populated.")

        return self


#: ``UtteranceMetadata.generated_utterances_func_name`` of a command whose ``Signature``
#: defines no ``generate_utterances``: the framework's default (``default_generate_utterances``).
DEFAULT_UTTERANCES_FUNC = "<default>"


def default_generate_utterances(signature, workflow, command_name: str) -> list[str]:
    """The utterances of a command whose ``Signature`` defines no ``generate_utterances``:
    exactly what the standard command-file body returns (``build/command_file_template``):
    the command's name in words, then ``generate_diverse_utterances`` of its
    ``plain_utterances``. A ``Signature`` that defines (or inherits) ``generate_utterances``
    keeps its own; this is never used for it."""
    from fastworkflow.train import generate_synthetic

    return [command_name.split("/")[-1].lower().replace("_", " ")] + \
        generate_synthetic.generate_diverse_utterances(
            getattr(signature, "plain_utterances", []), command_name)


class UtteranceMetadata(BaseModel):
    workflow_folderpath: Optional[str] = None
    plain_utterances: list[str]
    template_utterances: list[str] = []
    generated_utterances_module_filepath: str
    generated_utterances_func_name: str

    def get_generated_utterances_func(self, workflow_folderpath: str) -> list[str]:      
        module = python_utils.get_module(
            self.generated_utterances_module_filepath, 
            self.workflow_folderpath or workflow_folderpath
        )

        # Get the function from the module and execute it
        if not module or not self.generated_utterances_func_name:
            return None

        if self.generated_utterances_func_name == DEFAULT_UTTERANCES_FUNC:
            signature = getattr(module, "Signature", None)
            return functools.partial(default_generate_utterances, signature) if signature else None

        if '.' in self.generated_utterances_func_name:
            parts = self.generated_utterances_func_name.split('.')
            func = module
            for part in parts:
                func = getattr(func, part)
            return func

        return getattr(module, self.generated_utterances_func_name, None)

    @field_validator("plain_utterances", mode="before")
    def parse_plain_utterances(cls, plain_utterances: list[str]):
        for s in plain_utterances:
            if not s:
                raise ValueError("Plain utterance string cannot be empty")
        return plain_utterances

    @field_validator("template_utterances", mode="before")
    def parse_template_utterances(cls, template_utterances: list[str]):
        for s in template_utterances:
            if not s:
                raise ValueError("Plain utterance string cannot be empty")
        return template_utterances

    @field_validator("generated_utterances_module_filepath", mode="before")
    def parse_generated_utterances_module_path(
        cls, generated_utterances_module_filepath: str
    ):
        if not generated_utterances_module_filepath:
            raise ValueError("Generated utterances module path cannot be empty")
        return generated_utterances_module_filepath

    @field_validator("generated_utterances_func_name", mode="before")
    def parse_generatd_utterances_func_name(cls, generated_utterances_func_name: str):
        if not generated_utterances_func_name:
            raise ValueError("Generated utterances function name cannot be empty")
        return generated_utterances_func_name


class ContextMetadata(BaseModel):
    """Metadata describing a context callback implementation module."""

    workflow_folderpath: Optional[str] = None
    context_module_path: str
    context_class: str

    @field_validator("context_module_path", mode="before")
    def validate_context_module_path(cls, v):
        if not v:
            raise ValueError("context_module_path cannot be empty")
        return v

    @field_validator("context_class", mode="before")
    def validate_context_class(cls, v):
        if not v:
            raise ValueError("context_class cannot be empty")
        return v


def is_visible_planning(metadata: "CommandMetadata") -> bool:
    """A planning command the planner, the agent, the listings, the labels and the artifact check see:
    a planning command that is not framework machinery."""
    return bool(metadata.is_planning) and not metadata.framework


class CommandDirectory(BaseModel):
    """
    Centralized directory for all command implementations.
    Each command is uniquely identified by its name.
    """
    workflow_folderpath: str

    map_command_2_metadata: dict[str, CommandMetadata] = Field(default_factory=dict)
    map_command_2_utterance_metadata: dict[str, UtteranceMetadata] = Field(default_factory=dict)
    core_command_names: list[str] = Field(default_factory=list)
    # Mapping of context name to its metadata containing callback class module information
    map_context_2_metadata: dict[str, "ContextMetadata"] = Field(default_factory=dict)
    # Fingerprint of the command sources this snapshot was built from. Used to
    # invalidate the persisted cache when commands are added/removed/renamed/edited.
    source_fingerprint: Optional[str] = None
    # The command files that could not be read, by command name: the file does not parse, defines both generator
    # classes, or does not import. Such a command is not registered (not routable, not trained, not listed); its
    # message names the file and is listed by ``command_preview.load_problems``. A mistake found later, by the checks
    # that need the imported classes, is the command's load error too (``command_load_errors``).
    load_errors: dict[str, str] = Field(default_factory=dict)

    def record_load_error(self, command_name: str, command_file_path: str | Path, error: BaseException) -> None:
        """Record that *command_name*'s file cannot be read and unregister the command: the rest of the workflow
        still loads. The message names the file (relative to the workflow folder)."""
        path = str(command_file_path)
        shown = os.path.relpath(path, self.workflow_folderpath) if os.path.isabs(path) else path
        if isinstance(error, ImportError) and error.__cause__ is not None:
            error = error.__cause__     # python_utils.get_module wraps the import's own error
        text = str(error)
        if text.startswith(path):
            text = text[len(path):].lstrip(": ")
        if not isinstance(error, CommandFileError):
            text = f"{type(error).__name__}: {text}"
        if self.workflow_folderpath:
            text += raised_in_helper(error, path, self.workflow_folderpath)
        self.load_errors[command_name] = f"{shown}: {text}"
        self.map_command_2_metadata.pop(command_name, None)
        self.map_command_2_utterance_metadata.pop(command_name, None)

    def register_command_metadata(self, command_name: str, metadata: CommandMetadata):
        if not (command_name and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_command_2_metadata[command_name] = metadata

    def get_commands(self) -> list[str]:
        """Retrieve all command keys registered in the command directory."""
        return list(self.map_command_2_metadata.keys())

    def runs_only_as_action(self, command_name: str) -> bool:
        """Whether *command_name* is a command with no ``Signature`` (``declares_signature``), which runs only as an
        action: never by name, by words or from a listing. False for a name the directory does not hold."""
        metadata = self.map_command_2_metadata.get(command_name)
        return metadata is not None and not declares_signature(metadata.response_generation_module_path)

    def get_execution_commands(self) -> list[str]:
        """Command names that are not planning commands (routing, training, catalogs)."""
        return [
            name
            for name, metadata in self.map_command_2_metadata.items()
            if not metadata.is_planning and not metadata.framework
        ]

    def get_command_metadata(self, command_name: str) -> CommandMetadata:
        if command_name not in self.map_command_2_metadata:
            raise KeyError(f"Command key '{command_name}' not found.")
        return self.map_command_2_metadata[command_name]

    def register_utterance_metadata(self, command_name: str, metadata: UtteranceMetadata):
        if not (command_name and metadata):
            raise ValueError("command_key and metadata are required")
        self.map_command_2_utterance_metadata[command_name] = metadata

    def get_utterance_keys(self) -> list[str]:
        """Retrieve all utterance keys registered in the command directory."""
        return list(self.map_command_2_utterance_metadata.keys())

    def get_utterance_metadata(self, command_key: str) -> Optional[UtteranceMetadata]:
        if meta := self.map_command_2_utterance_metadata.get(command_key):
            return meta

        # -------- lazy hydration --------
        if command_key in self.map_command_2_metadata:
            self.ensure_command_hydrated(command_key)
            if meta := self.map_command_2_utterance_metadata.get(command_key):
                return meta
        
        return None

    @classmethod
    def get_commandinfo_folderpath(cls, workflow_folderpath: str) -> str:
        command_info_folderpath = Path(workflow_folderpath) / "___command_info"
        command_info_folderpath.mkdir(parents=True, exist_ok=True)
        return str(command_info_folderpath)

    def save(self):
        commandroutinginfo_folderpath = CommandDirectory.get_commandinfo_folderpath(self.workflow_folderpath)
        # Stamp the current source fingerprint so the persisted snapshot can be
        # validated (and invalidated) against the live _commands tree on reload.
        self.source_fingerprint = compute_commands_source_fingerprint(self.workflow_folderpath)
        with open(Path(commandroutinginfo_folderpath) / "command_directory.json", "w") as f:
            f.write(self.model_dump_json(indent=4))

    @classmethod
    def load(
        cls,
        workflow_folderpath: str,
        *,
        hydrate_utterances: bool = True,
        best_effort_inheritance: bool = False,
    ) -> "CommandDirectory":
        """
        Loads the command directory by scanning _commands folders from base workflows 
        and the current workflow, implementing workflow inheritance.
        
        Search order (last occurrence wins):
        1. Each path in workflow_inheritance_model.json["base"] (in order) -> their _commands folders
        2. <workflow_root>/_commands (current workflow - highest precedence)
        3. Built-in core workflow fastworkflow/_workflows/command_metadata_extraction
        
        Commands in context subdirectories are registered with qualified names 
        in the format 'ContextName/command_name'.
        """
        # Command file paths are recorded in `command_directory.json` and in the training signature, so they
        # must not take their spelling (and the current directory) from a relative workflow path.
        workflow_folderpath = os.path.abspath(workflow_folderpath)
        command_directory = cls(workflow_folderpath=workflow_folderpath)
        
        # Step 1: Load workflow inheritance model
        try:
            inheritance_model = WorkflowInheritanceModel.load(workflow_folderpath)
            base_paths = inheritance_model.resolve_base_paths(workflow_folderpath)
        except Exception as e:
            if best_effort_inheritance:
                base_paths = []
            else:
                raise RuntimeError(f"Error processing workflow inheritance: {e}") from e

        # A retired _skills/ folder (here or in a base) is a load error: its planning commands are not read
        for path in (*base_paths, workflow_folderpath):
            check_no_planning_commands_folder(path)
        
        # Step 2: framework-provided planning commands, at the LOWEST precedence (fix-3rsq.plan-review.1):
        # loaded first, so a base or app command of the same global name replaces them.
        cls._load_framework_planning_commands(command_directory)

        # Step 3: Process base workflows (in order of precedence)
        for base_path in base_paths:
            cls._load_commands_from_path(command_directory, base_path)
        
        # Step 4: Process current workflow commands (highest precedence for user commands)
        cls._load_commands_from_path(command_directory, workflow_folderpath)

        # Step 5: Populate utterance metadata when available (optional; defers imports)
        if hydrate_utterances:
            for command_key in list(command_directory.map_command_2_metadata):
                try:
                    utterance_metadata = command_directory.get_utterance_metadata(command_key)
                except Exception as error:  # noqa: BLE001 - a file that does not import costs that command only
                    metadata = command_directory.map_command_2_metadata[command_key]
                    command_directory.record_load_error(
                        command_key, metadata.response_generation_module_path, error)
                    continue
                if utterance_metadata:
                    command_directory.register_utterance_metadata(command_key, utterance_metadata)

        # Step 6: Register core commands (processed last, lowest precedence)
        cls._register_core_commands(command_directory)

        return command_directory

    @classmethod
    def scan(cls, workflow_folderpath: str) -> "CommandDirectory":
        """Load command metadata without hydrating utterances (no imports, no ``___command_info``)."""
        return cls.load(
            workflow_folderpath,
            hydrate_utterances=False,
            best_effort_inheritance=True,
        )
    
    @classmethod
    def _load_framework_planning_commands(cls, command_directory: "CommandDirectory") -> None:
        """Register the framework's global planning commands: the ``.py`` files directly in
        ``fastworkflow/review/commands`` (no context folder, so the global ``*`` scope). They are
        ordinary planning commands in every other respect (the same loader as an app's).
        """
        root = FRAMEWORK_PLANNING_COMMANDS_DIR
        if not root.is_dir():
            return
        # The framework's own internal workflows (the command metadata extraction workflow, which holds the
        # core commands) do not take the global planning commands: their trained artifacts predate them.
        import fastworkflow  # inline: the module is imported before the package is ready
        if os.path.abspath(command_directory.workflow_folderpath) in {
                os.path.abspath(fastworkflow.get_internal_workflow_path("command_metadata_extraction")),
                os.path.abspath(fastworkflow.get_internal_workflow_path("plan_metadata_extraction")),
        }:
            return
        for item_path in sorted(root.glob("*.py")):
            if item_path.is_file() and not item_path.name.startswith("_"):
                cls._register_command(command_directory, item_path.stem, item_path, str(root))
                command_directory.map_command_2_metadata[item_path.stem].framework = True

    @classmethod
    def _load_commands_from_path(cls, command_directory: "CommandDirectory", path: str):
        """
        Load commands from a specific workflow path.
        
        This method handles loading commands from both base workflows and the current workflow.
        Commands are registered with the command directory, with later calls overriding earlier ones
        for the same command name (implementing precedence).
        
        Args:
            command_directory: The CommandDirectory instance to register commands to
            path: Path to the workflow folder containing _commands/
        """
        commands_root_folder = Path(path) / "_commands"
        
        if not commands_root_folder.is_dir():
            # Skip silently if no _commands directory exists
            # This allows for base workflows that may not have commands
            return
            
        # Process files directly under _commands/ (global commands)
        for item_path in commands_root_folder.glob("*.py"):
            if item_path.is_file() and not item_path.name.startswith("_") and item_path.name != "__init__.py":
                command_name = item_path.stem
                cls._register_command_file(command_directory, command_name, item_path, path)
        
        # Process subdirectories (contexts) and their commands. Every such folder is a context, whether or not it
        # holds a command file of its own (a context whose commands are all inherited holds only its context class
        # file); the context model refuses a folder that holds no command file and inherits none.
        for context_dir in commands_root_folder.iterdir():
            if context_dir.is_dir() and not context_dir.name.startswith("_"):
                context_name = context_dir.name
                for command_file in context_dir.glob("*.py"):
                    if not command_file.name.startswith("_") and command_file.name != "__init__.py":
                        # Use qualified name format: "ContextName/command_name"
                        qualified_command_name = f"{context_name}/{command_file.stem}"
                        cls._register_command_file(command_directory, qualified_command_name, command_file, path)
                cls._register_context_class(command_directory, context_name, context_dir, path)
    
    @classmethod
    def _register_command_file(cls, command_directory: "CommandDirectory", command_name: str,
                               command_file_path: Path, workflow_folderpath: str) -> None:
        """Register one command file; a file that cannot be read (it does not parse, or it defines both generator
        classes) is that command's load error, not the workflow's."""
        try:
            cls._register_command(command_directory, command_name, command_file_path, workflow_folderpath)
        except (SyntaxError, ValueError) as error:
            command_directory.record_load_error(command_name, command_file_path, error)
            return
        # a later workflow's file of the same name replaces a base's, broken or not
        command_directory.load_errors.pop(command_name, None)

    @staticmethod
    def check_planning_command_path(name: str, path: str) -> None:
        """Validate a planning command's file name before load: its stem is a Python identifier.

        Raises :class:`fastworkflow.planning.planning_commands.PlanningCommandFormatError` when it is not.
        """
        # planning_commands imports command_directory at load (via command_context_model); import here to break the cycle
        from fastworkflow.planning.planning_commands import PlanningCommandFormatError

        _context, _, stem = name.rpartition("/")
        if not stem.isidentifier():
            raise PlanningCommandFormatError(
                f"{path}: a planning command's name is its file name, a Python identifier "
                f"(snake_case): rename it to {re.sub(r'[^0-9A-Za-z_]', '_', stem)}.py"
            )

    @classmethod
    def _register_command(
        cls,
        command_directory: "CommandDirectory",
        command_name: str,
        command_file_path: Path,
        workflow_folderpath: str,
        *,
        response_generation_class_name: str | None = None,
        is_planning: bool | None = None,
    ):
        """Helper method to register a command and its metadata"""
        command_filepath_str = str(command_file_path)

        # ------------------------------------------------------------
        # Lazy-loading variant: we avoid importing the command module
        # here.  Only the bare minimum metadata needed for routing is
        # stored up-front; Signature/utterance data is hydrated the first
        # time it is requested (see `ensure_command_hydrated`).
        # ------------------------------------------------------------

        if response_generation_class_name is None or is_planning is None:
            rg_name, detected_planning = _command_kind_from_ast(command_file_path)
            if response_generation_class_name is None:
                response_generation_class_name = rg_name
            if is_planning is None:
                is_planning = detected_planning

        metadata = CommandMetadata(
            workflow_folderpath=workflow_folderpath,
            # Defer Signature import – fill these lazily
            parameter_extraction_signature_module_path=None,
            input_for_param_extraction_class=None,
            command_parameters_class=None,
            # Response generation path is still required immediately
            response_generation_module_path=command_filepath_str,
            response_generation_class_name=response_generation_class_name,
            is_planning=is_planning,
        )
        command_directory.register_command_metadata(command_name, metadata)

    @staticmethod
    def _populate_utterance_metadata_for_command(
        command_directory, command_name, command_metadata, workflow_folderpath, module=None
    ):
        """Helper to extract utterance info from a command's Signature."""
        if module is None:
            module = python_utils.get_module(
                command_metadata.response_generation_module_path,
                command_metadata.workflow_folderpath or workflow_folderpath,
            )
        if not module:
            return

        Signature = getattr(module, "Signature", None)
        if not Signature:
            return

        # A Signature that defines (or inherits) generate_utterances keeps it; one that does
        # not gets the framework default, the standard body (default_generate_utterances).
        plain_utterances = getattr(Signature, "plain_utterances", [])
        template_utterances = getattr(Signature, "template_utterances", [])

        # The command file stays the source path: training fingerprints hash it.
        generated_utterances_module_filepath = command_metadata.response_generation_module_path
        generated_utterances_func_name = (
            "Signature.generate_utterances" if hasattr(Signature, "generate_utterances")
            else DEFAULT_UTTERANCES_FUNC)

        utterance_metadata = UtteranceMetadata(
            workflow_folderpath=command_metadata.workflow_folderpath or workflow_folderpath,
            plain_utterances=plain_utterances,
            template_utterances=template_utterances,
            generated_utterances_module_filepath=generated_utterances_module_filepath,
            generated_utterances_func_name=generated_utterances_func_name,
        )
        command_directory.register_utterance_metadata(command_name, utterance_metadata)

    @field_validator("map_command_2_metadata", mode="before")
    def validate_map_command_2_metadata(cls, v, info: ValidationInfo):
        # This validator is primarily for when CommandDirectory is created from a dict (e.g. JSON)
        # In our `load` method, we construct CommandMetadata objects directly.
        if not isinstance(v, dict):
            raise ValueError("map_command_2_metadata must be a dictionary.")
        
        workflow_folderpath = info.data.get('workflow_folderpath')
        
        for command_key, metadata_val in v.items():
            if not command_key:
                raise ValueError("Command key cannot be an empty string.")
            if isinstance(metadata_val, dict):
                # Ensure workflow_folderpath is present if CommandMetadata is being created from a dict
                if 'workflow_folderpath' not in metadata_val and workflow_folderpath:
                     metadata_val['workflow_folderpath'] = workflow_folderpath
                v[command_key] = CommandMetadata(**metadata_val)
            elif not isinstance(metadata_val, CommandMetadata):
                raise ValueError(f"Invalid metadata for command key '{command_key}'")
        return v

    @field_validator("map_command_2_utterance_metadata", mode="before")
    def parse_map_command_2_utterances(
        cls, v: dict[str, UtteranceMetadata], info: ValidationInfo
    ):
        # Similar to above, mainly for dict -> model conversion
        if not isinstance(v, dict):
            raise ValueError("map_command_2_utterance_metadata must be a dictionary.")
            
        workflow_folderpath = info.data.get('workflow_folderpath')
        for key, value in v.items():
            if isinstance(value, dict):
                if 'workflow_folderpath' not in value and workflow_folderpath:
                    value['workflow_folderpath'] = workflow_folderpath
                v[key] = UtteranceMetadata(**value)
            elif not isinstance(v[key], UtteranceMetadata): # Check v[key]
                raise ValueError(f"Invalid value for type metadata '{key}'")
        return v

    @staticmethod
    def _register_core_commands(command_directory: "CommandDirectory") -> None:
        """Dynamically discover and register core commands from the command_metadata_extraction workflow.

        Rules:
        1.  Core commands come from *all* python modules inside the internal
            ``command_metadata_extraction/_commands`` tree (recursively).
        """
        import fastworkflow  # inline: matches other internal-workflow guards in this module

        if os.path.abspath(command_directory.workflow_folderpath) == os.path.abspath(
                fastworkflow.get_internal_workflow_path("plan_metadata_extraction")):
            command_directory.core_command_names.clear()
            return

        # Clear any previously populated list first
        command_directory.core_command_names.clear()

        import fastworkflow
        internal_wf_path = fastworkflow.get_internal_workflow_path("command_metadata_extraction")
        internal_cmd_root = os.path.join(internal_wf_path, "_commands")
        
        # Get the discovered commands using the cached function
        discovered_commands = CommandDirectory._discover_core_commands(internal_wf_path)
        
        # Store sorted list
        command_directory.core_command_names.extend(sorted(discovered_commands))

        # ------------------------------------------------------------
        # Register each core command's metadata & utterances (unless
        # already registered in this directory build).
        # ------------------------------------------------------------
        for qualified_cmd in command_directory.core_command_names:
            # Determine the actual file path inside the internal workflow
            # irrespective of current workflow.
            if "/" in qualified_cmd:
                ctx, cmd = qualified_cmd.split("/", 1)
                module_rel_path = os.path.join(ctx, f"{cmd}.py")
            else:
                module_rel_path = f"{qualified_cmd}.py"

            module_path = os.path.join(internal_cmd_root, module_rel_path)

            # Try to inspect module for Signature presence
            module = python_utils.get_module(module_path, internal_wf_path)
            sig_exists = hasattr(module, "Signature") if module else False
            input_exists = bool(sig_exists and hasattr(module.Signature, "Input"))

            metadata = CommandMetadata(
                workflow_folderpath=internal_wf_path,
                parameter_extraction_signature_module_path=(module_path if sig_exists else None),
                input_for_param_extraction_class=("Signature" if sig_exists else None),
                command_parameters_class=("Signature.Input" if input_exists else None),
                response_generation_module_path=module_path,
                response_generation_class_name="ResponseGenerator",
            )
            command_directory.register_command_metadata(qualified_cmd, metadata)

        # Core contexts' callback classes (e.g. `occupiable = False` on IntentDetection)
        # are registered with their commands; the scan above does not reach them.
        for context_name in sorted({name.split("/")[0] for name in discovered_commands if "/" in name}):
            CommandDirectory._register_context_class(
                command_directory, context_name,
                Path(internal_cmd_root) / context_name, internal_wf_path)

    @staticmethod
    @lru_cache(maxsize=1)  # Only need to cache one result since core commands are fixed
    def _discover_core_commands(internal_wf_path: str) -> set[str]:
        """
        Discover core commands from the internal workflow.
        This is cached to avoid repeated filesystem scanning.
        
        Args:
            internal_wf_path: Path to the internal workflow
            
        Returns:
            set[str]: Set of discovered core command names
        """
        internal_cmd_root = os.path.join(internal_wf_path, "_commands")
        
        # --------------------------------------------------------
        # Deep-scan the internal workflow's _commands
        # directory to build the union of all command names.
        # --------------------------------------------------------
        discovered_commands: set[str] = set()
        for root, _dirs, files in os.walk(internal_cmd_root):
            if root.endswith('ErrorCorrection'):
                continue

            for filename in files:
                if (
                    filename.endswith(".py")
                    and not filename.startswith("_")
                    and filename != "__init__.py"
                ):
                    stem = os.path.splitext(filename)[0]

                    # Build qualified name relative to _commands root.
                    rel_path = os.path.relpath(os.path.join(root, filename), internal_cmd_root)
                    parts = os.path.splitext(rel_path)[0].split(os.sep)
                    qualified_name = "/".join(parts) if len(parts) > 1 else parts[0]
                    discovered_commands.add(qualified_name)

        if not discovered_commands:
            raise RuntimeError("No core commands discovered in internal workflow: {internal_wf_path}")

        return discovered_commands

    @classmethod
    def _register_context_class(cls, command_directory: "CommandDirectory", context_name: str, context_dir: Path, workflow_folderpath: str):
        """Register a context callback implementation if a file named '_<ContextName>.py' exists.

        An extensionless '_<ContextName>' (a spelling this loader once accepted) fails the load and names the
        replacement."""

        context_file_path = context_dir / f"_{context_name}.py"
        _check_context_class_file_name(context_name, context_dir)
        if not context_file_path.is_file():
            if (context_dir / f"_{context_name}").is_file():
                raise ValueError(
                    f"{context_dir / f'_{context_name}'}: the context class file is _{context_name}.py: "
                    f"rename _{context_name} to _{context_name}.py")
            context_file_path = None

        if context_file_path is None:
            # No context callback implementation available – silently ignore
            return

        metadata = ContextMetadata(
            workflow_folderpath=workflow_folderpath,
            context_module_path=str(context_file_path),
            context_class="Context",
        )
        command_directory.register_context_metadata(context_name, metadata)

    # ---------------------- Context Metadata helpers ---------------------- #

    def register_context_metadata(self, context_name: str, metadata: "ContextMetadata"):
        if not (context_name and metadata):
            raise ValueError("context_name and metadata are required")
        self.map_context_2_metadata[context_name] = metadata

    def get_context_metadata(self, context_name: str) -> "ContextMetadata":
        if context_name not in self.map_context_2_metadata:
            raise KeyError(f"Context '{context_name}' not found in map_context_2_metadata")
        return self.map_context_2_metadata[context_name]

    # ---------------------- Validators for context metadata ---------------------- #

    @field_validator("map_context_2_metadata", mode="before")
    def validate_map_context_2_metadata(cls, v, info: ValidationInfo):
        if not isinstance(v, dict):
            raise ValueError("map_context_2_metadata must be a dictionary.")

        workflow_folderpath = info.data.get('workflow_folderpath')

        for context_key, metadata_val in v.items():
            if not context_key:
                raise ValueError("Context key cannot be empty.")
            if isinstance(metadata_val, dict):
                if 'workflow_folderpath' not in metadata_val and workflow_folderpath:
                    metadata_val['workflow_folderpath'] = workflow_folderpath
                v[context_key] = ContextMetadata(**metadata_val)
            elif not isinstance(metadata_val, ContextMetadata):
                raise ValueError(f"Invalid metadata for context key '{context_key}'")
        return v

    # ------------------------------------------------------------------
    # Lazy hydration helpers
    # ------------------------------------------------------------------

    def ensure_command_hydrated(self, command_name: str):
        """Populate Signature-related metadata & utterances on demand.

        This method is idempotent and inexpensive after the first call
        thanks to the @lru_cache on the internal helper.
        """

        command_metadata = self.map_command_2_metadata.get(command_name)
        if command_metadata is None:
            raise KeyError(f"Command '{command_name}' is not registered. command_metadata is missing")

        utterance_metadata = self.map_command_2_utterance_metadata.get(command_name)
        if utterance_metadata is None:
            utterance_metadata = UtteranceMetadata(
                workflow_folderpath = '',
                plain_utterances = [],
                template_utterances = [],
                generated_utterances_module_filepath = 'dummy',
                generated_utterances_func_name = 'dummy'               
            )

        # If already hydrated in this process we are done. (When a directory is
        # deserialized from `command_directory.json` in a fresh process, the
        # string fields on `command_metadata` are already populated from the
        # snapshot, so we still ensure the command's in-memory `from_` edges
        # have been resolved once for its `Signature` class before returning.)
        if command_metadata.response_generation_class_name is not None and (
            command_metadata.input_for_param_extraction_class is not None and \
            utterance_metadata.plain_utterances
        ):
            _ensure_workflow_field_sources_resolved(command_metadata, self, command_name)
            return

        _lazy_hydrate_metadata(
            command_metadata.response_generation_module_path,
            command_metadata,
            self,
            command_name,
        )


# ------------------------------------------------------
# Module-level helper with its own cache so that repeated
# hydration calls across directories are cheap.
# ------------------------------------------------------

# Guard against re-entrant `resolve_field_sources` calls when `resolve_field_sources`
# loads `CommandContextModel`, which builds `CommandDirectory` and hydrates commands.
_RESOLVING_FIELD_SOURCES_WORKFLOWS: set[str] = set()
# Cache of `(workflow_abs_path, command_name) -> Signature class` already resolved
# and logged in this process. Keying on the `Signature` class object ensures that
# resolution and error logging happen at most once per `(workflow, command, Signature)`
# while a reloaded module (producing a new `Signature` class object) is resolved again.
_RESOLVED_COMMAND_SIGNATURES: dict[tuple[str, str], type] = {}
# ``(workflow_abs_path, command_name)`` already logged via :func:`_handle_command_field_source_errors`.
_LOGGED_FIELD_SOURCE_COMMAND_ERRORS: set[tuple[str, str]] = set()
_INTERNAL_CME_WORKFLOW_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "_workflows", "command_metadata_extraction")
)


def clear_resolved_command_signatures_cache() -> None:
    """Clear the per-command ``(workflow, command) -> Signature`` resolution cache."""
    _RESOLVED_COMMAND_SIGNATURES.clear()
    _LOGGED_FIELD_SOURCE_COMMAND_ERRORS.clear()


def _is_subject_workflow_command(
    command_name: str, metadata: CommandMetadata, directory: CommandDirectory
) -> bool:
    """True when *command_name* belongs to a subject workflow rather than the internal CME workflow."""
    if command_name in directory.core_command_names:
        return False
    if os.path.abspath(metadata.workflow_folderpath or "") == _INTERNAL_CME_WORKFLOW_PATH:
        return False
    if os.path.abspath(directory.workflow_folderpath or "") == _INTERNAL_CME_WORKFLOW_PATH:
        return False
    return True


def _handle_command_field_source_errors(
    command_name: str,
    errs: list[str],
    *,
    workflow_folderpath: str | None = None,
) -> bool:
    """Log a command's load errors (``command_load_errors``) once per process, keyed by workflow folder and
    command name. Logging is all it does: the errors themselves make the command fail to load. Returns True."""
    if errs:
        wf_abs = os.path.abspath(workflow_folderpath) if workflow_folderpath else ""
        log_key = (wf_abs, command_name)
        if log_key not in _LOGGED_FIELD_SOURCE_COMMAND_ERRORS:
            _LOGGED_FIELD_SOURCE_COMMAND_ERRORS.add(log_key)
            logger.error(f"command {command_name!r}: does not load: " + "; ".join(errs))
    return True


def command_load_errors(directory: CommandDirectory, command_name: str) -> list[str]:
    """Why *command_name* does not load, from the checks that read its imported classes
    (``planning_commands.resolve_field_sources``): a ``from_command_output`` that does not resolve, a call
    cycle, a facts field or a ``ui`` hint that is not valid, a retired or hand-written declaration, a
    ``@descends_to`` target that is no context, a ``__call__`` that does not take three arguments, an
    ``approval_template`` that cannot be filled, each as ``<file>: <message>``. Empty for a command of the
    framework's own workflows and for one with none. A file that cannot be read at all is not here: it is not
    registered (``CommandDirectory.load_errors``)."""
    metadata = directory.map_command_2_metadata.get(command_name)
    if metadata is None or not _is_subject_workflow_command(command_name, metadata, directory):
        return []
    wf_folder = directory.workflow_folderpath or metadata.workflow_folderpath
    if not wf_folder:
        return []
    from fastworkflow.planning import planning_commands  # circular import: planning_commands -> command_context_model -> command_directory

    wf_abs = os.path.abspath(wf_folder)
    if wf_abs in _RESOLVING_FIELD_SOURCES_WORKFLOWS:
        return []
    return planning_commands.resolve_field_sources(wf_abs).errors_for(command_name)


def _ensure_workflow_field_sources_resolved(
    metadata: CommandMetadata,
    directory: CommandDirectory,
    command_name: str,
    signature_cls: Optional[type] = None,
) -> bool:
    """Ensure ``resolve_field_sources`` has run for the workflow and the command's
    ``Signature`` has its ``available_from`` populated (including when ``CommandDirectory``
    was loaded from a persisted JSON snapshot in a new process).

    Cached per ``(workflow_folder, command_name, Signature class identity)`` so each
    command's resolution and error logging runs at most once per ``Signature`` class.
    Returns ``True`` when hydration should proceed, ``False`` if the command should be skipped.
    """
    if not _is_subject_workflow_command(command_name, metadata, directory):
        return True
    wf_folder = directory.workflow_folderpath or metadata.workflow_folderpath
    if not wf_folder:
        return True
    wf_abs = os.path.abspath(wf_folder)
    if wf_abs in _RESOLVING_FIELD_SOURCES_WORKFLOWS:
        return True

    module_path = metadata.response_generation_module_path
    if signature_cls is None:
        module = python_utils.get_module(module_path, metadata.workflow_folderpath or wf_folder)
        signature_cls = getattr(module, "Signature", None) if module else None
    if signature_cls is None:
        return True

    cache_key = (wf_abs, command_name)
    if _RESOLVED_COMMAND_SIGNATURES.get(cache_key) is signature_cls:
        return True

    from fastworkflow.planning import planning_commands  # circular import: planning_commands -> command_context_model -> command_directory

    _RESOLVING_FIELD_SOURCES_WORKFLOWS.add(wf_abs)
    try:
        resolution = planning_commands.resolve_field_sources(wf_abs)
    finally:
        _RESOLVING_FIELD_SOURCES_WORKFLOWS.discard(wf_abs)

    cmd_problems = planning_commands.resolve_signature_field_sources(
        command_name, signature_cls, module_path, resolution
    )
    _RESOLVED_COMMAND_SIGNATURES[cache_key] = signature_cls

    errs = resolution.errors_for(command_name) or [
        p.format() for p in cmd_problems if p.severity == "error"
    ]
    # Logged here; the errors make every class fetch of the command fail (``command_load_errors``).
    _handle_command_field_source_errors(command_name, errs, workflow_folderpath=wf_abs)
    return True


# We purposely do *not* cache by arguments because `metadata`/`directory` are
# unhashable Pydantic objects.  The helper itself safeguards against running
# twice for the same command by mutating `metadata` in-place.
def _lazy_hydrate_metadata(module_path: str, metadata: CommandMetadata, directory: CommandDirectory, command_name: str):
    """Load the command's module once and enrich its metadata object.

    Note: `metadata` is mutated in place which is fine because the object
    is unique within its `CommandDirectory` instance and, through the
    outer LRU cache, this function will execute at most once per command.
    """

    module = python_utils.get_module(module_path, metadata.workflow_folderpath)
    if not module:
        return  # cannot hydrate

    signature_cls = getattr(module, "Signature", None)
    if not signature_cls:
        return

    if not _ensure_workflow_field_sources_resolved(
        metadata, directory, command_name, signature_cls=signature_cls
    ):
        return

    metadata.parameter_extraction_signature_module_path = module_path
    metadata.input_for_param_extraction_class = "Signature"
    if hasattr(signature_cls, "Input"):
        metadata.command_parameters_class = "Signature.Input"

    # Utterance metadata (if not already registered)
    if command_name not in directory.map_command_2_utterance_metadata:
        CommandDirectory._populate_utterance_metadata_for_command(
            directory,
            command_name,
            metadata,
            metadata.workflow_folderpath or directory.workflow_folderpath,
            module,
        )

# Files (beyond *.py) that the command/routing snapshots derive from. Changes
# to these must also invalidate the persisted caches.
_CONTEXT_MODEL_BASENAMES = ("context_inheritance_model.json",)


#: The load error for a workflow that still has a ``_skills/`` folder (see ``check_no_planning_commands_folder``).
PLANNING_COMMANDS_FOLDER_RETIRED = (
    "_skills/ is no longer read; move these files into _commands/ (first make sure any script "
    "that regenerates _commands/ preserves them; see docs/migrating_skills_to_commands.md)"
)


def check_no_planning_commands_folder(workflow_folderpath: str) -> None:
    """Raise ValueError when the workflow folder still has a retired ``_skills/`` folder.

    Planning commands are read from ``_commands/`` (as are execution commands), so a workflow
    that keeps its planning commands in ``_skills/`` must move them before it loads.
    """
    if os.path.isdir(os.path.join(workflow_folderpath, "_skills")):
        raise ValueError(PLANNING_COMMANDS_FOLDER_RETIRED)


#: The one home of a workflow's helper modules: the package ``_commands/_shared/``, which the framework never reads as
#: commands, contexts, settings or training inputs; command files import it (``from ._shared.x import Y`` from a root
#: command, ``from .._shared.x import Y`` from a context folder).
SHARED_HELPERS_FOLDER = "_shared"


def helper_file_problems(workflow_folderpath: str) -> list[str]:
    """The underscore files and folders of ``_commands/`` the framework does not read, one ``<path>: <reason>`` each
    (bd fix-qjne.33): each is a workflow-wide load problem (``command_preview.load_problems``).

    ``_commands/`` holds command files, context folders (command files and the folder's own ``_<Context>.py``),
    ``_workflow.py``, its ``.json`` files and the helper package ``_shared/``; ``__init__.py`` and ``__pycache__`` are
    allowed everywhere. Any other ``.py`` file whose name starts with ``_``, directly in ``_commands/`` or in a context
    folder, and any other folder of ``_commands/`` whose name starts with ``_``, is reported: an underscore no longer
    means "ignored", so a misspelt settings or context class file is caught (with a did-you-mean when one name is
    close) and a helper module is told to move to ``_commands/_shared/``. ``_workflow.py`` in a context folder and
    ``_split.py`` are reported by ``workflow_agent.split_file_problems`` with their own messages.
    """
    root = Path(workflow_folderpath) / "_commands"
    if not root.is_dir():
        return []
    entries = sorted(root.iterdir())
    contexts = [entry.name for entry in entries
                if entry.is_dir() and not entry.name.startswith(("_", ".")) and entry.name != "__pycache__"]
    problems = []
    for entry in entries:
        name = entry.name
        if entry.is_dir():
            if name in (SHARED_HELPERS_FOLDER, "__pycache__") or name.startswith("."):
                continue
            if name.startswith("_"):
                problems.append(f"_commands/{name}/: a folder of _commands/ whose name starts with an underscore is "
                                f"no context and is never read; helper modules belong in the package "
                                f"_commands/{SHARED_HELPERS_FOLDER}/: move its modules there")
                continue
            for path in sorted(entry.glob("_*.py")):
                # a file that defines class Context is reported by the loader (_check_context_class_file_name)
                if path.name in ("__init__.py", f"_{name}.py", "_workflow.py", "_split.py") \
                        or _defines_context_class(path):
                    continue
                problems.append(_stray_underscore_file(f"_commands/{name}/{path.name}", "..",
                                                       {f"_{name}.py": f"_commands/{name}/_{name}.py"}))
        elif name.startswith("_") and name.endswith(".py") and name not in ("__init__.py", "_workflow.py",
                                                                              "_split.py"):
            close = {"_workflow.py": "_commands/_workflow.py",
                     **{f"_{context}.py": f"_commands/{context}/_{context}.py" for context in contexts}}
            problems.append(_stray_underscore_file(f"_commands/{name}", ".", close))
    return problems


def _stray_underscore_file(label: str, dots: str, expected: dict[str, str]) -> str:
    """The problem line of an underscore ``.py`` file the framework does not read: a did-you-mean when its name is
    close to a file the framework reads there (*expected*: file name -> where it belongs), and where a helper goes."""
    import difflib

    name = label.rsplit("/", 1)[-1]
    close = difflib.get_close_matches(name, list(expected), n=1, cutoff=0.85)
    guess = f" Did you mean {expected[close[0]]}?" if close else ""
    module = name[:-3].lstrip("_") or "helpers"
    return (f"{label}: is not read: in _commands/ a file whose name starts with an underscore is no longer ignored, "
            f"and only _workflow.py and a context folder's own _<Context>.py are read.{guess} A helper module belongs "
            f"in the package _commands/{SHARED_HELPERS_FOLDER}/: move it to _commands/{SHARED_HELPERS_FOLDER}/"
            f"{module}.py and import it as from {dots}{SHARED_HELPERS_FOLDER}.{module} import ...")


class CommandLoadError(ValueError):
    """Raised when a class of a command that does not load is asked for; the message is the command's load errors
    (``command_load_errors``), each naming the file and what to write instead."""


class CommandFileError(ValueError):
    """A command file the loader cannot classify (it defines both generator classes): that command's load error."""


def raised_in_helper(error: BaseException, command_file: str, workflow_folder: str) -> str:
    """`` (in <helper file>, line N)`` when *error*, raised while a command file was imported, comes from a helper
    module of the workflow (a file of ``_commands/`` other than the command's own), else "": the command's line then
    names the file to fix as well as the command it breaks."""
    import traceback

    while isinstance(error, ImportError) and error.__cause__ is not None:
        error = error.__cause__     # python_utils.get_module wraps the import's own error
    commands = os.path.join(os.path.abspath(workflow_folder), "_commands") + os.sep
    own = os.path.abspath(command_file)
    frames = [frame for frame in traceback.extract_tb(error.__traceback__)
              if os.path.abspath(frame.filename).startswith(commands) and os.path.abspath(frame.filename) != own]
    if isinstance(error, SyntaxError) and error.filename and os.path.abspath(error.filename).startswith(commands) \
            and os.path.abspath(error.filename) != own:
        return f" (in {os.path.relpath(error.filename, workflow_folder)}, line {error.lineno})"
    if not frames:
        return ""
    frame = frames[-1]
    return f" (in {os.path.relpath(frame.filename, workflow_folder)}, line {frame.lineno})"


def _check_context_class_file_name(context_name: str, context_dir: Path) -> None:
    """Raise ValueError when a ``_*.py`` of the context folder other than ``_<Context>.py`` defines a top-level
    ``class Context``: the loader reads the callbacks from ``_<Context>.py`` only, so they would be silently ignored."""
    for path in sorted(context_dir.glob("_*.py")):
        # _workflow.py is the workflow's own settings file: one in a context folder is reported, as never read, by
        # workflow_agent.split_file_problems
        if path.name in (f"_{context_name}.py", "_workflow.py"):
            continue
        if _defines_context_class(path):
            fix = (f"rename {path.name} to _{context_name}.py" if not (context_dir / f"_{context_name}.py").is_file()
                   else f"move the class to _{context_name}.py")
            raise ValueError(f"{path}: defines class Context, but the context class file of {context_name} is "
                             f"_{context_name}.py: {fix}")


def _defines_context_class(path: Path) -> bool:
    """Whether the file at *path* defines a top-level ``class Context`` (a file that cannot be read or parsed does not:
    it fails where it is imported)."""
    try:
        source = path.read_text(encoding="utf-8")
        if "class Context" not in source:
            return False
        tree = ast.parse(source, filename=str(path))
    except (OSError, SyntaxError, ValueError):
        return False
    return any(isinstance(node, ast.ClassDef) and node.name == "Context" for node in tree.body)


def _command_kind_from_ast(command_file_path: Path) -> tuple[str, bool]:
    """Return ``(response_generation_class_name, is_planning)`` without importing the module."""
    try:
        source = command_file_path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(command_file_path))
    except SyntaxError:
        raise
    except OSError as exc:
        raise ValueError(f"{command_file_path}: could not read command source ({exc})") from exc
    has_response_generator = False
    has_plan_generator = False
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            if node.name == "ResponseGenerator":
                has_response_generator = True
            elif node.name == "PlanGenerator":
                has_plan_generator = True
    if has_response_generator and has_plan_generator:
        raise CommandFileError(
            f"{command_file_path}: defines both ResponseGenerator and PlanGenerator; a command file holds one of "
            "them: ResponseGenerator for a command that runs code, PlanGenerator for a planning command. Move the "
            "other command to its own file"
        )
    if has_plan_generator:
        return "PlanGenerator", True
    return "ResponseGenerator", False


def _top_level_names(tree: ast.Module) -> set[str]:
    """The names a module's top level binds: classes, functions, assignments and imports (``as`` names included)."""
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update((alias.asname or alias.name).split(".")[0] for alias in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update(target.id for target in targets if isinstance(target, ast.Name))
    return names


@functools.lru_cache(maxsize=None)
def _file_top_level_names(path: str, _mtime_ns: int) -> frozenset[str]:
    try:
        return frozenset(_top_level_names(ast.parse(Path(path).read_text(encoding="utf-8"))))
    except (OSError, SyntaxError, ValueError):
        return frozenset()


def declares_signature(command_file_path: str | Path) -> bool:
    """Whether a command file declares ``Signature`` (its top level binds the name: a class, or an import of one).

    The one test of an action-only command (bd fix-apjz.9): a command with no ``Signature``, such as a startup command,
    runs only as an action (the workflow's startup action, a direct action). No typed name, fuzzy name, local model or
    remote model reaches it, and no listing names it; its name does not matter. Read from the file, no import; cached
    per file and modification time. A file that cannot be read declares nothing."""
    path = str(command_file_path)
    try:
        mtime = os.stat(path).st_mtime_ns
    except OSError:
        return False
    return "Signature" in _file_top_level_names(path, mtime)


def missing_generator_problem(command_file_path: str | Path) -> str | None:
    """Why a command file has no generator class, or None.

    One rule, the one the scanner classifies by (``_command_kind_from_ast``): a generator is a CLASS named
    ``ResponseGenerator`` or ``PlanGenerator`` defined at the file's top level (it may inherit a shared base from a
    module of ``_commands/_shared/``). So a file whose generator name is imported or assigned is refused, naming the
    class to define; a file that binds ``Signature`` and defines no generator has a misspelled or missing one; and a
    file that defines neither a ``Signature`` nor a generator is no command at all: a helper module, which lives in
    ``_commands/_shared/`` (the scanner never reads that folder). A file with a generator and no ``Signature`` (a
    startup command) is a command and is not reported.
    """
    try:
        tree = ast.parse(Path(command_file_path).read_text(encoding="utf-8"))
    except (OSError, SyntaxError, ValueError):
        return None     # a file that cannot be read or parsed fails its own load, with its own message
    names = _top_level_names(tree)
    defined = {node.name for node in tree.body if isinstance(node, ast.ClassDef)}
    generators = {"ResponseGenerator", "PlanGenerator"}
    if defined & generators:
        return None
    if bound := sorted((names & generators) - defined):
        return (f"{command_file_path}: {bound[0]} is imported or assigned here, not defined; a generator is a class "
                f"defined in the command file: write class {bound[0]}(<base>) in this file (it may inherit a base "
                "from a helper module of _commands/_shared/)")
    if "Signature" not in names:
        return (f"{command_file_path}: is not a command: it defines no Signature, ResponseGenerator or "
                "PlanGenerator. A file in _commands/ is a command; a helper module belongs in the package "
                f"_commands/_shared/ (_commands/_shared/{Path(command_file_path).name})")
    return (f"{command_file_path}: has a Signature but no ResponseGenerator or PlanGenerator; the class name must be "
            "exactly ResponseGenerator (a command that runs code) or PlanGenerator (a planning command); "
            "check its spelling and case")


def _command_source_roots(workflow_folderpath: str) -> list[Path]:
    """Return every `_commands` root that contributes to this workflow.

    This includes the workflow's own `_commands` plus any base-workflow
    `_commands` declared via `workflow_inheritance_model.json`. Base resolution
    is best-effort: if it fails we still fingerprint the local sources so the
    cache never silently goes stale.
    """
    roots: list[Path] = []
    with contextlib.suppress(Exception):
        inheritance_model = WorkflowInheritanceModel.load(workflow_folderpath)
        roots.extend(
            Path(base_path) / "_commands"
            for base_path in inheritance_model.resolve_base_paths(workflow_folderpath)
        )
    roots.append(Path(workflow_folderpath) / "_commands")
    # Core commands and their context classes are registered into every directory
    # (see `_register_core_commands`), so their source must invalidate it too.
    import fastworkflow
    roots.append(Path(fastworkflow.get_internal_workflow_path("command_metadata_extraction")) / "_commands")
    return roots


#: Per resolved workflow folder, what its last full scan saw: the stat of ``workflow_inheritance_model.json`` (which
#: decides the roots), the stat of every directory under the roots (and of each root), and the source files found.
#: The fingerprint is read on every direct action (``intent_loop.choose`` -> ``planning_commands.resolve_field_sources``);
#: while no directory changed (nothing was added, removed or renamed) the files are only stat-ed again.
_SOURCE_SCANS: dict[str, tuple] = {}


def _stat_key(path) -> Optional[tuple[int, int]]:
    try:
        st = os.stat(path)
    except OSError:
        return None
    return st.st_mtime_ns, st.st_size


def _source_digest(entries: list[tuple[str, int, int]]) -> str:
    return hashlib.sha256(repr(sorted(entries)).encode("utf-8")).hexdigest()


def compute_commands_source_fingerprint(workflow_folderpath: str) -> str:
    """Fingerprint the command sources a snapshot is derived from.

    The fingerprint is a hash over the *set* of source files and their
    `(size, mtime_ns)`. Unlike a "newest mtime" comparison, this detects
    additions, deletions, renames, and edits uniformly, which is required to
    safely invalidate `command_directory.json` and `routing_definition.json`.

    Always exact, never a time window: when the inheritance file and every directory of the last scan are as they
    were (no file was added, removed or renamed), the files that scan found are stat-ed again and hashed; otherwise
    the roots are scanned again (``_SOURCE_SCANS``). An edit in place changes the file's own stat.
    """
    workflow_folderpath = str(Path(workflow_folderpath).resolve())
    inheritance = os.path.join(workflow_folderpath, "workflow_inheritance_model.json")
    scan = _SOURCE_SCANS.get(workflow_folderpath)
    if scan is not None:
        inheritance_key, directories, files = scan
        if _stat_key(inheritance) == inheritance_key and all(
                _stat_key(directory) == key for directory, key in directories.items()):
            entries: list[tuple[str, int, int]] = []
            for path in files:
                try:
                    st = os.stat(path)
                except OSError:
                    break           # gone without its directory changing (a race): scan again below
                entries.append((path, st.st_size, st.st_mtime_ns))
            else:
                return _source_digest(entries)

    inheritance_key = _stat_key(inheritance)
    directories: dict[str, Optional[tuple[int, int]]] = {}
    entries = []
    for root in _command_source_roots(workflow_folderpath):
        directories[str(root)] = _stat_key(root)      # a root that appears later is a change too
        if not root.is_dir():
            continue
        for p in root.rglob("*"):
            if p.is_dir():
                directories[str(p)] = _stat_key(p)
                continue
            if not p.is_file():
                continue
            if p.suffix == ".py" or p.name in _CONTEXT_MODEL_BASENAMES:
                st = p.stat()
                entries.append((str(p), st.st_size, st.st_mtime_ns))

    # No hierarchy file is fingerprinted: parents come from the live chain, and a leftover
    # context_hierarchy_model.json is ignored (see CommandContextModel.load).

    _SOURCE_SCANS[workflow_folderpath] = (inheritance_key, directories, [path for path, _, _ in entries])
    return _source_digest(entries)


def _recorded_modules_exist(directory: "CommandDirectory") -> bool:
    """Whether every absolute module path the snapshot recorded is still on disk.

    The fingerprint does not cover fastworkflow's own core commands, whose
    absolute paths the snapshot also records, so a moved or reinstalled
    fastworkflow leaves a snapshot that fingerprints clean but points at files
    that are gone. Not folded into the fingerprint because checkpoint namespaces
    are keyed on it too.
    """
    paths = [
        path
        for metadata in directory.map_command_2_metadata.values()
        for path in (
            metadata.parameter_extraction_signature_module_path,
            metadata.response_generation_module_path,
        )
    ]
    paths.extend(
        metadata.context_module_path for metadata in directory.map_context_2_metadata.values()
    )
    return all(os.path.isfile(path) for path in paths if path and os.path.isabs(path))


@lru_cache(maxsize=32)
def get_cached_command_directory(workflow_folderpath: str) -> CommandDirectory:
    """Return a cached CommandDirectory, rebuilding only when sources change.

    The cache key is the *resolved* workflow path so that duplicate relative
    paths share the same entry. We validate the persisted `command_directory.json`
    against a fingerprint of the live `_commands` tree (see
    `compute_commands_source_fingerprint`). When the fingerprint matches we
    deserialize the JSON directly; otherwise we rebuild via
    `CommandDirectory.load()` and persist the new snapshot. The fingerprint is
    deletion- and rename-aware, unlike a plain mtime comparison.
    """

    workflow_folderpath = str(Path(workflow_folderpath).resolve())
    # Checked before the fast path: a snapshot written before a _skills/ folder appeared still matches the fingerprint
    check_no_planning_commands_folder(workflow_folderpath)

    cache_folder = Path(CommandDirectory.get_commandinfo_folderpath(workflow_folderpath))
    cache_file = cache_folder / "command_directory.json"

    current_fingerprint = compute_commands_source_fingerprint(workflow_folderpath)

    # Fast-path: load JSON if its stamped fingerprint matches the live sources
    if cache_file.exists():
        with contextlib.suppress(Exception):
            cached = CommandDirectory.model_validate_json(cache_file.read_text())
            if cached.source_fingerprint == current_fingerprint and _recorded_modules_exist(cached):
                return cached

    # (Re)build and persist (save() re-stamps the fingerprint)
    directory = CommandDirectory.load(workflow_folderpath)
    directory.save()
    return directory
