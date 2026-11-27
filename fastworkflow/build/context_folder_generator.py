from __future__ import annotations

"""Utility to generate context folders based on the command context model.

This module provides functionality to create the directory structure required
for context-aware commands, based on the inheritance relationships defined
in the context_inheritance_model.json file.
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

from fastworkflow.core.command_context_model import CommandContextModel, CommandContextModelValidationError
from fastworkflow.utils.logging import logger
from fastworkflow.core.context_utils import get_context_names

__all__ = ["ContextFolderGenerator"]


class ContextFolderGenerator:
    """Generates context folders based on the command context model."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "_commands/context_inheritance_model.json",
        model: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Initialize the context folder generator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "_commands/context_inheritance_model.json"
        """
        self.commands_root = Path(commands_root)
        self.model_path = Path(model_path)
        # ``{context: {"base": [...]}}`` given by the caller (``fastworkflow build`` derives it from the classes and
        # writes no file); without it the model is read from *model_path*
        self._model_data: Optional[Dict[str, Any]] = model

    def load_context_model(self) -> Dict[str, Any]:
        """Load the context model using the CommandContextModel.

        Returns:
            Dict[str, Any]: The parsed context model
        
        Raises:
            Exception: If the context model cannot be loaded
        """
        if self._model_data is not None:
            return self._model_data  # Return cached model if available

        try:
            # If an explicit model file path exists, load it directly
            if self.model_path.is_file():
                with self.model_path.open("r", encoding="utf-8") as f:
                    self._model_data = json.load(f) or {}
                return self._model_data

            # If the model file does not exist, raise to satisfy tests
            if not self.model_path.exists():
                raise FileNotFoundError(f"Context model file '{self.model_path}' not found.")

            # Fallback: derive workflow root and use CommandContextModel
            workflow_root = (
                self.model_path.parent.parent
                if self.model_path.parent.name == "_commands"
                else self.model_path.parent
            )
            model_obj = CommandContextModel.load(workflow_root)
            self._model_data = model_obj._command_contexts
            return self._model_data
        except (CommandContextModelValidationError, Exception) as e:
            logger.error(f"Error loading context model: {e}")
            raise

    def generate_folders(self) -> Dict[str, Path]:
        """Generate context folders based on the model.
        
        Creates a folder for each context in the model,
        except for the global "*" context. Also creates a _<ContextName>.py file
        in each context folder with an empty Context class (no get_parent: the context is top level until its
        author declares parents and writes get_parent together).
        
        Returns:
            Dict[str, Path]: Mapping of context names to their folder paths
        """
        # Load context model
        context_model = self.load_context_model()

        # Ensure root commands directory exists
        self.commands_root.mkdir(exist_ok=True, parents=True)

        # Create context folders
        contexts = get_context_names(context_model)
        if '*' in contexts:
            contexts.remove('*')  # Global context doesn't need a folder

        created_folders = {}

        for context in contexts:
            context_dir = self.commands_root / context
            context_dir.mkdir(exist_ok=True)
            logger.debug(f"Created context folder: {context_dir}")
            created_folders[context] = context_dir

            # Create _<ContextName>.py file if it doesn't exist
            handler_file = context_dir / f"_{context}.py"
            if not handler_file.exists():
                # No get_parent is written: a context is top level until its author declares parents = [...] and
                # writes get_parent together (a get_parent with no parents is a load error, and parents cannot be
                # derived from code because one get_parent may return records of several contexts). The build
                # adds the context's inherits line to this file after creating it.
                handler_content = f'''class Context:
    """The {context} context. It is top level: to place its records inside another context, declare
    parents = [...] (every context a record can sit in) and write get_parent(cls, record) beside it."""
'''

                # Write the file
                handler_file.write_text(handler_content)
                logger.debug(f"Created context handler file: {handler_file}")

        return created_folders 