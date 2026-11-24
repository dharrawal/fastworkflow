from __future__ import annotations

"""Utility to regenerate only the inheritance block in the command context model.

This module provides functionality to update the inheritance relationships in
the context_inheritance_model.json file.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Set, Optional, List

from fastworkflow.build.class_analysis_structures import ClassInfo
from fastworkflow.utils.logging import logger
from fastworkflow.utils.context_utils import get_context_names

__all__ = ["InheritanceBlockRegenerator"]


class InheritanceBlockRegenerator:
    """Regenerates only the inheritance block in the command context model."""

    def __init__(
        self, 
        commands_root: str | Path = "_commands", 
        model_path: str | Path = "_commands/context_inheritance_model.json"
    ) -> None:
        """Initialize the inheritance block regenerator.

        Args:
            commands_root: Path to the commands directory, defaults to "_commands"
            model_path: Path to the command context model JSON file, defaults to
                "context_inheritance_model.json"
        """
        self.commands_root = Path(commands_root)
        self.model_path = Path(model_path)
        self._model_data: Optional[Dict[str, Any]] = None

    def scan_contexts(self) -> Set[str]:
        """Scan the commands directory to identify contexts.
        
        Returns:
            Set[str]: Set of context names found in the directory structure
        """
        contexts = set()
        
        if not self.commands_root.exists():
            logger.warning(f"Commands root directory {self.commands_root} does not exist")
            return contexts
        
        try:
            for item in self.commands_root.iterdir():
                if item.is_dir() and not item.name.startswith('_'):
                    contexts.add(item.name)
                    logger.debug(f"Found context: {item.name}")
        except PermissionError:
            logger.error(f"Permission denied when scanning {self.commands_root}")
        except Exception as e:
            logger.error(f"Error scanning contexts: {e}")
        
        return contexts
        
    def load_existing_model(self) -> Dict[str, Any]:
        """Load the existing context model from file.
        
        Returns:
            Dict[str, Any]: The loaded context model, or a default model if the file
                doesn't exist or is invalid
        """
        if self._model_data is not None:
            return self._model_data  # Return cached model if available
        
        try:
            if self.model_path.exists():
                with open(self.model_path, 'r', encoding='utf-8') as f:
                    self._model_data = json.load(f)
                    logger.debug(f"Loaded existing model from {self.model_path}")
                    return self._model_data
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in model file {self.model_path}: {e}")
        except PermissionError:
            logger.error(f"Permission denied when reading {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model file: {e}")
        
        # Return default model if file doesn't exist or is invalid
        self._model_data = {}
        return self._model_data
    
    def regenerate_inheritance(self, classes: Optional[Dict[str, ClassInfo]] = None) -> Dict[str, Any]:
        """Merge the bases found from the classes (or, with none, an entry for each context folder) into the file.

        An entry already in the file keeps every base it lists, in its order; a base found from the classes that it
        does not list is added after them. A context with no entry gets one. Nothing is removed, so a hand-written
        base survives and analysing no classes changes no existing entry. Each entry added or changed is printed.

        Args:
            classes: Optional dictionary mapping class names to ClassInfo objects from AST analysis.

        Returns:
            Dict[str, Any]: The merged context model (flat: context name -> {"base": [...]}).
        """
        if classes:
            all_class_names = set(classes.keys())
            # Only bases that are also in the analysed set
            discovered = {name: [b for b in info.bases if b in all_class_names] for name, info in classes.items()}
        else:
            # Without class info no base can be found: an empty entry for each context folder
            discovered = {context: [] for context in sorted(self.scan_contexts())}

        existing_model = self.load_existing_model()
        inheritance_map: Dict[str, Any] = dict(existing_model)
        for context, bases in discovered.items():
            entry = existing_model.get(context)
            if entry is None:
                inheritance_map[context] = {"base": list(bases)}
                print(f"{self.model_path.name}: {context}: added with bases {list(bases)}")
                continue
            old = list(entry.get("base", [])) if isinstance(entry, dict) else []
            merged = old + [b for b in bases if b not in old]
            if merged != old:
                inheritance_map[context] = {**(entry if isinstance(entry, dict) else {}), "base": merged}
                print(f"{self.model_path.name}: {context}: bases {old} -> {merged}")

        self.write_model(inheritance_map)
        self._model_data = inheritance_map
        return inheritance_map

    def write_model(self, model: Dict[str, Any]) -> None:
        """Write the updated model to file.
        
        Args:
            model: The model data to write (flat structure, no inheritance wrappers)
        """
        try:
            # Ensure parent directories exist
            self.model_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(self.model_path, 'w', encoding='utf-8') as f:
                json.dump(model, f, indent=2)
                logger.debug(f"Updated model written to {self.model_path}")
        except PermissionError:
            logger.error(f"Permission denied when writing to {self.model_path}")
        except Exception as e:
            logger.error(f"Error writing model file: {e}") 