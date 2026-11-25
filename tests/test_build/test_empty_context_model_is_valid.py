import os
import json
import pytest
from pathlib import Path
from fastworkflow.command_context_model import CommandContextModel

def test_empty_context_model_is_valid(tmp_path):
    """Test that an empty context model is valid: a workflow with no context folder and no declarations."""
    (tmp_path / "_commands").mkdir(parents=True, exist_ok=True)
    
    # Load the model
    model = CommandContextModel.load(tmp_path)
    
    # Verify it loads without errors and contains only the injected commands
    assert isinstance(model._command_contexts, dict)
    
    # Check that the model contains exactly the auto-injected contexts/commands
    expected_contexts = {"*", "IntentDetection"}
    assert set(model._command_contexts.keys()) == expected_contexts
    
    expected_global_commands = ["wildcard"]
    assert model._command_contexts["*"]["/"] == expected_global_commands
    
    expected_intent_commands = [
        "IntentDetection/go_up",
        "IntentDetection/reset_context",
        "IntentDetection/what_can_i_do",
        "IntentDetection/what_is_current_context"
    ]
    assert model._command_contexts["IntentDetection"]["/"] == expected_intent_commands 