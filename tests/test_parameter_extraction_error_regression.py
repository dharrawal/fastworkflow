import pytest
import os
import fastworkflow
from fastworkflow import Workflow
from fastworkflow.utils.signatures import InputForParamExtraction
from fastworkflow._workflows.command_metadata_extraction._commands.wildcard import ParameterExtraction
from fastworkflow._workflows.command_metadata_extraction.parameter_extraction import NOT_FOUND, park
from pydantic import BaseModel, Field
from unittest.mock import patch, MagicMock


class MockCommandParameters(BaseModel):
    user_name: str = Field(description="User name")


class TestParameterExtractionErrorRegression:
    """Test that parameter extraction correctly handles error state and direct parameter input."""
    
    @patch('fastworkflow._workflows.command_metadata_extraction.parameter_extraction.InputForParamExtraction')
    @patch('fastworkflow.RoutingRegistry')
    def test_parameter_extraction_in_error_state(self, mock_routing_registry, mock_input_for_param_extraction):
        """Test that direct parameter values are correctly extracted in error state."""
        
        # Create mock workflows with mocked internals
        cme_workflow = MagicMock()
        app_workflow = MagicMock()
        
        # Set up the context to simulate parameter extraction error state
        cme_workflow.context = {
            "app_workflow": app_workflow,
        }
        
        # The pending parameter loop a failed extraction left, with the missing field
        command_name = "ChatRoom/set_current_user"
        mock_routing_registry.get_definition.return_value.get_command_class.return_value = MockCommandParameters
        park(cme_workflow, command_name, MockCommandParameters.model_construct(user_name=NOT_FOUND))
        
        # Create the parameter extractor with a direct parameter value
        command_value = "nundini"  # This is the direct parameter value
        extractor = ParameterExtraction(cme_workflow, app_workflow, command_name, command_value)
        
        # Mock validate_parameters to return valid (returns 4 values: is_valid, error_msg, suggestions, missing_fields)
        mock_input_instance = MagicMock()
        mock_input_instance.validate_parameters.return_value = (True, "All required parameters are valid.", {}, [])
        mock_input_for_param_extraction.create.return_value = mock_input_instance
        
        # Execute the extraction
        result = extractor.extract()
        
        # Verify the result
        assert result.parameters_are_valid is True
        assert result.cmd_parameters is not None
        assert hasattr(result.cmd_parameters, 'user_name')
        assert result.cmd_parameters.user_name == command_value
    
    def test_direct_parameter_extraction(self):
        """The reply merge (the one the CME uses for the open field) fills the asked parameter from a bare value."""
        for value in ("john_doe", "jane_doe"):
            params = MockCommandParameters.model_construct(user_name=NOT_FOUND)
            result = ParameterExtraction._extract_and_merge_missing_parameters(params, value)
            assert result.user_name == value
