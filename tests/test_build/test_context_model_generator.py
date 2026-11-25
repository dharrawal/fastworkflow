import os
import json
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

import pytest

from fastworkflow.build.context_model_generator import generate_context_model
from fastworkflow.build.class_analysis_structures import ClassInfo, MethodInfo, PropertyInfo

def make_class_with_methods_and_properties():
    class_info = ClassInfo('User', 'application/user.py')
    class_info.methods.append(MethodInfo('GetDetails', [{'name': 'user_id', 'annotation': 'int'}], docstring='Get details.'))
    class_info.methods.append(MethodInfo('Update', [{'name': 'user_id', 'annotation': 'int'}, {'name': 'email', 'annotation': 'str'}], docstring='Update user.'))
    
    # This property will trigger get_properties.py
    email_prop = PropertyInfo('Email', docstring='User email', type_annotation='str')
    class_info.properties.append(email_prop)
    
    # Explicitly make 'Email' settable to trigger set_properties.py
    # In a real scenario, this would be populated by resolve_inherited_properties based on _property_setters
    class_info.all_settable_properties.append(PropertyInfo('Email', docstring='User email to set', type_annotation='str'))

    return {'User': class_info}

def test_generate_context_model():
    classes = make_class_with_methods_and_properties()
    with tempfile.TemporaryDirectory() as tmpdir:
        # generate_context_model returns the model dictionary and writes no model file (bd fix-qjne.37.1)
        model_data = generate_context_model(classes, tmpdir)
        
        expected_file_path = os.path.join(tmpdir, "_commands/context_inheritance_model.json")
        assert not os.path.exists(expected_file_path), "build must not write the removed context model file"

        # Use the returned model_data for content assertions
        # New schema checks - flat structure
        assert 'User' in model_data
        assert 'base' in model_data['User']
        assert model_data['User']['base'] == []

        # No "/" keys should exist anywhere in the JSON
        json_as_str = json.dumps(model_data)
        assert '"/"' not in json_as_str, "Found deprecated '/' key in context model JSON"
        
        # No inheritance should exist
        assert 'inheritance' not in model_data

class TestContextModelGenerator:
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_classes(self):
        """Create mock classes for testing."""
        # Base class
        base_class = ClassInfo(
            name="BaseClass",
            module_path="application/base_class.py",
            docstring="A base class",
            bases=[]
        )
        base_class.methods = [
            MethodInfo(
                name="base_method",
                parameters=[],
                docstring="A base method"
            )
        ]
        classes = {"BaseClass": base_class}
        # Derived class
        derived_class = ClassInfo(
            name="DerivedClass",
            module_path="application/derived_class.py",
            docstring="A derived class",
            bases=["BaseClass"]
        )
        derived_class.methods = [
            MethodInfo(
                name="derived_method",
                parameters=[],
                docstring="A derived method"
            )
        ]
        classes["DerivedClass"] = derived_class

        # Another class with no inheritance
        standalone_class = ClassInfo(
            name="StandaloneClass",
            module_path="application/standalone_class.py",
            docstring="A standalone class",
            bases=[]
        )
        standalone_class.methods = [
            MethodInfo(
                name="standalone_method",
                parameters=[],
                docstring="A standalone method"
            )
        ]
        classes["StandaloneClass"] = standalone_class

        return classes
    
    def test_generate_context_model_new_format(self, temp_dir, mock_classes):
        """Test that generate_context_model creates a model in the new flat format."""
        # Generate the context model
        model = generate_context_model(mock_classes, temp_dir)
        
        # Check the model structure - flat format, no inheritance keys
        assert "DerivedClass" in model
        assert "BaseClass" in model
        assert "StandaloneClass" in model
        
        # Check DerivedClass inherits from BaseClass
        assert model["DerivedClass"] == {"base": ["BaseClass"]}
        
        # Check BaseClass has no base classes
        assert model["BaseClass"] == {"base": []}
        
        # StandaloneClass should be in the model even with no base classes
        assert model["StandaloneClass"] == {"base": []}
        
        # No model file is written: the bases are declared on class Context (bd fix-qjne.37.1)
        assert not os.path.exists(os.path.join(temp_dir, "_commands/context_inheritance_model.json"))
    
    def test_generate_context_model_preserves_existing_entries(self, temp_dir, mock_classes):
        """A hand-written context file is kept byte for byte, also for a context no class gives (bd fix-qjne.37.1)."""
        existing = os.path.join(temp_dir, "_commands", "ExistingClass", "_ExistingClass.py")
        os.makedirs(os.path.dirname(existing))
        with open(existing, "w") as f:
            f.write('class Context:\n    inherits = ["BaseClass"]\n')
        derived_dir = os.path.join(temp_dir, "_commands", "DerivedClass")
        os.makedirs(derived_dir)

        model = generate_context_model(mock_classes, temp_dir)

        # Only the analysed classes are in the model; the hand-written file is untouched
        assert "ExistingClass" not in model
        with open(existing) as f:
            assert f.read() == 'class Context:\n    inherits = ["BaseClass"]\n'
        # A context folder with no _<Context>.py gets one with the derived line
        with open(os.path.join(derived_dir, "_DerivedClass.py")) as f:
            assert f.read() == 'class Context:\n    inherits = ["BaseClass"]\n'
        assert model["DerivedClass"] == {"base": ["BaseClass"]}

    def test_generate_context_model_no_existing_file(self, temp_dir, mock_classes):
        """Without a _commands folder nothing is written at all: there is no model file and no context folder."""
        model = generate_context_model(mock_classes, temp_dir)

        assert not os.path.exists(os.path.join(temp_dir, "_commands"))
        assert "DerivedClass" in model
        assert "BaseClass" in model
        assert "StandaloneClass" in model
