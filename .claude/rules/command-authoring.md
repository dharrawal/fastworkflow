---
paths:
  - "**/_commands/**"
  - "**/context_inheritance_model.json"
---

# Authoring fastWorkflow Commands

## Workflow Directory Structure

```
my_workflow/
├── application/                     # Your app code (untouched)
├── _commands/                       # Command implementations (generated + edited)
│   ├── context_inheritance_model.json  # Context/command hierarchy
│   └── <command_name>.py            # Single-file command (preferred)
├── ___command_info/                 # Generated at train-time (gitignore)
├── ___workflow_contexts/            # Session state at run-time (gitignore)
└── ___convo_info/                   # Conversation logs (gitignore)
```

Add to `.gitignore`: `___workflow_contexts`, `___command_info`, `___convo_info`

## Command File Structure (Single-File Pattern)

New commands use a **single `.py` file** in `_commands/`. Legacy commands with subdirectories (`parameter_extraction/`, `response_generation/`, `utterances/`) are being migrated to this pattern. When you encounter both, use the single file.

```python
# _commands/<command_name>.py

class Signature:
    plain_utterances: list[str] = [...]   # Seed utterances for training
    template_utterances: list[str] = [...] # Optional parameterized patterns

    class Input(BaseModel):               # Pydantic params; use NOT_FOUND default
        model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)
        some_param: Annotated[str, Field(default="NOT_FOUND", description="...", examples=[...])]

    class Output(BaseModel):              # Structured result
        ...

    @staticmethod
    def generate_utterances(...): ...     # Optional: diverse utterance generation

    @staticmethod
    def db_lookup(workflow_snapshot, command) -> list[str]: ...     # Optional
    @staticmethod
    def process_extracted_parameters(...): ...  # Optional post-extraction hook


class ResponseGenerator:
    def __call__(self, workflow_snapshot, command: str, cmd_parameters: Signature.Input) -> fastworkflow.CommandOutput:
        return self._process_command(workflow_snapshot, command, cmd_parameters)

    @staticmethod
    def _process_command(workflow_snapshot, command, cmd_parameters) -> fastworkflow.CommandOutput:
        # Your business logic here
        ...
```

## Context Model (`context_inheritance_model.json`)

Each context entry has exactly two possible keys:
- `/` — list of command names available in that context
- `base` — list of parent context names whose commands are inherited

To add a new command: update the JSON, then create `_commands/<command_name>.py`. `CommandRoutingDefinition` validates that every declared command has an implementation.
