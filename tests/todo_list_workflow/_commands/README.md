# FastWorkflow Commands

This directory contains FastWorkflow command files generated from the Python application in `./tests/todo_list_workflow/application`.

## Overview

The generated code includes:
- Command files for all public methods and properties in the application
- A command context model mapping classes to commands

## Usage

These commands can be used with FastWorkflow's orchestration and agent frameworks to enable chat-based and programmatic interaction with the application.

## Available Commands

### inheritance Context

No commands in this context.



## Context Model

The `class Context` of each context's `_commands/<Context>/_<Context>.py` declares the contexts whose commands it also has, `inherits = ["BaseClass1", ...]`, and the contexts its records sit in, `parents = ["Container", ...]` (only when it has `get_parent`). A context that declares neither needs no `_<Context>.py`.

Structure example:

```python
class Context:
    inherits = ["BaseClass1"]
```

### Contexts and Commands

#### inheritance Context
No commands in this context.

## Extending

To add new commands:

1. Add new public methods or properties to your application classes
2. Run the FastWorkflow build tool again to regenerate the command files and documentation

## Testing

You can test these commands using FastWorkflow's MCP server and agent interfaces.
