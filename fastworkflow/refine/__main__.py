import argparse
import os
import sys

import fastworkflow
from fastworkflow.utils.env import add_env_file_arguments
from fastworkflow.utils.logging import logger
from fastworkflow.build.__main__ import init_env
from fastworkflow.build.genai_postprocessor import run_genai_postprocessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Refine a FastWorkflow by enhancing command metadata."
    )
    parser.add_argument('--workflow-folderpath', '-w', required=True, help='Path to the workflow folder to refine')
    add_env_file_arguments(parser)
    return parser.parse_args()


def _validate_workflow_folder(workflow_folderpath: str) -> None:
    if not os.path.isdir(workflow_folderpath):
        print(f"Error: Workflow directory '{workflow_folderpath}' does not exist or is not accessible.")
        sys.exit(1)
    commands_dir = os.path.join(workflow_folderpath, "_commands")
    if not os.path.isdir(commands_dir):
        print(f"Error: Commands directory not found at '{commands_dir}'. Please run 'fastworkflow build' first.")
        sys.exit(1)


def refine_main(args):
    """Entry point for the CLI refine command (invoked from fastworkflow.cli)."""
    try:
        workflow_folderpath = args.workflow_folderpath
        _validate_workflow_folder(workflow_folderpath)
        # The workflow's env files, as `train` loads them: the metadata model (LLM_COMMAND_METADATA_GEN, or the base
        # model it falls back to) and its key are read from them.
        init_env(args)

        # Prepare args-like object for post-processor when needed
        class _Dummy:
            pass
        _dummy = _Dummy()
        _dummy.workflow_folderpath = workflow_folderpath
        logger.info("Running GenAI metadata refinement with LibCST...")
        run_genai_postprocessor(_dummy, classes={}, functions={})

        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    args = parse_args()
    code = refine_main(args)
    sys.exit(code)


if __name__ == "__main__":
    main()
