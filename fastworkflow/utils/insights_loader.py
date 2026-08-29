"""
Utility module for loading workflow-specific insights for knowledge distillation.

This module provides functionality to load planning and execution insights from
workflow directories, enabling the integration of learned patterns and anti-patterns
into agent prompts.
"""

import re
from pathlib import Path
from typing import Optional
from fastworkflow.utils.logging import logger


# The [DR31] provenance marker: the insight id, appended to a written insight
# line as an HTML comment. It renders invisibly in markdown and sits outside
# both numbering regexes (`^(\d+)\.\s` and `^## (\d+)\.`), so a file can be
# renumbered or hand-edited without orphaning the reference — which is exactly
# what an id built on the entry number could not survive (§13.1).
INSIGHT_MARKER_RE = re.compile(r"[ \t]*<!--\s*(ins-[0-9a-f]{12})\s*-->")


def format_insight_marker(insight_id: str) -> str:
    """The trailing marker for one written insight line (§13.1).

    Two leading spaces, matching the design's worked example:
    ``7. Never call update_task before verifying the task exists  <!-- ins-... -->``
    """
    return f"  <!-- {insight_id} -->"


def strip_insight_markers(content: str) -> str:
    """Remove every `[DR31]` marker, and the whitespace ahead of it, from *content*.

    `[DR56]`: there are THREE prompt consumers of an insights corpus, not one —
    the distillation extractor, the live agent's signature docstring
    (`workflow_agent.py:503-509`) and the planner (`:618-619`). An unstripped
    marker would therefore ride into every production agent and planner prompt
    on every turn of any workflow with an `Insights/` directory, distillation
    or not. Stripping here is what keeps every consumer's prompt byte-for-byte
    what it is today; the marker is read only by the ledger tooling below.
    """
    return INSIGHT_MARKER_RE.sub("", content)


def marked_insight_ids(content: str) -> list[str]:
    """The insight ids marked in *content*, in file order (§13.2, file -> ledger).

    This is the direction `text_hash` cannot serve: a hash of the text stops
    resolving the moment a human edits the line, and a human editing a
    distilled rule is precisely when its provenance matters most (§21,
    objection 5).
    """
    return INSIGHT_MARKER_RE.findall(content)


def load_workflow_insights(workflow_folderpath: str, insight_type: str) -> Optional[str]:
    """
    Load workflow-specific insights for knowledge distillation.

    This function looks for insights files in the workflow's Insights directory,
    following the convention: workflow_folderpath/Insights/<workflow_name>/<insight_file>

    Args:
        workflow_folderpath: Absolute path to workflow directory
        insight_type: Type of insights to load. Supported values:
            - 'planning_agent': Planning strategies and patterns
            - 'execution_agent': Execution anti-patterns and rules

    The `[DR31]` insight-id markers are stripped before returning, so every
    consumer — extractor, agent, planner — sees what it sees today `[DR56]`.

    Returns:
        Insights content as string if found, None otherwise

    Example:
        >>> insights = load_workflow_insights('/path/to/my_workflow', 'planning_agent')
        >>> if insights:
        ...     print(f"Loaded {len(insights)} characters of insights")
    """
    # Extract workflow name from folderpath (last component)
    workflow_name = Path(workflow_folderpath).name

    # Build path: workflow_folderpath/Insights/<workflow_name>/
    insights_dir = Path(workflow_folderpath) / "Insights" / workflow_name

    if not insights_dir.exists():
        logger.debug(f"No Insights directory found at {insights_dir}")
        return None

    # Map insight type to filename
    filename_map = {
        "planning_agent": "planning_agent_insights.md",
        "execution_agent": "execution_agent_anti_patterns.md"
    }

    if insight_type not in filename_map:
        logger.warning(f"Unknown insight type: {insight_type}. Supported types: {list(filename_map.keys())}")
        return None

    insight_file = insights_dir / filename_map[insight_type]

    if not insight_file.exists():
        logger.debug(f"Insights file not found: {insight_file}")
        return None

    try:
        content = strip_insight_markers(insight_file.read_text(encoding='utf-8'))
        logger.info(f"Loaded {insight_type} insights from {insight_file} ({len(content)} characters)")
        return content
    except Exception as e:
        logger.error(f"Error reading insights file {insight_file}: {e}")
        return None
