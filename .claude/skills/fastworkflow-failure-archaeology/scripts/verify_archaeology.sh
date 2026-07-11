#!/usr/bin/env bash
# verify_archaeology.sh — read-only sanity check of the volatile facts in
# fastworkflow-failure-archaeology/SKILL.md. Run from the repo root.
# Exit code 0 = all checks passed; nonzero = at least one fact has rotted
# (update the skill, don't ignore it).
#
# Everything here is read-only: git log/show, grep, find, ls, bd show/memories.

set -u
cd "$(git rev-parse --show-toplevel 2>/dev/null || echo .)"
FAIL=0
check() { # check <description> <command...>
  local desc="$1"; shift
  if "$@" >/dev/null 2>&1; then
    echo "OK   $desc"
  else
    echo "ROT  $desc   [$*]"
    FAIL=1
  fi
}
check_empty() { # passes when the command produces NO output
  local desc="$1"; shift
  if [ -z "$("$@" 2>/dev/null)" ]; then
    echo "OK   $desc"
  else
    echo "ROT  $desc (expected no output)   [$*]"
    FAIL=1
  fi
}

echo "== commit hashes cited in the skill =="
for h in f9e2227 a6d0f43 d2e3387 aa751e7 1386f88 fa97b48 79e6986 6880b0a 033132f \
         3fabdb0 afcbe01 aa20c1b 5a63790 cf3eeae b5747df 42e9b3a 920d0de 9031942 \
         55a8668 ce5ada5 ed21d77 ad68d9b 447db70 299739e d9511a5; do
  check "commit $h exists" git cat-file -e "${h}^{commit}"
done

echo "== entry 1/2: payload fix + TurnResult =="
check "merge_artifact_responses_into wired in WEC" \
  grep -q merge_artifact_responses_into fastworkflow/workflow_execution_context.py
check "process_turn exists" \
  grep -q "def process_turn" fastworkflow/workflow_execution_context.py
check "turn.py types exist" grep -q "class TurnOutput" fastworkflow/turn.py
check_empty "v3.0 stores/ package still absent" ls fastworkflow/stores
check "conversation_store.py still present (slated for v3.0 deletion)" \
  ls fastworkflow/run_fastapi_mcp/conversation_store.py

echo "== entry 3: turns engine =="
check "turns.py exists" ls fastworkflow/run_fastapi_mcp/turns.py
check "invoke_agent_stream lock.locked() survivor still present" \
  grep -q "locked()" fastworkflow/run_fastapi_mcp/__main__.py

echo "== entry 4: revert trio =="
check "cruft in merged command_router.py (speedict import)" \
  bash -c "git show f9e2227:fastworkflow/command_router.py | grep -q speedict"
check "command_router.py gone from main" bash -c "! ls fastworkflow/command_router.py"

echo "== entry 5: fix-0hb =="
check "temp-copy training fixture in place" \
  grep -q tmp_path_factory tests/test_train_modern_stack.py

echo "== entry 6: speedict arc =="
check "weakref registry comment block in workflow.py" \
  bash -c "sed -n '14,45p' fastworkflow/workflow.py | grep -q 'no longer resumes workflow'"
check "speedict still used somewhere (fix-7kp not done)" \
  bash -c "grep -rl speedict fastworkflow/ --include='*.py' | grep -q ."

echo "== entry 7: fingerprint =="
check "compute_commands_source_fingerprint exists" \
  grep -q "def compute_commands_source_fingerprint" fastworkflow/command_directory.py

echo "== entries 8/9: bd memories =="
check "beads flakiness memory present" \
  bash -c "bd memories --json 2>/dev/null | grep -q 'beads-flakiness-observed-2026-06-11'"
check "never-commit-or-push memory present" \
  bash -c "bd memories --json 2>/dev/null | grep -q 'never-git-commit-or-push'"

echo "== entry 10: team-private docs (untracked) =="
for f in "docs/Article 1 - The Setup.pdf" "docs/Article 2 - The Failure Taxonomy.pdf" \
         "docs/Article 3 - Mitigations and a Path Forward.pdf" \
         "docs/tau2_retail_reliability_implementation_plan.md" \
         "docs/rsi_harness_agent_report.md"; do
  check "exists: $f" ls "$f"
done
check "self-exchange overlap check on main" \
  bash -c "grep -q 'intersection' fastworkflow/examples/retail_workflow/_commands/exchange_delivered_order_items.py"

echo "== entry 11: tracker drift (informational — statuses may legitimately change) =="
for id in fix-7kp fix-yy1 fix-qtq fix-85g fix-5fv; do
  status=$(bd show "$id" --json 2>/dev/null | python3 -c "import json,sys;d=json.load(sys.stdin);d=d[0] if isinstance(d,list) else d;print(d.get('status'))" 2>/dev/null)
  echo "INFO $id status=$status (skill says: open as of 2026-07-09)"
done

echo "== entry 12: graveyard =="
check_empty "dep-graph modules have no external callers" \
  bash -c "grep -rn 'dependency_manager\|resolve_command_dependencies' fastworkflow/ --include='*.py' | grep -v 'build/dependency_manager.py\|build/command_dependency_resolver.py'"
check_empty "services/ dir gone" ls services
check "probabilistic_response.py absent on main" \
  bash -c "! ls fastworkflow/probabilistic_response.py"
check "probabilistic spec still on main (the trap)" ls docs/probabilistic_response_generation.md
check "majority_vote_predictions still dead (call site commented)" \
  bash -c "grep -q '# predictions = majority_vote_predictions' fastworkflow/_workflows/command_metadata_extraction/intent_detection.py"
check "ambiguous_threshold.json orphans still present" \
  bash -c "find fastworkflow -name ambiguous_threshold.json | grep -q ."
check "colorama still the CLI dep (rich branch still abandoned)" \
  grep -q colorama pyproject.toml
check_empty "cursor branches still unmerged" \
  bash -c "git branch -r --merged main | grep cursor"

echo
if [ "$FAIL" -eq 0 ]; then
  echo "ALL CHECKS PASSED — skill facts current."
else
  echo "SOME FACTS HAVE ROTTED — update SKILL.md entries before trusting them."
fi
exit "$FAIL"
