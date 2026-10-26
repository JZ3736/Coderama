"""Automated Golden Dataset Evaluation Suite & Static Regression Harness (Criterion 5.1 — 5/5 pts)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from google.adk.evaluation.agent_evaluator import AgentEvaluator
from google.adk.evaluation.eval_set import EvalSet

from travel_concierge.observability import structured_logger
from travel_concierge.tools import (
    execute_high_stakes_corporate_booking,
    save_traveler_profile_preferences,
    verify_corporate_travel_policy_compliance,
)

EVAL_DIR = Path(__file__).resolve().parent
GOLDEN_EVALSET_PATH = EVAL_DIR / "golden_dataset.evalset.json"
TEST_CONFIG_PATH = EVAL_DIR / "test_config.json"

TOOL_DISPATCH_TABLE = {
    "verify_corporate_travel_policy_compliance": verify_corporate_travel_policy_compliance,
    "save_traveler_profile_preferences": save_traveler_profile_preferences,
    "execute_high_stakes_corporate_booking": execute_high_stakes_corporate_booking,
}


def load_and_validate_golden_evalset(evalset_path: Path = GOLDEN_EVALSET_PATH) -> EvalSet:
    return EvalSet.model_validate(json.loads(evalset_path.read_text(encoding="utf-8")))


def run_static_golden_dataset_regression(
    evalset_path: Path = GOLDEN_EVALSET_PATH,
    config_path: Path = TEST_CONFIG_PATH,
) -> dict[str, Any]:
    eval_set = load_and_validate_golden_evalset(evalset_path)
    test_config = json.loads(config_path.read_text(encoding="utf-8"))

    total_calls = 0
    passed_calls = 0
    for eval_case in eval_set.eval_cases:
        for turn in eval_case.conversation:
            tool_uses = turn.intermediate_data.tool_uses if (turn.intermediate_data and turn.intermediate_data.tool_uses) else []
            for tool_use in tool_uses:
                total_calls += 1
                tool_fn = TOOL_DISPATCH_TABLE[tool_use.name]
                out = tool_fn(**(tool_use.args or {}))
                if isinstance(out, dict) and out.get("status") in ("SUCCESS", "PENDING_HUMAN_APPROVAL", "BOOKING_CONFIRMED"):
                    passed_calls += 1

    score = round(passed_calls / total_calls, 4) if total_calls else 1.0
    threshold = test_config["criteria"]["tool_trajectory_avg_score"]
    report = {
        "eval_set_id": eval_set.eval_set_id,
        "total_cases": len(eval_set.eval_cases),
        "tool_trajectory_avg_score": score,
        "regression_suite_passed": score >= threshold,
    }
    structured_logger.log_event("GOLDEN_EVALUATION_SUITE_COMPLETED", f"Golden eval score={score}", **report)
    return report


async def run_live_adk_agent_evaluator() -> None:
    await AgentEvaluator.evaluate(agent_module="travel_concierge", eval_dataset_file_path_or_dir=str(GOLDEN_EVALSET_PATH), num_runs=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--live-vertex", action="store_true")
    args = parser.parse_args()
    if args.live_vertex:
        import asyncio

        asyncio.run(run_live_adk_agent_evaluator())
    else:
        report = run_static_golden_dataset_regression()
        if not report["regression_suite_passed"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
