"""
Patch for language_eval/evaluator.py to emit live sample events during evaluation.

When the LMEVAL_LIVE_EVENTS_PATH environment variable is set, each scored sample
is written as a JSONL line to that file in real-time. This allows the API layer
to stream progress to the UI while evaluation is running.

Applied by the Dockerfile after pip install.
"""

import importlib
import json
import os
import sys
from datetime import datetime


def _make_serializable(obj):
    """Convert non-serializable objects to strings."""
    if isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_serializable(item) for item in obj]
    return str(obj)


_patched = False


def patch_evaluator():
    """Monkey-patch language_eval.evaluator.evaluate to emit live events."""
    global _patched
    if _patched:
        return
    _patched = True

    from language_eval import evaluator

    _original_evaluate = evaluator.evaluate

    def _patched_evaluate(*args, **kwargs):
        live_events_path = os.environ.get("LMEVAL_LIVE_EVENTS_PATH")

        if not live_events_path:
            return _original_evaluate(*args, **kwargs)

        # We need to intercept the sample logging inside evaluate().
        # The cleanest way is to wrap the function and use a callback via
        # a thread-local / global that the inner loop can check.
        # But since evaluate() is a single function with the loop inline,
        # we'll use a different approach: patch the task's process_results
        # to emit events after each sample is scored.

        # Get task_dict from args
        if len(args) >= 2:
            task_dict = args[1]
        else:
            task_dict = kwargs.get("task_dict")

        if task_dict is None:
            return _original_evaluate(*args, **kwargs)

        # Count total docs across all tasks for progress tracking
        eval_tasks = task_dict.get("tasks", {}) if isinstance(task_dict, dict) else {}
        total_docs = 0
        task_doc_counts = {}
        for task_name, task in eval_tasks.items():
            try:
                doc_count = len(list(task.test_docs())) if hasattr(task, "test_docs") and task.has_test_docs else 0
                if doc_count == 0 and hasattr(task, "validation_docs") and task.has_validation_docs:
                    doc_count = len(list(task.validation_docs()))
            except Exception:
                doc_count = 0
            task_doc_counts[task_name] = doc_count
            total_docs += doc_count

        # Wrap each task's process_results to emit live events
        events_written = [0]
        original_process_results = {}

        for task_name, task in eval_tasks.items():
            original_process_results[task_name] = task.process_results

            def make_wrapper(t_name, original_fn, t_task):
                def wrapper(doc, results):
                    metrics = original_fn(doc, results)

                    # Write live event
                    try:
                        target = t_task.doc_to_target(doc) if hasattr(t_task, "doc_to_target") else ""
                        # Extract the prompt/question from the doc
                        doc_text = t_task.doc_to_text(doc) if hasattr(t_task, "doc_to_text") else ""

                        event = {
                            "index": events_written[0],
                            "total": total_docs,
                            "task_name": t_name,
                            "doc_id": events_written[0],
                            "prompt": str(doc_text)[:2000] if doc_text else "",
                            "target": str(target)[:1000] if target else "",
                            "response": str(results[0])[:2000] if results else "",
                            "metrics": {k: _make_serializable(v) for k, v in metrics.items()},
                            "timestamp": datetime.now().isoformat(),
                        }

                        with open(live_events_path, "a") as f:
                            f.write(json.dumps(event, default=str, ensure_ascii=False) + "\n")

                        events_written[0] += 1
                    except Exception as e:
                        # Don't let event writing break evaluation
                        print(f"Warning: failed to write live event: {e}", file=sys.stderr)

                    return metrics
                return wrapper

            task.process_results = make_wrapper(task_name, task.process_results, task)

        try:
            result = _original_evaluate(*args, **kwargs)
        finally:
            # Restore original process_results
            for task_name, task in eval_tasks.items():
                if task_name in original_process_results:
                    task.process_results = original_process_results[task_name]

        return result

    evaluator.evaluate = _patched_evaluate


# Auto-apply when imported
patch_evaluator()
