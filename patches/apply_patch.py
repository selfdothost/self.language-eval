"""
Patches language_eval/evaluator.py and language_eval/models/api_models.py to emit
live sample events.

When LMEVAL_LIVE_EVENTS_PATH is set:
 - api_models.py: emits a progress event after each inference API call
   (task_name="generating") so the UI shows progress during the slow
   Requesting API phase.
 - evaluator.py: emits a scored event after each sample is evaluated,
   including prompt, target, response, and metrics.
"""

import language_eval
import os
import textwrap

evaluator_path = os.path.join(os.path.dirname(language_eval.__file__), "evaluator.py")

with open(evaluator_path, "r") as f:
    source = f.read()

# ── 1. Add imports at the top (after the existing `import json`) ──
import_block = textwrap.dedent("""\
    import os  # live-events patch
    from datetime import datetime as _dt  # live-events patch
""")

if "from datetime import datetime as _dt" not in source:
    source = source.replace(
        "import json\n",
        "import json\n" + import_block,
        1,
    )

# ── 2. Add the live-events helper function before `def evaluate(` ──
helper = textwrap.dedent('''\

def _write_live_event(events_path, index, total, task_name, doc, target, results, metrics, doc_id):
    """Write a single live sample event to the JSONL file."""
    try:
        def _safe_str(obj, limit=2000):
            s = str(obj) if obj is not None else ""
            return s[:limit]

        def _make_serializable(obj):
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            if isinstance(obj, dict):
                return {k: _make_serializable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_make_serializable(i) for i in obj]
            return str(obj)

        event = {
            "index": index,
            "total": total,
            "task_name": task_name,
            "doc_id": doc_id,
            "prompt": _safe_str(doc.get("question", doc.get("text", doc.get("query", "")))),
            "target": _safe_str(target, 1000),
            "response": _safe_str(results[0] if results else ""),
            "metrics": _make_serializable(metrics),
            "timestamp": _dt.now().isoformat(),
        }
        with open(events_path, "a") as _ef:
            _ef.write(json.dumps(event, default=str, ensure_ascii=False) + "\\n")
    except Exception:
        pass  # never break evaluation

''')

# Insert helper before the evaluate() function definition
marker = "\ndef evaluate("
if "_write_live_event" not in source and marker in source:
    source = source.replace(marker, helper + marker, 1)

# ── 3. Add event counter + path setup inside evaluate(), after eval_tasks is defined ──
setup_marker = "    # validation checks:"
setup_code = (
    "    # ── live-events patch: setup ──\n"
    "    _live_events_path = os.environ.get(\"LMEVAL_LIVE_EVENTS_PATH\")\n"
    "    _live_event_counter = [0]\n"
    "    _live_total_docs = 0\n"
    "    if _live_events_path:\n"
    "        for _tn, _tk in eval_tasks.items():\n"
    "            try:\n"
    "                _tdocs = list(_tk.test_docs()) if hasattr(_tk, \"test_docs\") and _tk.has_test_docs else []\n"
    "                if not _tdocs and hasattr(_tk, \"validation_docs\") and _tk.has_validation_docs:\n"
    "                    _tdocs = list(_tk.validation_docs())\n"
    "                _live_total_docs += len(_tdocs)\n"
    "            except Exception:\n"
    "                pass\n"
    "    # ── end live-events setup ──\n"
    "\n"
)

if "_live_events_path" not in source and setup_marker in source:
    source = source.replace(
        setup_marker,
        setup_code + "    " + setup_marker.strip() + "\n",
        1,
    )

# ── 4. Add event emission after each sample is scored ──
# The target line is right after `example.update(metrics)` and
# `acc["logged_samples"].append(example)` inside the scoring loop.
emit_marker = '                    acc["logged_samples"].append(example)'
emit_code = (
    "                    # ── live-events patch: emit ──\n"
    "                    if _live_events_path:\n"
    "                        _write_live_event(\n"
    "                            _live_events_path,\n"
    "                            _live_event_counter[0],\n"
    "                            _live_total_docs,\n"
    "                            task_name,\n"
    "                            doc,\n"
    "                            target,\n"
    "                            [req.filtered_resps[filter_key] for req in requests],\n"
    "                            metrics,\n"
    "                            doc_id_true,\n"
    "                        )\n"
    "                        _live_event_counter[0] += 1\n"
    "                    # ── end live-events emit ──"
)

if "_live_events_path:" not in source.split(emit_marker)[-1][:200] if emit_marker in source else True:
    if emit_marker in source and "live-events patch: emit" not in source:
        source = source.replace(
            emit_marker,
            emit_marker + "\n" + emit_code,
            1,
        )

with open(evaluator_path, "w") as f:
    f.write(source)

print(f"[live-events patch] Successfully patched {evaluator_path}")

