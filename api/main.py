"""
FastAPI server for controlling lm-evaluation-harness benchmark jobs.
Runs on port 8096, manages evaluation lifecycle and results.
Mirrors the bigcode-eval API pattern for consistency.
"""

import asyncio
import html
import json
import os
import re
import subprocess
import uuid
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiofiles
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# ─── Constants ──────────────────────────────────────────────────────────

WORKSPACE = Path("/workspace")
RESULTS_DIR = WORKSPACE / "results"
LOGS_DIR = WORKSPACE / "logs"
JOBS_STATE_FILE = RESULTS_DIR / ".jobs.json"

API_VERSION = "1.0.0"

# ─── Sanitization ──────────────────────────────────────────────────────

_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_\-]{0,63}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_MAX_STRING_LEN = 512 * 1024


def _validate_id(value: str, label: str = "ID") -> str:
    if not _SAFE_ID_RE.match(value):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {label}: must be alphanumeric with hyphens/underscores, 1-64 chars",
        )
    return value


def _sanitize_string(s: str) -> str:
    s = _CONTROL_CHAR_RE.sub("", s)
    if len(s) > _MAX_STRING_LEN:
        s = s[:_MAX_STRING_LEN] + "\n... [truncated]"
    s = html.escape(s, quote=True)
    return s


def _sanitize_value(obj: Any) -> Any:
    if isinstance(obj, str):
        return _sanitize_string(obj)
    elif isinstance(obj, dict):
        return {k: _sanitize_value(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_value(item) for item in obj]
    return obj


def _sanitize_log_line(line: str) -> str:
    return _CONTROL_CHAR_RE.sub("", line)


# ─── Task Discovery ────────────────────────────────────────────────────

def _discover_tasks() -> List[str]:
    """Use language_eval's task manager to list available tasks."""
    try:
        from language_eval.tasks import TaskManager
        tm = TaskManager()
        return sorted(tm.all_tasks)
    except Exception as e:
        print(f"Warning: could not discover tasks: {e}")
        return []


_ALL_TASKS: List[str] = []


def _get_all_tasks() -> List[str]:
    global _ALL_TASKS
    if not _ALL_TASKS:
        _ALL_TASKS = _discover_tasks()
    return _ALL_TASKS


# ─── Enums and Models ───────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class JobCreate(BaseModel):
    tasks: str = Field(
        ...,
        description="Comma-separated task names (e.g. 'hellaswag', 'arc_challenge,mmlu', 'leaderboard')",
    )
    base_url: str = Field(
        ...,
        description="Full URL of the chat completions endpoint (e.g. http://selfUI:8080/api/chat/completions)",
    )
    model: str = Field(
        default="default",
        description="Model name/ID for inference and result identification",
    )
    api_key: Optional[str] = Field(
        default=None,
        description="API key for the endpoint (set as OPENAI_API_KEY)",
    )
    # Model type
    model_type: str = Field(
        default="local-chat-completions",
        description="language-eval model type: 'local-chat-completions' or 'local-completions'",
    )
    # Generation parameters
    num_fewshot: Optional[int] = Field(
        default=None,
        ge=0,
        description="Number of few-shot examples (None = task default)",
    )
    batch_size: int = Field(default=1, ge=1)
    limit: Optional[int] = Field(
        default=None,
        ge=1,
        description="Number of samples to evaluate (None = all)",
    )
    # Output
    log_samples: bool = Field(
        default=True,
        description="Log individual sample results",
    )
    apply_chat_template: bool = Field(
        default=True,
        description="Apply chat template for chat-completion models",
    )
    dry_run: bool = Field(
        default=False,
        description="Generate synthetic results without running model inference (for UI integration testing)",
    )
    dry_run_delay: float = Field(
        default=0.125,
        ge=0,
        le=5.0,
        description="Delay in seconds between synthetic samples in dry-run mode (default: 0.125s)",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "tasks": "hellaswag",
                "base_url": "http://selfUI:8080/api/chat/completions",
                "model": "Qwen-7B",
                "model_type": "local-chat-completions",
                "num_fewshot": 5,
                "limit": 10,
            }
        }


class Job(BaseModel):
    job_id: str
    status: JobStatus
    tasks: str
    model: str
    base_url: str
    pid: Optional[int] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    log_file: str
    results_dir: str
    error_message: Optional[str] = None
    config: Dict[str, Any] = {}


class HealthResponse(BaseModel):
    status: str
    running_jobs: int
    jobs_total: int
    available_tasks: int
    api_version: str


class TaskInfo(BaseModel):
    name: str
    category: str


# ─── State ──────────────────────────────────────────────────────────────

_jobs: Dict[str, Job] = {}
_processes: Dict[str, subprocess.Popen] = {}

app = FastAPI(title="LM-Eval Harness API", version=API_VERSION)


# ─── Persistence ────────────────────────────────────────────────────────

def _ensure_dirs():
    for d in [RESULTS_DIR, LOGS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def _load_jobs():
    global _jobs
    if JOBS_STATE_FILE.exists():
        data = json.loads(JOBS_STATE_FILE.read_text())
        for job_id, job_data in data.items():
            try:
                job_data["created_at"] = datetime.fromisoformat(job_data["created_at"])
                if job_data.get("started_at"):
                    job_data["started_at"] = datetime.fromisoformat(job_data["started_at"])
                if job_data.get("finished_at"):
                    job_data["finished_at"] = datetime.fromisoformat(job_data["finished_at"])
                job = Job(**job_data)
                if job.status == JobStatus.RUNNING:
                    job.status = JobStatus.FAILED
                    job.error_message = "Process lost on restart"
                    job.finished_at = datetime.now()
                _jobs[job_id] = job
            except Exception as e:
                print(f"Failed to load job {job_id}: {e}")


def _save_jobs():
    tmp = JOBS_STATE_FILE.with_suffix(".tmp")
    data = {jid: j.model_dump(mode="json") for jid, j in _jobs.items()}
    tmp.write_text(json.dumps(data, indent=2, default=str))
    tmp.replace(JOBS_STATE_FILE)


# ─── Background Polling ────────────────────────────────────────────────

async def _poll_jobs():
    while True:
        await asyncio.sleep(5)
        changed = False
        for job_id, job in list(_jobs.items()):
            if job.status != JobStatus.RUNNING:
                continue

            proc = _processes.get(job_id)
            if not proc:
                continue

            rc = proc.poll()
            if rc is not None:
                job.exit_code = rc
                job.finished_at = datetime.now()
                job.status = JobStatus.COMPLETED if rc == 0 else JobStatus.FAILED
                if rc != 0:
                    try:
                        log_lines = Path(job.log_file).read_text().splitlines()
                        tail = log_lines[-10:] if len(log_lines) >= 10 else log_lines
                        job.error_message = _sanitize_string("\n".join(tail))
                    except Exception:
                        job.error_message = f"Process exited with code {rc}"
                del _processes[job_id]
                changed = True

        if changed:
            _save_jobs()


@app.on_event("startup")
async def startup_event():
    _ensure_dirs()
    _load_jobs()
    asyncio.create_task(_poll_jobs())


# ─── Task Discovery Endpoints ──────────────────────────────────────────

# Leaderboard v2 categories
_LEADERBOARD_V2_TASKS = {
    "leaderboard_ifeval", "leaderboard_bbh", "leaderboard_math_hard",
    "leaderboard_gpqa", "leaderboard_musr", "leaderboard_mmlu_pro",
}

# Classic v1 leaderboard tasks
_LEADERBOARD_V1_TASKS = {
    "arc_challenge", "arc_easy", "hellaswag", "mmlu", "truthfulqa_mc2",
    "winogrande", "gsm8k",
}


def _categorize_task(name: str) -> str:
    if name in _LEADERBOARD_V2_TASKS or name == "leaderboard":
        return "leaderboard-v2"
    if name in _LEADERBOARD_V1_TASKS:
        return "leaderboard-v1"
    if name.startswith("mmlu"):
        return "mmlu"
    if name.startswith("arc"):
        return "arc"
    if name.startswith("gsm"):
        return "math"
    if name.startswith("bbh"):
        return "reasoning"
    if name.startswith("gpqa"):
        return "reasoning"
    if name.startswith("truthfulqa"):
        return "truthfulqa"
    if name.startswith("hellaswag"):
        return "commonsense"
    if name.startswith("winogrande"):
        return "commonsense"
    return "other"


@app.get("/api/tasks")
def list_tasks() -> List[TaskInfo]:
    return [
        TaskInfo(name=name, category=_categorize_task(name))
        for name in _get_all_tasks()
    ]


@app.get("/api/tasks/categories")
def list_task_categories() -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {}
    for name in _get_all_tasks():
        cat = _categorize_task(name)
        categories.setdefault(cat, []).append(name)
    return categories


# ─── Dry-Run Synthetic Data ───────────────────────────────────────────

import random
import threading
import time as _time

_DRY_RUN_BENCHMARKS = {
    "hellaswag": {
        "metrics": {"acc,none": (0.25, 0.65), "acc_norm,none": (0.30, 0.70), "acc_stderr,none": (0.005, 0.02), "acc_norm_stderr,none": (0.005, 0.02)},
        "prompts": [
            ("A chef is in a kitchen. He picks up a knife and", "begins to carefully dice the onions on the cutting board."),
            ("The dog runs across the yard. It then", "jumps over the fence and chases a squirrel up a tree."),
            ("A student opens a textbook. She", "starts reading the first chapter about the history of mathematics."),
            ("The mechanic looks under the hood. He", "identifies a loose belt and tightens it with a wrench."),
            ("Two children are playing in the park. They", "take turns pushing each other on the swings."),
        ],
    },
    "arc_challenge": {
        "metrics": {"acc,none": (0.20, 0.55), "acc_norm,none": (0.25, 0.60), "acc_stderr,none": (0.008, 0.025), "acc_norm_stderr,none": (0.008, 0.025)},
        "prompts": [
            ("Which of the following is a renewable resource?\nA) Coal\nB) Natural gas\nC) Solar energy\nD) Petroleum", "C"),
            ("What causes the seasons on Earth?\nA) Distance from the sun\nB) Tilt of Earth's axis\nC) Speed of rotation\nD) Size of the moon", "B"),
            ("Which layer of the atmosphere contains the ozone layer?\nA) Troposphere\nB) Stratosphere\nC) Mesosphere\nD) Thermosphere", "B"),
        ],
    },
    "mmlu": {
        "metrics": {"acc,none": (0.25, 0.70), "acc_stderr,none": (0.005, 0.03)},
        "prompts": [
            ("In economics, what does GDP stand for?\nA) Gross Domestic Product\nB) General Distribution Plan\nC) Global Development Program\nD) Government Deficit Percentage", "A"),
            ("Which amendment to the US Constitution abolished slavery?\nA) 12th\nB) 13th\nC) 14th\nD) 15th", "B"),
        ],
    },
    "truthfulqa_mc2": {
        "metrics": {"acc,none": (0.30, 0.65), "acc_stderr,none": (0.005, 0.02)},
        "prompts": [
            ("What happens to you if you eat watermelon seeds?", "Nothing happens, they pass through your digestive system."),
            ("Can you teach an old dog new tricks?", "Yes, older dogs can learn new behaviors with patience and positive reinforcement."),
        ],
    },
    "winogrande": {
        "metrics": {"acc,none": (0.50, 0.75), "acc_stderr,none": (0.005, 0.02)},
        "prompts": [
            ("The trophy doesn't fit in the brown suitcase because _ is too big.", "trophy"),
            ("The city councilmen refused the demonstrators a permit because they feared violence.", "councilmen"),
        ],
    },
    "gsm8k": {
        "metrics": {"exact_match,strict-match": (0.05, 0.50), "exact_match,flexible-extract": (0.10, 0.55)},
        "prompts": [
            ("Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes muffins with 4. She sells the rest for $2 each. How much does she make daily?", "18"),
            ("A store sells apples for $2 each. If you buy 5 and pay with $20, how much change?", "10"),
        ],
    },
    "leaderboard_ifeval": {
        "metrics": {"prompt_level_strict_acc,none": (0.20, 0.60), "inst_level_strict_acc,none": (0.25, 0.65)},
        "prompts": [
            ("Write a 200-word essay about climate change. Your response must contain exactly 3 bullet points.", "Climate change is one of the most pressing issues..."),
            ("List 5 countries in Europe. Use all capital letters.", "FRANCE, GERMANY, SPAIN, ITALY, PORTUGAL"),
        ],
    },
}

# Fallback for any benchmark not in the templates
_DRY_RUN_DEFAULT = {
    "metrics": {"acc,none": (0.20, 0.65), "acc_stderr,none": (0.005, 0.03)},
    "prompts": [
        ("What is the answer to this test question?", "The correct answer is A."),
        ("Please evaluate the following statement.", "The statement is true based on the given context."),
    ],
}


def _run_dry_run_job(job_id: str, req: JobCreate):
    """Background thread that generates synthetic language-eval results with realistic timing."""
    job = _jobs.get(job_id)
    if not job:
        return

    output_dir = Path(job.results_dir)
    model_safe = req.model.replace("/", "__")
    model_dir = output_dir / model_safe
    model_dir.mkdir(parents=True, exist_ok=True)

    log_path = Path(job.log_file)
    events_path = LOGS_DIR / f"{job_id}.events.jsonl"
    date_id = datetime.now().isoformat().replace(":", "-")

    tasks = [t.strip() for t in req.tasks.split(",")]
    num_samples = req.limit or 10
    delay = req.dry_run_delay

    with open(log_path, "w") as log_fh:
        log_fh.write(f"[dry-run] Starting synthetic evaluation for {req.model}\n")
        log_fh.write(f"[dry-run] Tasks: {', '.join(tasks)}, Samples per task: {num_samples}, Delay: {delay}s\n\n")

        all_results = {}
        all_n_samples = {}
        all_versions = {}
        all_higher_is_better = {}
        event_index = 0
        total_events = len(tasks) * num_samples

        for task_name in tasks:
            template = _DRY_RUN_BENCHMARKS.get(task_name, _DRY_RUN_DEFAULT)
            prompts = template["prompts"]
            metrics_ranges = template["metrics"]

            log_fh.write(f"[dry-run] Running task: {task_name} ({num_samples} samples)\n")
            log_fh.flush()

            # Generate per-sample data
            samples = []
            for i in range(num_samples):
                prompt_text, target = prompts[i % len(prompts)]
                correct = random.random() < 0.5
                response = target if correct else f"incorrect_response_{i}"

                sample_metrics = {}
                metric_names = []
                for metric_name in metrics_ranges:
                    if "stderr" not in metric_name:
                        metric_names.append(metric_name)
                        sample_metrics[metric_name] = 1.0 if correct else 0.0

                sample = {
                    "doc_id": i,
                    "doc": {"question": prompt_text, "task_name": task_name},
                    "target": target,
                    "arguments": [[prompt_text, {"until": ["\n"]}]],
                    "resps": [[response]],
                    "filtered_resps": [[response]],
                    "task_name": task_name,
                    "metrics": metric_names,
                    **sample_metrics,
                }
                samples.append(sample)

                # Write live event
                event = {
                    "index": event_index,
                    "total": total_events,
                    "task_name": task_name,
                    "doc_id": i,
                    "prompt": prompt_text,
                    "target": target,
                    "response": response,
                    "metrics": sample_metrics,
                    "timestamp": datetime.now().isoformat(),
                }
                with open(events_path, "a") as ef:
                    ef.write(json.dumps(event) + "\n")

                event_index += 1
                log_fh.write(f"  [{event_index}/{total_events}] {task_name} doc_{i}: {'correct' if correct else 'incorrect'}\n")
                log_fh.flush()

                _time.sleep(delay)

            # Write samples JSONL
            samples_file = model_dir / f"samples_{task_name}_{date_id}.jsonl"
            with open(samples_file, "w") as sf:
                for s in samples:
                    sf.write(json.dumps(s, default=str) + "\n")

            # Aggregate metrics for this task
            task_metrics = {}
            for metric_name, (lo, hi) in metrics_ranges.items():
                task_metrics[metric_name] = round(random.uniform(lo, hi), 4)
            task_metrics["alias"] = task_name
            all_results[task_name] = task_metrics
            all_n_samples[task_name] = {"original": num_samples, "effective": num_samples}
            all_versions[task_name] = 1.0
            all_higher_is_better[task_name] = {k: True for k in metrics_ranges if "stderr" not in k}

        # Write aggregated results JSON
        eval_time = round(total_events * delay + random.uniform(0.5, 2.0), 2)
        results_data = {
            "results": all_results,
            "n-samples": all_n_samples,
            "config": {
                "model": req.model,
                "model_args": f"model={req.model},base_url={req.base_url},num_concurrent=1,max_retries=3,tokenized_requests=False",
                "batch_size": req.batch_size,
                "batch_sizes": [],
                "device": None,
                "use_cache": None,
                "limit": num_samples,
                "bootstrap_iters": 100000,
                "gen_kwargs": None,
            },
            "versions": all_versions,
            "n-shot": {t: 0 for t in tasks},
            "higher_is_better": all_higher_is_better,
            "model_name": req.model,
            "total_evaluation_time_seconds": eval_time,
            "git_hash": "dry-run",
            "date": datetime.now().isoformat(),
        }
        results_file = model_dir / f"results_{date_id}.json"
        results_file.write_text(json.dumps(results_data, indent=2, default=str))

        # Update .jobs.json
        jobs_meta = {}
        if JOBS_STATE_FILE.parent != RESULTS_DIR:
            pass
        jobs_meta_file = RESULTS_DIR / ".jobs.json"
        if jobs_meta_file.exists():
            try:
                jobs_meta = json.loads(jobs_meta_file.read_text())
            except Exception:
                pass
        jobs_meta[job_id] = {
            "created_at": int(_time.time()),
            "model": req.model,
            "tasks": req.tasks,
            "status": "completed",
            "dry_run": True,
        }
        tmp = jobs_meta_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(jobs_meta, indent=2))
        tmp.replace(jobs_meta_file)

        log_fh.write(f"\n[dry-run] Completed! {total_events} samples across {len(tasks)} tasks in {eval_time}s\n")
        log_fh.write(f"[dry-run] Results: {results_file}\n")

    # Mark job as completed
    job.exit_code = 0
    job.finished_at = datetime.now()
    job.status = JobStatus.COMPLETED
    _save_jobs()


# ─── Jobs Endpoints ────────────────────────────────────────────────────

@app.post("/api/jobs", status_code=201)
def create_job(req: JobCreate) -> Job:
    """Start a new language-eval evaluation job."""
    # Validate string inputs
    for field_name, value in [("tasks", req.tasks), ("model", req.model), ("base_url", req.base_url)]:
        if "\x00" in value:
            raise HTTPException(status_code=400, detail=f"Invalid {field_name}: contains null bytes")
        if any(c in value for c in [";", "|", "&", "`", "$", "(", ")", "\n", "\r"]):
            raise HTTPException(status_code=400, detail=f"Invalid {field_name}: contains disallowed characters")
    if req.api_key and ("\x00" in req.api_key or "\n" in req.api_key):
        raise HTTPException(status_code=400, detail="Invalid api_key")

    if not req.dry_run and req.model_type not in ("local-chat-completions", "local-completions"):
        raise HTTPException(status_code=400, detail="model_type must be 'local-chat-completions' or 'local-completions'")

    # Check if any job is already running
    for job in _jobs.values():
        if job.status == JobStatus.RUNNING:
            raise HTTPException(
                status_code=409,
                detail=f"An evaluation job is already running (job_id: {job.job_id})",
            )

    job_id = str(uuid.uuid4())[:8]
    output_dir = str(RESULTS_DIR / job_id)
    log_file = str(LOGS_DIR / f"{job_id}.log")

    # ── Dry-run mode: generate synthetic results in a background thread ──
    if req.dry_run:
        config = req.model_dump()
        job = Job(
            job_id=job_id,
            status=JobStatus.RUNNING,
            tasks=req.tasks,
            model=req.model,
            base_url=req.base_url,
            pid=None,
            created_at=datetime.now(),
            started_at=datetime.now(),
            log_file=log_file,
            results_dir=output_dir,
            config=config,
        )
        _jobs[job_id] = job
        _save_jobs()

        t = threading.Thread(target=_run_dry_run_job, args=(job_id, req), daemon=True)
        t.start()

        return job

    # ── Normal mode: launch language_eval CLI ──────────────────────────────────
    # Build CLI command
    # language-eval POSTs directly to base_url, so it must include /chat/completions
    effective_base_url = req.base_url.rstrip("/")
    if req.model_type == "local-chat-completions" and not effective_base_url.endswith("/chat/completions"):
        effective_base_url += "/chat/completions"
    cmd = [
        "language_eval",
        "--model", req.model_type,
        "--model_args", f"model={req.model},base_url={effective_base_url},num_concurrent=1,max_retries=3,tokenized_requests=False",
        "--tasks", req.tasks,
        "--batch_size", str(req.batch_size),
        "--output_path", output_dir,
    ]

    if req.num_fewshot is not None:
        cmd.extend(["--num_fewshot", str(req.num_fewshot)])
    if req.limit is not None:
        cmd.extend(["--limit", str(req.limit)])
    if req.log_samples:
        cmd.append("--log_samples")
    if req.apply_chat_template and req.model_type == "local-chat-completions":
        cmd.append("--apply_chat_template")

    # Open log file
    try:
        log_fh = open(log_file, "w")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to open log: {e}")

    # Launch process
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if req.api_key:
        env["OPENAI_API_KEY"] = req.api_key

    # Live events file for streaming progress
    live_events_file = str(LOGS_DIR / f"{job_id}.events.jsonl")
    env["LMEVAL_LIVE_EVENTS_PATH"] = live_events_file

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            env=env,
        )
    except Exception as e:
        log_fh.close()
        raise HTTPException(status_code=500, detail=f"Failed to start evaluation: {e}")

    config = req.model_dump()

    job = Job(
        job_id=job_id,
        status=JobStatus.RUNNING,
        tasks=req.tasks,
        model=req.model,
        base_url=req.base_url,
        pid=proc.pid,
        created_at=datetime.now(),
        started_at=datetime.now(),
        log_file=log_file,
        results_dir=output_dir,
        config=config,
    )

    _jobs[job_id] = job
    _processes[job_id] = proc
    _save_jobs()

    return job


@app.get("/api/jobs")
def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
) -> List[Job]:
    jobs = sorted(_jobs.values(), key=lambda j: j.created_at, reverse=True)
    if status:
        jobs = [j for j in jobs if j.status == status]
    return jobs


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> Job:
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/api/jobs/{job_id}/logs")
async def get_job_logs(
    job_id: str, tail: int = Query(100, ge=1, le=10000), stream: bool = Query(False)
):
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    log_path = Path(job.log_file)
    if not log_path.exists():
        raise HTTPException(status_code=404, detail="Log file not found")

    if not stream:
        try:
            lines = log_path.read_text().splitlines()
            sanitized = [_sanitize_log_line(l) for l in lines[-tail:]]
            return {"lines": sanitized, "total_lines": len(lines)}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def generate():
        async with aiofiles.open(log_path, "r") as f:
            await f.seek(0, 2)
            while True:
                line = await f.readline()
                if line:
                    yield _sanitize_log_line(line)
                else:
                    j = _jobs.get(job_id)
                    if j and j.status not in (JobStatus.RUNNING, JobStatus.PENDING):
                        remaining = await f.readline()
                        while remaining:
                            yield _sanitize_log_line(remaining)
                            remaining = await f.readline()
                        break
                    await asyncio.sleep(0.5)

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/api/jobs/{job_id}/live")
async def get_job_live(job_id: str):
    """Stream live sample events as SSE during a running evaluation.

    Reads from the JSONL events file written by the patched evaluator.
    Each event contains: index, total, task_name, prompt, target, response, metrics.
    """
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    events_path = LOGS_DIR / f"{job_id}.events.jsonl"

    async def generate():
        lines_sent = 0
        while True:
            # Read new event lines from the JSONL file
            if events_path.exists():
                try:
                    async with aiofiles.open(events_path, "r") as f:
                        all_lines = await f.readlines()
                    new_lines = all_lines[lines_sent:]
                    for line in new_lines:
                        line = line.strip()
                        if line:
                            yield f"data: {line}\n\n"
                            lines_sent += 1
                except Exception:
                    pass

            # Check if job is done
            j = _jobs.get(job_id)
            if j and j.status not in (JobStatus.RUNNING, JobStatus.PENDING):
                # Flush any remaining lines
                if events_path.exists():
                    try:
                        async with aiofiles.open(events_path, "r") as f:
                            all_lines = await f.readlines()
                        for line in all_lines[lines_sent:]:
                            line = line.strip()
                            if line:
                                yield f"data: {line}\n\n"
                    except Exception:
                        pass
                yield f"event: done\ndata: {{\"status\": \"{j.status}\"}}\n\n"
                break

            await asyncio.sleep(1)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.delete("/api/jobs/{job_id}")
def cancel_job(job_id: str):
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")

    proc = _processes.get(job_id)
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        del _processes[job_id]

    job.status = JobStatus.CANCELLED
    job.finished_at = datetime.now()
    _save_jobs()

    return {"status": "cancelled", "job_id": job_id}


@app.delete("/api/jobs/{job_id}/purge")
def purge_job(job_id: str):
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot purge a running job. Cancel it first.")

    import shutil

    deleted_files = []
    log_path = Path(job.log_file)
    if log_path.exists():
        log_path.unlink()
        deleted_files.append(str(log_path))

    results_path = Path(job.results_dir)
    if results_path.exists():
        shutil.rmtree(results_path)
        deleted_files.append(str(results_path))

    del _jobs[job_id]
    _save_jobs()

    return {"purged": True, "job_id": job_id, "files_removed": deleted_files}


# ─── Results Endpoints ──────────────────────────────────────────────────

def _find_results_json(job: Job) -> Optional[Path]:
    """Find the results JSON file produced by language-eval in the output directory.

    language-eval writes results to: {output_path}/{model_name}/results_{timestamp}.json
    """
    results_dir = Path(job.results_dir)
    if not results_dir.exists():
        return None
    for path in results_dir.rglob("results_*.json"):
        return path
    return None


@app.get("/api/results")
def list_results() -> List[Dict[str, Any]]:
    """List all completed evaluation results."""
    results = []
    for job_id, job in _jobs.items():
        if job.status != JobStatus.COMPLETED:
            continue
        results_file = _find_results_json(job)
        if not results_file:
            continue
        try:
            data = json.loads(results_file.read_text())
            task_results = data.get("results", {})
            scores = {}
            for task_name, metrics in task_results.items():
                scores[task_name] = {
                    k: round(v, 4) if isinstance(v, float) else v
                    for k, v in metrics.items()
                    if not k.startswith("_")
                }

            results.append(_sanitize_value({
                "id": job_id,
                "job_id": job_id,
                "model": job.model,
                "tasks": job.tasks,
                "scores": scores,
                "config": job.config,
                "created_at": job.created_at.isoformat(),
                "finished_at": job.finished_at.isoformat() if job.finished_at else None,
            }))
        except Exception as e:
            print(f"Failed to parse results for job {job_id}: {e}")

    return sorted(results, key=lambda r: r.get("finished_at") or "", reverse=True)


@app.get("/api/results/{job_id}")
def get_result(job_id: str) -> Dict[str, Any]:
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    results_file = _find_results_json(job)
    if not results_file:
        raise HTTPException(status_code=404, detail="Results not found")

    try:
        data = json.loads(results_file.read_text())
        return _sanitize_value(data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read results: {e}")


@app.get("/api/results/{job_id}/samples")
def get_result_samples(
    job_id: str,
    task: Optional[str] = Query(None, description="Filter by task name"),
) -> List[Dict[str, Any]]:
    """Get per-sample details for a completed evaluation.

    language-eval writes samples as JSONL files: samples_{task}_{timestamp}.jsonl
    """
    _validate_id(job_id, "job_id")
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    results_dir = Path(job.results_dir)
    if not results_dir.exists():
        raise HTTPException(status_code=404, detail="Results directory not found")

    samples = []
    for path in sorted(results_dir.rglob("samples_*.jsonl")):
        # Optionally filter by task name (filename: samples_{task}_{date}.jsonl)
        if task:
            fname = path.stem  # samples_{task}_{date}
            if task not in fname:
                continue
        try:
            with open(path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        sample = json.loads(line)
                        samples.append(_sanitize_value(sample))
        except Exception as e:
            print(f"Failed to parse samples from {path}: {e}")

    if not samples:
        raise HTTPException(status_code=404, detail="No sample data found")

    return samples


# ─── Health Endpoint ────────────────────────────────────────────────────

@app.get("/health")
def health() -> HealthResponse:
    running_count = sum(1 for j in _jobs.values() if j.status == JobStatus.RUNNING)
    return HealthResponse(
        status="ok",
        running_jobs=running_count,
        jobs_total=len(_jobs),
        available_tasks=len(_get_all_tasks()),
        api_version=API_VERSION,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8096, log_level="info")
