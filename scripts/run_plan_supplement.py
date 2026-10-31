"""Host+Plan revision supplement (spec 0046): Self-Refine, CoVe and NEA-Refine on the Host+Plan answers.

Completes the 3 (parent answer) x 4 (revision) design of the formal batch (run_id 1008,
directory outputs/formal-test270-20261008). The formal records stay unchanged; this script keeps its own
append-only ledger in outputs/formal-test270-20261008/supplement-plan-revisions/, takes the formal
Host+Plan (``P``) answers and their task analysis as parents, runs the same revision methods
(Self-Refine = ``SR`` slot, CoVe = ``CV`` slot, NEA-Refine = ``FB`` slot of the formal transport) and the same
two-judge six-dimension scoring, and records the new groups as ``SRP``, ``CVP`` and ``FP``.

  plan     print scope and transport, no API calls
  run      generate (and score unless --generation-only); rerun the same command to resume
  score    complete the scoring: reuse the judgments already made in the run, request the missing ones
           on the all-official transport (spec 0046: the relay quota ran out), write scoring/final/
  analyze  write supplement-plan-revisions/analysis.json (no API calls)
  merge    append the three groups to the formal answers/judgments files (spec 0046; no API calls)
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from nea.providers import config  # noqa: E402
from nea.runtime.records import answer_sha256, canonical_bytes, digest, write_once  # noqa: E402
import run_gplan_cove_supplement as cv0  # noqa: E402  (spec 0045: CoVe on an explicit parent, long-path writes)

RUN_ID = "1016-plan-revisions"
SOURCE = cv0.SOURCE
SUPPLEMENT_ROOT = SOURCE / "supplement-plan-revisions"
SOURCE_SHA = cv0.SOURCE_SHA
PARENT = "P"
# New group -> (formal transport slot whose stage registration it reuses, label)
GROUPS = {"SRP": "SR", "CVP": "CV", "FP": "FB"}
# Standard MiMo (thinking disabled, 16 in flight) and official MiniMax-M3.1-Flash (2 in flight) as in 0045;
# MiniMax-M3 (judge, thinking disabled) goes through the relay (spec 0046), 8 in flight (relay default).
TRANSPORT = {"mimo_api": "standard", "mimo_concurrency": 16, "minimax_concurrency": 2,
             "minimax_m3_api": "relay-m3", "minimax_relay_concurrency": 8}
# Scoring completion after the relay key ran out: every remaining MiniMax request goes to the official endpoint.
TRANSPORT_SCORE = {"mimo_api": "standard", "mimo_concurrency": 16, "minimax_concurrency": 2, "minimax_m3_api": "official"}
DIMENSIONS = cv0.DIMENSIONS
COMPARISONS = (("SRP-P", "SRP", "P"), ("CVP-P", "CVP", "P"), ("FP-P", "FP", "P"),
               ("FP-SRP", "FP", "SRP"), ("FP-CVP", "FP", "CVP"),
               ("FB-B", "FB", "B"), ("F-L", "F", "L"))
read = cv0.read
source_phase_dir = cv0.source_phase_dir


def long_path(path: Path) -> Path:
    full = str(Path(path).resolve())
    prefix = "\\\\?\\"
    return Path(prefix + full) if sys.platform == "win32" and len(full) >= 240 and not full.startswith(prefix) else Path(path)


def read_long(path: Path):
    return json.loads(long_path(path).read_text(encoding="utf-8"))


REFUSAL_STATUSES = frozenset({403, 422})


def refusal_file(run_root: Path) -> Path:
    return run_root / "provider-refusals.jsonl"


def load_refusals(run_root: Path) -> list[dict]:
    path = refusal_file(run_root)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def last_http_status(phase_root: Path, key: str):
    files = sorted(long_path(phase_root / "failures" / key).glob("*.json"))
    return json.loads(long_path(files[-1]).read_text(encoding="utf-8")).get("http_status") if files else None


def install_refusal_guard(phase) -> None:
    """Spec 0046: a request the provider permanently refuses (HTTP 403/422 content filter) must not stall the run.

    The refusal is recorded append-only in provider-refusals.jsonl and the request is skipped from then on;
    the caller turns a refused generation stage into "parent answer retained", a refused score stays missing.
    """
    import time
    from nea.runtime.formal_transport import prepare
    original = phase.call
    run_root = phase.run.root.parents[2]

    def keys_of(jobs):
        return prepare(phase.run, phase.env.generator, jobs, question_splits=phase.env.question_set.qid_splits(),
                       phase_split=phase.source_split, parent_index=phase.parent_index,
                       _model_identity=getattr(phase.env.router, "model_identity", None))[1]

    def call(jobs):
        jobs = list(jobs)
        for _ in range(200):
            known = {row["key"] for row in load_refusals(run_root)}
            keys = keys_of(jobs) if jobs else {}
            active = [job for job in jobs if keys[job] not in known]
            results = original(active)
            results = {**results, **{job: None for job in jobs if keys[job] in known}}
            if phase.stop_reason != "http_status_error":
                return results
            fresh = [job for job in active if results.get(job) is None
                     and last_http_status(phase.run.root, keys[job]) in REFUSAL_STATUSES]
            if not fresh:
                return results
            with refusal_file(run_root).open("a", encoding="utf-8") as handle:
                for job in fresh:
                    handle.write(json.dumps({"generator": phase.env.generator, "qid": job.qid, "base": job.base,
                        "arm": job.arm, "stage": job.stage, "role": job.role, "key": keys[job],
                        "http_status": last_http_status(phase.run.root, keys[job]), "time": time.time()},
                        ensure_ascii=False) + "\n")
            # The transport registered a one-hour cooldown for this request; it is not a quota stop, release it
            # (record kept in cooldowns-released/) so the other requests continue.
            cooldowns = run_root / "supervision" / "cooldowns"
            released = run_root / "supervision" / "cooldowns-released"
            refused = {keys[job] for job in fresh}
            for path in sorted(cooldowns.glob("*.json")):
                if json.loads(path.read_text(encoding="utf-8")).get("key") in refused:
                    released.mkdir(parents=True, exist_ok=True)
                    path.replace(released / path.name)
            phase.stop_reason = None
        raise RuntimeError("too many provider refusals in one wave")

    phase.call = call


def run_root_of(phase) -> Path:
    return phase.run.root.parents[2]


def generation_root(run_root: Path, generator: str) -> Path:
    return run_root / "operator-v2" / generator / "test"


def prior_judgments(run_root: Path, generator: str) -> dict:
    """Successful judgments already made in the generation run (complete file and incomplete snapshots)."""
    base = generation_root(run_root, generator)
    found = {}
    files = list(base.glob("quality/*/judgments.json")) + list((base / "incomplete" / "quality").glob("*/judgments-*.json"))
    for path in sorted(files):
        for row in read_long(path)["rows"]:
            if row["status"] == "ok":
                found.setdefault((row["arm"], row["qid"], row["base"], row["teacher"]), row)
    return found


def final_judgments(run_root: Path, generator: str) -> list[dict]:
    return read(run_root / "scoring" / "final" / f"{generator}.json")["rows"]


def source_manifest() -> dict:
    """SHA256 of every formal file this supplement reads, keyed by its outputs/1008 registered path."""
    files = []
    for generator in config.EXPERIMENT_GENERATORS:
        files += [source_phase_dir(generator) / name for name in ("answers.json", "judgments.json")]
    files += sorted((SOURCE / "operator-v2" / "shared" / "test").glob("*.json"))
    return {cv0.LEGACY_PREFIX + path.relative_to(SOURCE).as_posix(): original_sha256(path) for path in files}


def original_sha256(path: Path) -> str:
    """SHA256 of a formal file as it was registered: rows of the groups merged by spec 0046 removed (CVL stays)."""
    data = path.read_bytes()
    if path.name in {"answers.json", "judgments.json"} and path.parent.name == "test":
        document = json.loads(data)
        data = canonical_bytes({**document, "rows": [row for row in document["rows"] if row.get("arm") not in GROUPS]})
    return hashlib.sha256(data).hexdigest()


def merge(run_root: Path) -> dict:
    """Append the SRP/CVP/FP answers and final judgments to the formal files (spec 0046; idempotent).

    Dropping the three groups and re-serializing reproduces the SHA256 recorded in source-1008-manifest.json.
    Every target is checked before any file is written.
    """
    registered = read(run_root / "source-1008-manifest.json")
    plans, summary = [], {}
    for generator in config.EXPERIMENT_GENERATORS:
        phase_root = run_root / "operator-v2" / generator / "test"
        additions = {"answers": read(phase_root / "supplement-answers.json")["rows"],
                     "judgments": final_judgments(run_root, generator)}
        for name, extra in additions.items():
            path = source_phase_dir(generator) / f"{name}.json"
            document = read(path)
            key = next(k for k in registered if k.endswith(f"/{generator}/test/{name}.json"))
            if original_sha256(path) != registered[key]:
                raise RuntimeError(f"original rows of {path} differ from the registered SHA256")
            present = [row for row in document["rows"] if row.get("arm") in GROUPS]
            if present:
                if present != extra:
                    raise RuntimeError(f"{path} already holds different Host+Plan group rows")
                summary[f"{generator}/{name}"] = "already merged"
                continue
            plans.append((path, canonical_bytes({**document, "rows": document["rows"] + extra})))
            summary[f"{generator}/{name}"] = f"+{len(extra)} rows"
    for path, data in plans:
        temp = path.with_suffix(".json.tmp")
        temp.write_bytes(data)
        temp.replace(path)
    return summary


def run_generator(phase, generation_only: bool = False) -> dict:
    from nea.pipeline.artifacts import RunDirectoryLock
    from nea.pipeline import formal_evaluation
    from nea.pipeline.formal_evaluation import run_evaluations
    from nea.pipeline.registry import BASES
    if not getattr(formal_evaluation.write_once, "long_path_safe", False):
        formal_evaluation.write_once = cv0.long_path_safe(formal_evaluation.write_once)
        formal_evaluation.write_once.long_path_safe = True
    generator = phase.run.identity["generator"]
    source = {(row["qid"], row["base"]): row for row in read(source_phase_dir(generator) / "answers.json")["rows"]
              if row["arm"] == PARENT and row.get("comparison", "main") == "main"}
    phase._scheduled_jobs = set()
    phase._cells = [(row["qid"], base) for row in phase.rows for base in BASES]
    phase._payloads = {cell: phase.base_payload(phase._by_qid[cell[0]], cell[1]) for cell in phase._cells}
    install_refusal_guard(phase)
    parents, phase._tasks = {}, {}
    for cell in phase._cells:
        row = source[cell]
        if row.get("status") != "ok" or row["evidence"] != phase._payloads[cell]["knowledge_chunks"]:
            raise RuntimeError(f"1008 Host+Plan parent or evidence differs for {cell}")
        if not isinstance(row.get("task"), dict):
            raise RuntimeError(f"1008 Host+Plan row has no task analysis for {cell}")
        parents[cell] = row["answer"]
        phase._tasks[cell] = row["task"]
    # Transport slots SR/FB/CV all register the parent as arm B; point that slot at the Host+Plan answer.
    phase.parent_index = {(*cell, "B"): answer_sha256(text) for cell, text in parents.items()}
    contexts = {(*cell, slot): phase._context((*cell, slot), parents[cell], generic=slot == "SR")
                for cell in phase._cells for slot in ("SR", "FB")}
    final, _ = phase._operators(contexts)
    final.update({(*cell, "CV"): record for cell, record in cv0.cove_on_parent(phase, parents).items()})
    refused = {(r["qid"], r["base"], r["arm"]) for r in load_refusals(run_root_of(phase))
               if r["generator"] == generator and r["stage"] != "judge"}
    rows = []
    for cell in phase._cells:
        for arm, slot in GROUPS.items():
            record = final.get((*cell, slot)) if slot != "CV" else final.get((*cell, "CV"))
            failed = record is None or str(record.get("fallback_reason") or "").endswith("_failed")
            if record is not None and (*cell, slot) in refused:
                record.update(answer=parents[cell], answer_sha256=answer_sha256(parents[cell]), revised=False,
                              status="parent_retained", fallback_reason="provider_content_refused")
                failed = False
            if record is not None:
                record["parent_arm"] = PARENT
            row = phase._answer_row((*cell, arm), None if failed else record["answer"], revision=record)
            row["parent_arm"] = PARENT
            rows.append(row)
    complete = all(row["status"] == "ok" for row in rows) and not phase.stop_reason
    with RunDirectoryLock(phase.run.root):
        if complete:
            write_once(phase.run.root / "supplement-answers.json", {"rows": rows})
        else:
            write_once(phase.run.root / "snapshots" / digest(rows) / "supplement-answers.json", {"rows": rows})
    summary = {"generator": generator, "answers": sum(row["status"] == "ok" for row in rows),
               "positions": len(rows), "generation_complete": complete, "stop_reason": phase.stop_reason,
               "provider_refused_positions": sum(r.get("revision", {}).get("fallback_reason") == "provider_content_refused" for r in rows)}
    if complete and not generation_only:
        evaluation = run_evaluations(phase, rows, scoring_only=True)["summary"]
        summary.update(judgments=evaluation["judgments"], judgments_missing=evaluation["judgments_missing"],
                       evaluation_status=evaluation["status"])
    summary["complete"] = complete and summary.get("evaluation_status") == "COMPLETE"
    return summary


def score_generator(phase, run_root: Path, rescore_all: bool) -> dict:
    from nea.pipeline.formal_evaluation import run_evaluations
    from nea.pipeline import formal_evaluation
    from nea.pipeline.registry import BASES
    if not getattr(formal_evaluation.write_once, "long_path_safe", False):
        formal_evaluation.write_once = cv0.long_path_safe(formal_evaluation.write_once)
        formal_evaluation.write_once.long_path_safe = True
    generator = phase.run.identity["generator"]
    source = generation_root(run_root, generator) / "supplement-answers.json"
    if not source.is_file():
        return {"generator": generator, "complete": False, "reason": "generation not complete"}
    rows = read(source)["rows"]
    prior = {} if rescore_all else prior_judgments(run_root, generator)
    todo = [row for row in rows if any((row["arm"], row["qid"], row["base"], t) not in prior for t in ("teacher1", "teacher2"))]
    phase._scheduled_jobs = set()
    phase._cells = [(row["qid"], base) for row in phase.rows for base in BASES]
    phase._payloads = {cell: phase.base_payload(phase._by_qid[cell[0]], cell[1]) for cell in phase._cells}
    install_refusal_guard(phase)
    new = {}
    if todo:
        result = run_evaluations(phase, todo, scoring_only=True)
        for row in result["judgments"]:
            if row["status"] == "ok":
                new[(row["arm"], row["qid"], row["base"], row["teacher"])] = row
    final, missing = [], 0
    for row in sorted(rows, key=lambda r: (r["arm"], r["qid"], r["base"])):
        for teacher in ("teacher1", "teacher2"):
            key = (row["arm"], row["qid"], row["base"], teacher)
            if key in prior:
                final.append({**prior[key], "ledger": "generation-run"})
            elif key in new:
                final.append({**new[key], "ledger": "scoring-run"})
            else:
                missing += 1
    complete = missing == 0 and not phase.stop_reason
    target = run_root / "scoring" / "final" / f"{generator}.json"
    if complete:
        write_once(target, {"protocol": "nea-formal8-plan-revisions-scoring-v1", "rows": final})
    return {"generator": generator, "judgments": len(final), "reused": sum(r["ledger"] == "generation-run" for r in final),
            "requested_rows": len(todo), "missing": missing, "stop_reason": phase.stop_reason, "complete": complete,
            "provider_refused_requests": len([r for r in load_refusals(run_root_of(phase)) if r["generator"] == generator])}


def analyze(run_root: Path) -> dict:
    from statistics import fmean
    from nea.evaluation.formal_analysis import _estimate, _round
    from nea.evaluation.statistics import BOOTSTRAP_REPLICATES, SEED
    quality, chars, revised, calls = {}, {}, {}, {}
    for generator in config.EXPERIMENT_GENERATORS:
        # The formal files may already hold the merged groups (spec 0046); take those from the ledger only.
        answers = [row for row in read(source_phase_dir(generator) / "answers.json")["rows"] if row["arm"] not in GROUPS]
        judgments = [row for row in read(source_phase_dir(generator) / "judgments.json")["rows"]
                     if row.get("comparison", "main") == "main" and row["arm"] not in GROUPS]
        phase_root = run_root / "operator-v2" / generator / "test"
        extra = read(phase_root / "supplement-answers.json")["rows"]
        answers += extra
        judgments += final_judgments(run_root, generator)
        for row in answers:
            if row.get("comparison", "main") == "main":
                chars.setdefault(row["arm"], []).append(len(row["answer"]))
        for row in extra:
            revised.setdefault(row["arm"], []).append(row["answer_sha256"] != row["parent_sha256"])
            for stage, count in row["revision"]["calls"].items():
                calls.setdefault(row["arm"], {}).setdefault(stage, 0)
                calls[row["arm"]][stage] += count
        for row in judgments:
            if row["status"] != "ok":
                raise RuntimeError("missing judgment")
            quality.setdefault((row["qid"], row["arm"]), []).append(row["scores"])
    if any(len(scores) != 8 for scores in quality.values()):
        raise RuntimeError("each question needs four cells x two judges per group")
    per = {key: {d: fmean(s[d] for s in scores) for d in DIMENSIONS} for key, scores in quality.items()}
    arms = sorted({arm for _, arm in per})
    qids = sorted({qid for qid, _ in per})

    def delta(left, right, dimension="overall_quality"):
        return {qid: per[qid, left][dimension] - per[qid, right][dimension] for qid in qids}

    reference = json.loads(next(row["value"] for row in csv.DictReader(
        (SOURCE / "paper-tables" / "test" / SOURCE_SHA / "table-06.csv").open(encoding="utf-8"))
        if row["record_type"] == "inference"))["primary"]["comparisons"]["F-L"]
    check = _estimate(delta("F", "L"))
    # The archived reference used target clusters; only its point effect is comparable.
    # New uncertainty is computed from whole-question resampling, never matched to old intervals.
    reproduced = _round(check["effect"]) == _round(reference["effect"])
    comparisons = {label: {"overall_quality": _estimate(delta(left, right)),
                           "answer_correctness": _estimate(delta(left, right, "answer_correctness"))}
                   for label, left, right in COMPARISONS}
    return {"protocol": "nea-formal8-plan-revisions-supplement-v1", "source_analysis_sha256": SOURCE_SHA,
        "bootstrap": {"unit": "qid", "replicates": BOOTSTRAP_REPLICATES, "seed": SEED},
        "n_qids": len(qids), "reference_check": {"comparison": "F-L", "reproduced": reproduced,
            "scope": "point_effect_only", "historical_uncertainty_comparable": False},
        "means": {arm: {d: _round(fmean(per[qid, arm][d] for qid in qids)) for d in DIMENSIONS} for arm in arms},
        "mean_chars": {arm: _round(fmean(values)) for arm, values in sorted(chars.items())},
        "comparisons": comparisons,
        "revision": {arm: {"positions": len(values), "revised": sum(values), "revised_rate": _round(sum(values) / len(values))}
                     for arm, values in sorted(revised.items())},
        "requests": {arm: {"by_stage": dict(sorted(stages.items())),
                           "per_answer": _round(sum(stages.values()) / len(revised[arm]))}
                     for arm, stages in sorted(calls.items())},
        "status": "ESTIMATED" if reproduced else "REFERENCE_MISMATCH",
        "note": "Supplementary comparisons; not part of the 1008 fixed sequence or its Holm family."}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("plan", "run", "score", "analyze", "merge"))
    parser.add_argument("--rescore-all", action="store_true", help="score: ignore earlier judgments (test only)")
    parser.add_argument("--run-id", default=RUN_ID)
    parser.add_argument("--sample-size", type=int, help="first-N smoke run in its own run-id; never a result")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--question-set", type=Path, default=ROOT / "knowledge_base/questions.jsonl")
    parser.add_argument("--knowledge-root", type=Path, default=ROOT / "knowledge_base")
    parser.add_argument("--prompts-dir", type=Path, default=ROOT / "prompts")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--generator", choices=("all", *config.EXPERIMENT_GENERATORS), default="all")
    parser.add_argument("--max-rounds", type=int, default=40)
    parser.add_argument("--generation-only", action="store_true")
    args = parser.parse_args(argv)
    args.generators = list(config.EXPERIMENT_GENERATORS) if args.generator == "all" else [args.generator]
    if args.sample_size is not None and args.run_id == RUN_ID:
        raise SystemExit("a sample run needs its own --run-id")
    from nea.runtime.identity import validate_run_root
    run_root = SUPPLEMENT_ROOT if args.run_id == RUN_ID else validate_run_root(ROOT, args.output_root, args.run_id)
    if args.command == "plan":
        from nea.pipeline.formal_entry import execution_plan
        n = 270 if args.sample_size is None else args.sample_size
        print(json.dumps({"run_root": str(run_root), "transport": execution_plan(**TRANSPORT),
            "groups": GROUPS, "positions_per_generator": n * 2 * len(GROUPS),
            "generation_requests_upper_bound_without_retries": n * 2 * 2 * 9,
            "judgment_requests_without_retries": n * 2 * len(GROUPS) * 2 * 2}, ensure_ascii=False, indent=1))
        return 0
    if args.command == "score":
        from nea.pipeline.entry import repository_cli_lock
        from nea.providers.pool import ProviderPool
        score_root = run_root / "scoring"
        score_root.mkdir(parents=True, exist_ok=True)
        import time
        from nea.runtime.formal_recovery import POLICY
        write_once(score_root / "recovery-policy.json", POLICY)
        args.run_id = args.run_id + "-scoring"
        summaries = []
        for _ in range(args.max_rounds):
            wait = max(0.0, max((json.loads(p.read_text(encoding="utf8"))["retry_at"]
                                 for p in (score_root / "supervision" / "cooldowns").glob("*.json")), default=0) - time.time())
            if wait:
                print(json.dumps({"event": "cooldown", "seconds": round(wait)}), flush=True)
                time.sleep(min(wait, 600))
                continue
            number = len(list((score_root / "supervision" / "rounds").glob("*.json"))) + 1
            write_once(score_root / "supervision" / "rounds" / f"{number:06d}.json",
                       {"round": number, "time": time.time(), "transport": TRANSPORT_SCORE, "generators": args.generators})
            with repository_cli_lock(ROOT):
                pool = ProviderPool(args.env_file, **TRANSPORT_SCORE)
                phases = cv0.build_phases(args, score_root, pool)
                with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="plan-score") as executor:
                    summaries = list(executor.map(lambda phase: score_generator(phase, run_root, args.rescore_all), phases))
            print(json.dumps({"round": number, "generators": summaries}, ensure_ascii=False), flush=True)
            if all(item["complete"] for item in summaries):
                return 0
            time.sleep(30)
        return 2
    if args.command == "merge":
        print(json.dumps(merge(run_root), ensure_ascii=False, indent=1))
        return 0
    if args.command == "analyze":
        first, second = analyze(run_root), analyze(run_root)
        if canonical_bytes(first) != canonical_bytes(second):
            raise RuntimeError("analysis is not deterministic")
        target = run_root / "analysis.json"
        if target.exists() and target.read_bytes() != canonical_bytes(first):
            raise RuntimeError("analysis.json exists with different content")
        if not target.exists():
            target.write_bytes(canonical_bytes(first))
        print(json.dumps({k: first[k] for k in ("status", "reference_check", "revision", "requests")}, ensure_ascii=False))
        return 0 if first["status"] == "ESTIMATED" else 2
    from nea.pipeline.entry import repository_cli_lock
    from nea.providers.pool import ProviderPool
    run_root.mkdir(parents=True, exist_ok=True)
    manifest = source_manifest()
    write_once(run_root / "source-1008-manifest.json", manifest)
    if read(run_root / "source-1008-manifest.json") != manifest:
        raise RuntimeError("formal inputs changed since the supplement was registered")
    import time
    from nea.runtime.formal_recovery import POLICY
    write_once(run_root / "recovery-policy.json", POLICY)
    rounds = run_root / "supervision" / "rounds"
    summaries, complete = [], False
    for _ in range(args.max_rounds):
        wait = max(0.0, max((json.loads(p.read_text(encoding="utf8"))["retry_at"]
                             for p in (run_root / "supervision" / "cooldowns").glob("*.json")), default=0) - time.time())
        if wait:
            print(json.dumps({"event": "cooldown", "seconds": round(wait)}), flush=True)
            time.sleep(min(wait, 600))
            continue
        number = len(list(rounds.glob("*.json"))) + 1
        write_once(rounds / f"{number:06d}.json", {"round": number, "time": time.time(), "transport": TRANSPORT,
                                                  "generators": args.generators, "generation_only": args.generation_only})
        with repository_cli_lock(ROOT):
            pool = ProviderPool(args.env_file, **TRANSPORT)
            phases = cv0.build_phases(args, run_root, pool)
            with ThreadPoolExecutor(max_workers=len(phases), thread_name_prefix="plan-rev") as executor:
                summaries = list(executor.map(lambda phase: run_generator(phase, args.generation_only), phases))
        complete = all(item["complete"] or (args.generation_only and item["generation_complete"]) for item in summaries)
        print(json.dumps({"round": number, "complete": complete, "generators": summaries}, ensure_ascii=False), flush=True)
        if complete:
            break
        time.sleep(30)
    return 0 if complete else 2


if __name__ == "__main__":
    raise SystemExit(main())
