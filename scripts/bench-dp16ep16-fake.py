"""MiniMax DP16 decode benchmark using SGLang's fake KV transfer."""
import argparse
import ast
import contextlib
import contextvars
import csv
import gzip
import hashlib
import importlib
import inspect
import json
import math
import os
import random
import sys
import threading
import time
import types
import uuid
from pathlib import Path

DP_SIZE = 16
INPUT_LEN = 147456
OUTPUT_LEN = 1024
SEED = 42
PROMPTS = 64
REFERENCE_SHA256 = "28a24124753116c9cc88ea8bb4accb0968d1786cb346474c76cdb412a45c0dfe"
HISTORY = (
    ("warmup", 16), ("fresh", 256), ("prefix64", 64), ("prefix144", 144),
    ("prefix224", 224), ("conditioned-r1", 256), ("conditioned-r2", 256),
)
CAPTURE = contextvars.ContextVar("m31_metadata", default=None)
META_KEYS = (
    "prompt_tokens", "completion_tokens", "cached_tokens", "cached_tokens_details",
    "finish_reason", "spec_accept_length", "spec_accept_rate", "spec_verify_ct",
    "spec_correct_drafts_histogram", "spec_num_correct_drafts", "spec_num_proposed_drafts",
)


def cpu_client_environment():
    # This process is an HTTP client. Visibility changes never affect the server.
    os.environ["HIP_VISIBLE_DEVICES"] = ""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["SGLANG_IS_IN_CI"] = "0"


def load_native_serving():
    """Load the actual client without importing server GPU kernels for one literal.

    The installed client imports FAKE_BOOTSTRAP_HOST from a server module whose
    imports can initialize FP8 GPU kernels in a CPU-only client.
    Read that exact literal via AST, provide only this dependency during import,
    then restore sys.modules. Native request/metric functions remain untouched.
    """
    cpu_client_environment()
    import sglang
    name = "sglang.srt.disaggregation.utils"
    source = Path(sglang.__file__).parent / "srt/disaggregation/utils.py"
    tree = ast.parse(source.read_text())
    assignments = [node for node in tree.body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "FAKE_BOOTSTRAP_HOST"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise RuntimeError("FAKE_BOOTSTRAP_HOST dependency changed; review CPU import adapter")
    literal = ast.literal_eval(assignments[0].value)
    if not isinstance(literal, str):
        raise RuntimeError("Unexpected fake bootstrap literal")
    previous = sys.modules.get(name)
    lightweight = types.ModuleType(name)
    lightweight.FAKE_BOOTSTRAP_HOST = literal
    sys.modules[name] = lightweight
    try:
        return importlib.import_module("sglang.benchmark.serving")
    finally:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous


class MetadataJSON:
    """Observe the existing native JSON parse, with no second stream parser."""
    def __init__(self, original, record_first_token=False):
        self.original = original
        self.record_first_token = record_first_token

    def loads(self, value):
        result = self.original.loads(value)
        capture = CAPTURE.get()
        if capture is not None and isinstance(result, dict):
            meta = result.get("meta_info") or {}
            capture.update({key: meta[key] for key in META_KEYS if key in meta})
            if (
                self.record_first_token
                and (meta.get("completion_tokens") or 0) > 0
                and "first_completion_perf_counter" not in capture
            ):
                capture["first_completion_perf_counter"] = time.perf_counter()
        return result

    def __getattr__(self, name):
        return getattr(self.original, name)


def native_arguments(serving, argv):
    """Use the installed native CLI parser, retaining its version-specific defaults."""
    original_run, original_argv = serving.run_benchmark, sys.argv
    captured = []
    serving.run_benchmark = lambda args: captured.append(args)
    sys.argv = ["sglang.bench_serving"] + argv
    try:
        serving.cli_main()
    finally:
        serving.run_benchmark, sys.argv = original_run, original_argv
    return captured[0]


@contextlib.contextmanager
def native_adapter(serving, rows, captured, session):
    original_loader = serving.get_dataset
    original_json = serving.orjson
    original_request = serving.ASYNC_REQUEST_FUNCS["sglang"]
    original_requests = serving.requests
    original_factory = serving._create_bench_client_session
    if "orjson.loads" not in inspect.getsource(original_request):
        raise RuntimeError("Native request parser changed; use the cookbook's SGLang version")

    async def observe(request_func_input, pbar=None):
        meta = {"client_start_unix": time.time()}
        token = CAPTURE.set(meta)
        try:
            output = await original_request(request_func_input=request_func_input, pbar=pbar)
        finally:
            CAPTURE.reset(token)
        meta["client_end_unix"] = time.time()
        first_completion = meta.pop("first_completion_perf_counter", None)
        if output.success and output.ttft == 0.0 and first_completion is not None:
            output.ttft = first_completion - output.start_time
        captured.append((request_func_input, output, meta))
        return output

    def client_factory():
        import aiohttp
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     read_bufsize=10 * 1024**2, trust_env=False)

    serving.get_dataset = lambda *a, **kw: rows
    serving.orjson = MetadataJSON(original_json, record_first_token=True)
    serving.ASYNC_REQUEST_FUNCS["sglang"] = observe
    serving.requests = session
    serving._create_bench_client_session = client_factory
    try:
        yield
    finally:
        serving.get_dataset = original_loader
        serving.orjson = original_json
        serving.ASYNC_REQUEST_FUNCS["sglang"] = original_request
        serving.requests = original_requests
        serving._create_bench_client_session = original_factory


def save(directory, name, data):
    (directory / name).write_text(json.dumps(data, indent=2) + "\n")


def validate_inputs(data, require_reference):
    rows = data["rows"]
    if ((data["seed"], data["input_len"], data["output_len"], len(rows)) !=
            (SEED, INPUT_LEN, OUTPUT_LEN, PROMPTS)
            or any(len(row) != INPUT_LEN for row in rows)):
        raise ValueError("Expected 64 prompts with fixed 147456 input and 1024 output tokens, seed 42")
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if require_reference and digest != REFERENCE_SHA256:
        raise ValueError(f"Input digest {digest} differs from the cookbook reference")
    return digest


def frozen_inputs(path, require_reference):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        data = json.load(source)
    return data, validate_inputs(data, require_reference)


def prepare(args):
    cpu_client_environment()
    import numpy as np
    from transformers import AutoTokenizer
    from sglang.benchmark.datasets.random import sample_random_requests
    from sglang.benchmark.utils import is_file_valid_json

    if not is_file_valid_json(args.prepare_from_sharegpt):
        raise ValueError(f"Expected a local ShareGPT JSON file: {args.prepare_from_sharegpt}")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, local_files_only=True)
    random.seed(SEED)
    np.random.seed(SEED)
    source = sample_random_requests(input_len=INPUT_LEN, output_len=OUTPUT_LEN, num_prompts=PROMPTS,
                                    range_ratio=1.0, tokenizer=tokenizer, dataset_path=str(args.prepare_from_sharegpt),
                                    random_sample=True, return_text=False)
    data = {"seed": SEED, "input_len": INPUT_LEN, "output_len": OUTPUT_LEN,
            "rows": [row.prompt for row in source]}
    digest = validate_inputs(data, args.require_reference_inputs)
    args.inputs.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.inputs, "xt", encoding="utf-8") as target:
        json.dump(data, target)
    return data, digest


def monitor_loads(requests, base_url, destination, stop):
    with requests.Session() as session, destination.open("w", encoding="utf-8") as stream:
        session.trust_env = False
        while not stop.is_set():
            sample = {"sample_unix": time.time()}
            try:
                response = session.get(base_url + "/v1/loads?include=core,memory,spec", timeout=3)
                response.raise_for_status()
                sample.update(response.json())
            except (requests.RequestException, ValueError) as error:
                sample["error"] = str(error)
            stream.write(json.dumps(sample) + "\n")
            stream.flush()
            stop.wait(2)


def sampled_peaks(path, records):
    if not records:
        return {}
    start = min(row["client_start_unix"] for row in records)
    end = max(row["client_end_unix"] for row in records)
    running = []
    rank_peaks = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        ranks = sample.get("loads") or []
        if not (start <= sample["sample_unix"] <= end and len(ranks) == DP_SIZE):
            continue
        running.append(sum(row["num_running_reqs"] for row in ranks))
        for row in ranks:
            if row["max_total_num_tokens"]:
                rank = row["dp_rank"]
                used = row["num_used_tokens"] / row["max_total_num_tokens"]
                rank_peaks[rank] = max(rank_peaks.get(rank, 0), used)
    return {"running_peak": max(running, default=None), "ranks": rank_peaks,
            "full_samples": len(running)}


def csv_summary(output, cases):
    fieldnames = tuple(cases[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)


def run(args, batch, output_dir, data, digest, serving):
    import requests
    from sglang.benchmark.datasets.common import DatasetRow

    output_dir.mkdir()
    base_url = f"http://{args.host}:{args.port}"
    input_len, output_len = data["input_len"], data["output_len"]
    run_id = uuid.uuid4().hex
    active_ranks = min(DP_SIZE, batch)

    with requests.Session() as session:
        session.trust_env = False

        def fetch(path):
            response = session.get(f"{base_url}/{path}", timeout=15)
            response.raise_for_status()
            return response.json()

        info = fetch("server_info")
        save(output_dir, "server_info.json", info)
        expected = {
            "disaggregation_mode": "decode",
            "disaggregation_transfer_backend": "fake",
            "disable_radix_cache": True,
            "nnodes": 4,
            "attn_dp_size": DP_SIZE,
            "ep_size": DP_SIZE,
            "tp_size": DP_SIZE,
            "enable_eplb": True,
            "enable_hisparse": args.hisparse,
        }
        mismatches = {key: info.get(key) for key, value in expected.items() if info.get(key) != value}
        if mismatches:
            raise RuntimeError(f"Unexpected server settings: {mismatches}")

        before = fetch("v1/loads?include=core,memory,spec")
        loads = sorted(before["loads"], key=lambda row: row["dp_rank"])
        save(output_dir, "loads-before.json", before)
        if [row["dp_rank"] for row in loads] != list(range(DP_SIZE)):
            raise RuntimeError("Expected live KV capacity for all 16 DP ranks")
        if any(row["num_running_reqs"] or row["num_waiting_reqs"] for row in loads):
            raise RuntimeError("Wait until all DP ranks are idle before benchmarking")

        page = info["page_size"]
        reserve = (info.get("speculative_num_draft_tokens") or 0) if args.require_residency else 0
        request_tokens = ((input_len + output_len + reserve + page - 1) // page) * page
        config = info.get("hisparse_config")
        if isinstance(config, str):
            config = json.loads(config)
        ratio = float(config["host_to_device_ratio"]) if args.hisparse else 1.0
        rank_budget = []
        for rank, load in enumerate(loads):
            requests_on_rank = len(range(rank, batch, active_ranks)) if rank < active_ranks else 0
            rank_budget.append({
                "dp_rank": rank,
                "requests": requests_on_rank,
                "tokens_required_for_full_residency": requests_on_rank * request_tokens,
                "capacity_tokens": int(load["max_total_num_tokens"] * ratio) // page * page,
                "loads_reported_capacity_tokens": load["max_total_num_tokens"],
                "running_request_limit": load.get("max_running_requests"),
            })
        save(output_dir, "capacity-budget.json", rank_budget)
        for rank in rank_budget:
            if (
                rank["tokens_required_for_full_residency"] > rank["capacity_tokens"]
                or rank["requests"] > (rank["running_request_limit"] or batch)
            ):
                message = f"DP{rank['dp_rank']} cannot hold full batch: {rank}"
                if args.require_residency:
                    raise RuntimeError(message)
                print(message, flush=True)

        rows = [
            DatasetRow(
                prompt=data["rows"][index % len(data["rows"])],
                prompt_len=input_len,
                output_len=output_len,
                extra_request_body={
                    "rid": f"fake-prefill-{run_id}.{index}",
                    "routed_dp_rank": index % active_ranks,
                    "sampling_params": {
                        "temperature": 0, "top_p": 1, "top_k": 1, "sampling_seed": 42,
                        "ignore_eos": True, "max_new_tokens": output_len, "stream_interval": 1,
                    },
                },
            )
            for index in range(batch)
        ]
        model = args.model
        argv = [
            "--backend", "sglang", "--host", args.host, "--port", str(args.port),
            "--model", model, "--tokenizer", args.tokenizer or model,
            "--dataset-name", "random", "--tokenize-prompt", "--random-input-len", str(input_len),
            "--random-output-len", str(output_len), "--num-prompts", str(batch),
            "--max-concurrency", str(batch), "--request-rate", str(args.request_rate), "--seed", "42",
            "--warmup-requests", "0", "--ready-check-timeout-sec", "0", "--fake-prefill",
            "--output-details", "--output-file", str(output_dir / "native.jsonl"),
        ]
        save(output_dir, "measurement.json", {
            "mode": "fake-prefill/decode", "hisparse": args.hisparse, "bs": batch, "input_tokens": input_len,
            "output_tokens": output_len, "input_rows_sha256": digest,
            "prewarm_requests": 0, "request_rate": str(args.request_rate), "routing": "request index modulo active DP ranks",
            "argv": argv,
        })
        captured = []
        native_args = native_arguments(serving, argv)
        stop = threading.Event()
        load_log = output_dir / "loads.jsonl"
        worker = threading.Thread(target=monitor_loads, args=(requests, base_url, load_log, stop))
        worker.start()
        try:
            with native_adapter(serving, rows, captured, session):
                result = serving.run_benchmark(native_args)
        finally:
            stop.set()
            worker.join()

        records = []
        for request, response, meta in captured:
            body = request.extra_request_body
            generated_text = getattr(response, "generated_text", "")
            records.append({
                "generated_text_sha256": hashlib.sha256(generated_text.encode()).hexdigest() if generated_text else None,
                "replacement_chars": generated_text.count("\ufffd"),
                **meta, "id": int(body["rid"].rsplit(".", 1)[1]),
                "routed_dp_rank": body["routed_dp_rank"],
                "success": response.success, "error": response.error,
                "ttft_ms": response.ttft * 1000,
                "tpot_ms": (response.latency - response.ttft) * 1000 / (output_len - 1),
            })
        records.sort(key=lambda record: record["id"])
        save(output_dir, "requests.json", records)
        if records:
            save(output_dir, "measured-window.json", {
                "start": min(row["client_start_unix"] for row in records),
                "end": max(row["client_end_unix"] for row in records),
            })
        save(output_dir, "loads-after.json", fetch("v1/loads?include=core,memory,spec"))

    valid = (
        result["completed"] == batch
        and [row["id"] for row in records] == list(range(batch))
        and all(
            row["success"] and row.get("prompt_tokens") == input_len
            and row.get("completion_tokens") == output_len
            for row in records
        )
    )
    peaks = sampled_peaks(load_log, records)
    save(output_dir, "sampled-peaks.json", peaks)
    correct = sum((row.get("spec_num_correct_drafts") or 0) for row in records)
    verifies = sum((row.get("spec_verify_ct") or 0) for row in records)
    summary = {
        "mode": "fake-prefill/decode", "hisparse": args.hisparse, "bs": batch, "completed": result["completed"],
        "output_tps": result["output_throughput"], "mean_ttft_ms": result["mean_ttft_ms"],
        "mean_tpot_ms": result["mean_tpot_ms"], "verify_count": verifies,
        "correct_drafts": correct, "weighted_accept_length": 1 + correct / verifies if verifies else None,
        "p99_tpot_ms": result.get("p99_tpot_ms"),
        "p99_e2e_latency_ms": result.get("p99_e2e_latency_ms"),
        "observed_running_peak": peaks.get("running_peak"),
        "peak_kv_scheduler_fraction": max(peaks.get("ranks", {}).values(), default=None),
        "valid": valid,
    }
    save(output_dir, "validation.json", summary)
    print(summary, flush=True)
    if not valid:
        raise RuntimeError(f"BS{batch}: inspect {output_dir / 'requests.json'} for failed requests")
    return {"run": output_dir.name, **summary}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="Frozen .json.gz input file")
    parser.add_argument("--model", default=os.environ.get("M31_MODEL_PATH"))
    parser.add_argument("--tokenizer", default=os.environ.get("M31_TOKENIZER_PATH"))
    parser.add_argument("--prepare-from-sharegpt", type=Path, metavar="LOCAL_JSON")
    parser.add_argument("--bs", type=int, nargs="+")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=11011)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check-inputs", action="store_true")
    parser.add_argument("--require-reference-inputs", action="store_true")
    parser.add_argument("--hisparse", action="store_true")
    parser.add_argument("--require-residency", action="store_true")
    parser.add_argument("--history", action="store_true",
                        help="Run BS16, fresh BS256, BS64/144/224, then two conditioned BS256 rounds")
    parser.add_argument("--request-rate", type=float, default=math.inf)
    args = parser.parse_args()
    if not args.request_rate > 0:
        parser.error("--request-rate must be positive or inf")
    if args.prepare_from_sharegpt and not args.tokenizer:
        parser.error("--prepare-from-sharegpt requires --tokenizer")
    if args.prepare_from_sharegpt and args.check_inputs:
        parser.error("Use either preparation or input validation")
    if args.prepare_from_sharegpt or args.check_inputs:
        data, digest = prepare(args) if args.prepare_from_sharegpt else frozen_inputs(args.inputs, args.require_reference_inputs)
        print(f"Verified {len(data['rows'])} inputs, SHA256={digest}, cookbook_reference={digest == REFERENCE_SHA256}")
        return
    if args.output is None or not args.model:
        parser.error("Benchmarking requires --output and --model (or M31_MODEL_PATH)")
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.history:
        if not args.hisparse or args.bs is not None or args.repeats != 1:
            parser.error("--history requires --hisparse and defines the batch sequence and repeats")
        if not math.isinf(args.request_rate):
            parser.error("--history requires an unlimited request rate")
        args.require_residency = True
        cases = HISTORY
    else:
        sizes = args.bs or [64, 128, 144]
        limit = 256 if args.hisparse else 144
        if not all(1 <= value <= limit for value in sizes) or len(set(sizes)) != len(sizes):
            parser.error(f"--bs requires distinct integers in 1..{limit}")
        cases = [(f"bs{batch}-r{repeat}", batch)
                 for repeat in range(1, args.repeats + 1) for batch in sizes]
    data, digest = frozen_inputs(args.inputs, args.require_reference_inputs)
    serving = load_native_serving()
    args.output.mkdir(parents=True)
    summaries = []
    for name, batch in cases:
        summaries.append(run(args, batch, args.output / name, data, digest, serving))
        csv_summary(args.output, summaries)


if __name__ == "__main__":
    main()
