"""Measure MiniMax on a real SGLang PD router with the frozen RSC prompts."""
import argparse
import concurrent.futures
import csv
import gzip
import hashlib
import json
import statistics
import sys
import threading
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener
from uuid import uuid4


PREFIX_LEN = 131072
INPUT_LEN = 147456
OUTPUT_LENGTHS = {"prefill": 1, "decode": 1024}


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def request_json(opener, url):
    with opener.open(url, timeout=10) as response:
        return json.load(response)


def generate(url, tokens, output_len, salt, request_id, barrier=None):
    payload = {
        "input_ids": tokens,
        "cache_salt": salt,
        "rid": request_id,
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "top_p": 1,
            "top_k": 1,
            "ignore_eos": True,
            "max_new_tokens": output_len,
            "stream_interval": 1,
            "sampling_seed": 42,
        },
    }
    request = Request(
        url + "/generate",
        json.dumps(payload).encode(),
        {"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    opener = build_opener(ProxyHandler({}))
    first = None
    completion = None
    done = False
    meta = {}
    if barrier is not None:
        barrier.wait(180)
    start = time.monotonic()
    with opener.open(request, timeout=3600) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = line[5:].strip()
            if event == b"[DONE]":
                done = True
                break
            chunk = json.loads(event)
            if "error" in chunk:
                raise RuntimeError(chunk)
            event_meta = chunk.get("meta_info", {})
            count = event_meta.get("completion_tokens", 0)
            if count:
                received = time.monotonic()
                if first is None:
                    first = received
                completion = received
                meta.update(event_meta)
    if not done or first is None or meta.get("prompt_tokens") != len(tokens):
        raise RuntimeError({"done": done, "first": first, "meta": meta})
    if meta.get("completion_tokens") != output_len:
        raise RuntimeError({"expected_output": output_len, "meta": meta})
    finish = meta.get("finish_reason", {})
    if not isinstance(finish, dict) or finish.get("type") != "length":
        raise RuntimeError({"expected_finish": "length", "meta": meta})
    return {
        "request_id": request_id,
        "start": start,
        "first": first,
        "completion": completion,
        "ttft_ms": 1000 * (first - start),
        "tpot_ms": 1000 * (completion - first) / (output_len - 1) if output_len > 1 else None,
        "meta": meta,
    }


def sample_loads(args, output, done, peaks):
    opener = build_opener(ProxyHandler({}))
    with (output / "loads.jsonl").open("w") as stream:
        while not done.is_set():
            for role in OUTPUT_LENGTHS:
                record = {"unix_time": time.time(), "role": role}
                try:
                    url = args.prefill if role == "prefill" else args.decode
                    loads = request_json(opener, url + "/v1/loads?include=core")["loads"]
                    running = sum(rank["num_running_reqs"] for rank in loads)
                    waiting = sum(rank["num_waiting_reqs"] for rank in loads)
                    record.update(running=running, waiting=waiting, ranks=loads)
                    prefix = role + "_peak_"
                    for name, value in (("running", running), ("waiting", waiting)):
                        key = prefix + name
                        peaks[key] = max(value, peaks.get(key, value))
                except Exception as exc:
                    record["error"] = str(exc)
                stream.write(json.dumps(record) + "\n")
                stream.flush()
            done.wait(2)


def measure(args, rows, role, size, output):
    output.mkdir(parents=True)
    salt = "m31-pd-" + uuid4().hex
    prompts = rows[:size]
    output_len = OUTPUT_LENGTHS[role]
    if role == "prefill":
        prefix = rows[0][:PREFIX_LEN]
        prompts = [prefix + prompt[PREFIX_LEN:] for prompt in prompts]
        if len({tuple(prompt[PREFIX_LEN:PREFIX_LEN + 128]) for prompt in prompts}) != size:
            raise RuntimeError("Prefill prompts must diverge on the first new KV page.")
        warmup = generate(args.router, prefix, 1, salt, salt + "-warm")
        write_json(output / "warmup.json", warmup)

    barrier = threading.Barrier(size)
    results = []
    done = threading.Event()
    peaks = {}
    sampler = threading.Thread(target=sample_loads, args=(args, output, done, peaks), daemon=True)
    sampler.start()
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=size) as pool:
            requests = {
                pool.submit(
                    generate, args.router, prompt, output_len, salt, f"{salt}-{index}", barrier
                ): index
                for index, prompt in enumerate(prompts)
            }
            for pending in concurrent.futures.as_completed(requests):
                index = requests[pending]
                try:
                    result = {"index": index, **pending.result()}
                except Exception as exc:
                    result = {"index": index, "error": str(exc)}
                results.append(result)
                write_json(output / "requests.json", sorted(results, key=lambda row: row["index"]))
    finally:
        done.set()
        sampler.join()

    valid = [row for row in results if "error" not in row]
    summary = {
        "stage": role,
        "bs": size,
        "completed": len(valid),
        "input_tokens": INPUT_LEN,
        "output_tokens_per_request": output_len,
        "shared_prefix_tokens": PREFIX_LEN if role == "prefill" else 0,
        "status": "passed" if len(valid) == size else "failed",
    }
    if len(valid) == size:
        end = max(row["completion"] for row in valid)
        duration = end - min(row["start"] for row in valid)
        verify = sum(row["meta"].get("spec_verify_ct", 0) for row in valid)
        accepted = sum(row["meta"].get("spec_num_correct_drafts", 0) for row in valid)
        summary.update(
            ttft_ms=statistics.mean(row["ttft_ms"] for row in valid),
            tpot_ms=statistics.mean(row["tpot_ms"] for row in valid) if role == "decode" else None,
            output_tok_s=size * output_len / duration,
            decode_window_tok_s=(
                size * (output_len - 1) / (end - min(row["first"] for row in valid))
                if role == "decode" else None
            ),
            reported_cache_min=min(row["meta"].get("cached_tokens", 0) for row in valid),
            verify_count=verify,
            weighted_acceptance=1 + accepted / verify if verify else None,
        )
        if role == "prefill" and summary["reported_cache_min"] != PREFIX_LEN:
            summary.update(status="failed", error="Prefill did not hit the expected shared prefix.")
    summary.update(peaks)
    write_json(output / "summary.json", summary)
    return summary


def select_prompts(frozen, count):
    """Choose distinct first new KV pages while retaining each source's tokens."""
    prompts = []
    pages = set()
    for index in range(count):
        row = frozen[index % len(frozen)]
        for offset in range(1 if index >= len(frozen) else 0, 2048):
            page = tuple(row[PREFIX_LEN + offset:PREFIX_LEN + offset + 128])
            if page in pages:
                continue
            pages.add(page)
            prompts.append(row[offset:] + row[:offset] if offset else row)
            break
        else:
            raise RuntimeError("Could not derive a unique first new KV page.")
    return prompts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--router", required=True)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bs", type=int, nargs="+", default=[1, 2, 4, 8])
    args = parser.parse_args()
    if any(not 1 <= size <= 128 for size in args.bs) or len(set(args.bs)) != len(args.bs):
        parser.error("Use distinct batch sizes in 1..128.")
    with gzip.open(args.input, "rt") as stream:
        dataset = json.load(stream)
    rows = dataset["rows"]
    protocol = json.loads((args.input.parent.parent / "protocol.json").read_text())
    digest = hashlib.sha256(json.dumps(rows).encode()).hexdigest()
    if (dataset["seed"] != 42 or len(rows) != 64 or any(len(row) != INPUT_LEN for row in rows)
            or digest != protocol["input"]["rows_sha256"]):
        raise RuntimeError("Unexpected RSC input dataset.")
    rows = select_prompts(rows, max(args.bs))
    args.output.mkdir()
    opener = build_opener(ProxyHandler({}))
    info = {
        role: request_json(opener, url + "/server_info")
        for role, url in (("prefill", args.prefill), ("decode", args.decode))
    }
    write_json(args.output / "server-info.json", info)
    if info["prefill"].get("disaggregation_mode") != "prefill":
        raise RuntimeError("Expected a disaggregated Prefill worker.")
    if (info["decode"].get("disaggregation_mode") != "decode"
            or info["decode"].get("disaggregation_transfer_backend") != "mooncake"
            or not info["decode"].get("enable_hisparse")):
        raise RuntimeError("Expected a disaggregated Mooncake HiSparse Decode worker.")
    write_json(args.output / "protocol.json", {
        "dataset_sha256": digest, "batch_sizes": args.bs, "router": args.router,
        "expanded_rows_sha256": hashlib.sha256(json.dumps(rows).encode()).hexdigest(),
        "transfer_backend": "mooncake", "seed": dataset["seed"],
        "note": "Decode outputs use independent prompts; Prefill TTFT includes real KV transfer.",
    })
    columns = [
        "stage", "bs", "status", "completed", "input_tokens", "output_tokens_per_request",
        "shared_prefix_tokens", "ttft_ms", "tpot_ms", "output_tok_s", "decode_window_tok_s",
        "reported_cache_min", "verify_count", "weighted_acceptance", "error",
        "prefill_peak_running", "prefill_peak_waiting", "decode_peak_running", "decode_peak_waiting",
    ]
    passed = True
    with (args.output / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for role in OUTPUT_LENGTHS:
            for size in args.bs:
                target = args.output / role / f"bs{size}"
                print(f"Running {role} BS{size}", flush=True)
                try:
                    summary = measure(args, rows, role, size, target)
                except Exception as exc:
                    summary = {"stage": role, "bs": size, "status": "failed", "error": str(exc)}
                    target.mkdir(parents=True, exist_ok=True)
                    write_json(target / "summary.json", summary)
                writer.writerow(summary)
                stream.flush()
                print(json.dumps(summary), flush=True)
                passed &= summary["status"] == "passed"
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
