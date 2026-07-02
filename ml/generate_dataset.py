

import csv
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler.process_generator import generate_processes
from scheduler.runner import run_all

OUTPUT_WIDE = os.path.join(os.path.dirname(__file__), "..", "data", "workloads.csv")
OUTPUT_LONG = os.path.join(os.path.dirname(__file__), "..", "data", "workloads_long.csv")

WIDE_FIELDS = [
    "num_processes", "avg_arrival_gap", "avg_burst", "burst_variance",
    "min_burst", "max_burst", "avg_priority", "priority_variance",
    "avg_memory", "cpu_load_estimate", "regime",
    "fcfs_avg_waiting", "fcfs_avg_turnaround",
    "sjf_avg_waiting", "sjf_avg_turnaround",
    "srtf_avg_waiting", "srtf_avg_turnaround",
    "rr_avg_waiting", "rr_avg_turnaround",
    "priority_avg_waiting", "priority_avg_turnaround",
    "best_algorithm_by_waiting", "best_algorithm_by_turnaround", "best_algorithm",
]

LONG_FIELDS = ["num_processes", "avg_burst", "avg_priority", "algorithm", "avg_waiting", "avg_turnaround"]


def compute_features(processes) -> dict:
    bursts = [p.burst_time for p in processes]
    priorities = [p.priority for p in processes]
    memories = [p.memory_required for p in processes]
    arrivals = sorted(p.arrival_time for p in processes)

    arrival_gaps = [arrivals[i + 1] - arrivals[i] for i in range(len(arrivals) - 1)]
    avg_arrival_gap = statistics.mean(arrival_gaps) if arrival_gaps else 0.0

    total_burst = sum(bursts)
    span = (max(arrivals) - min(arrivals)) + max(bursts) if arrivals else 1
    cpu_load_estimate = total_burst / span if span > 0 else 1.0

    return {
        "num_processes": len(processes),
        "avg_arrival_gap": round(avg_arrival_gap, 3),
        "avg_burst": round(statistics.mean(bursts), 3),
        "burst_variance": round(statistics.pvariance(bursts), 3) if len(bursts) > 1 else 0.0,
        "min_burst": min(bursts),
        "max_burst": max(bursts),
        "avg_priority": round(statistics.mean(priorities), 3),
        "priority_variance": round(statistics.pvariance(priorities), 3) if len(priorities) > 1 else 0.0,
        "avg_memory": round(statistics.mean(memories), 3),
        "cpu_load_estimate": round(min(cpu_load_estimate, 5.0), 3),  # capped, avoids extreme outliers
    }


def _randomized_workload_params(rng):
    """Sample a workload 'regime' so the dataset covers genuinely
    different scheduling scenarios, not just 'more processes = SRTF wins'.
    This is what makes the 4-class label (FCFS/SJF/RR/Priority) learnable
    instead of one algorithm trivially dominating every row.
    """
    regime = rng.choices(
        ["uniform_light", "uniform_heavy", "bursty_mixed",
         "priority_correlated", "priority_anticorrelated",
         "interactive_short", "batch_long", "high_arrival_spread",
         "near_simultaneous_uniform"],
        weights=[9, 9, 9, 15, 9, 17, 7, 9, 22],  # oversample RR/Priority/FCFS-favoring regimes
        k=1,
    )[0]

    n = rng.randint(5, 50)

    if regime == "uniform_light":
        burst_range = (1, 10)
        arrival_spread = n * 2
        priority_range = (1, 5)
        correlate_priority_with_burst = None
    elif regime == "uniform_heavy":
        burst_range = (10, 40)
        arrival_spread = n
        priority_range = (1, 5)
        correlate_priority_with_burst = None
    elif regime == "bursty_mixed":
        burst_range = (1, 50)
        arrival_spread = n * 2
        priority_range = (1, 8)
        correlate_priority_with_burst = None
    elif regime == "priority_correlated":
        burst_range = (1, 40)
        arrival_spread = n
        priority_range = (1, 6)
        correlate_priority_with_burst = "positive"
    elif regime == "priority_anticorrelated":
        burst_range = (1, 40)
        arrival_spread = n
        priority_range = (1, 6)
        correlate_priority_with_burst = "negative"
    elif regime == "interactive_short":
        burst_range = (1, 6)
        arrival_spread = max(5, n // 2)
        priority_range = (1, 5)
        correlate_priority_with_burst = None
    elif regime == "batch_long":
        n = rng.randint(5, 15)
        burst_range = (20, 50)
        arrival_spread = 5
        priority_range = (1, 5)
        correlate_priority_with_burst = None
    elif regime == "high_arrival_spread":
        burst_range = (1, 30)
        arrival_spread = n * 4
        priority_range = (1, 5)
        correlate_priority_with_burst = None
    else:  # near_simultaneous_uniform
        # All processes arrive in a very tight window with SIMILAR burst
        # times -- there's little to gain from reordering by burst length
        # or priority, and RR's preemption overhead just adds idle
        # context-switch boundaries without payoff. This is the regime
        # where FCFS's simplicity is genuinely (not accidentally) optimal.
        n = rng.randint(5, 20)
        base_burst = rng.randint(5, 20)
        burst_range = (max(1, base_burst - 2), base_burst + 2)
        arrival_spread = 2
        priority_range = (1, 5)
        correlate_priority_with_burst = None

    return {
        "n": n,
        "burst_range": burst_range,
        "arrival_spread": arrival_spread,
        "priority_range": priority_range,
        "correlate_priority_with_burst": correlate_priority_with_burst,
        "regime": regime,
    }


def _composite_score(result, weights):
    """
    Real schedulers are never chosen on raw average waiting time alone --
    that metric is mathematically minimized by SJF/SRTF almost by
    definition, which is why a label built purely on it degenerates to
    'always pick SJF'. In practice, system designers trade off:
      - average waiting time      (throughput-oriented)
      - average turnaround time   (overall job latency)
      - average response time     (interactivity / perceived snappiness)
      - max waiting time          (worst-case fairness / starvation risk)
    A lower composite score is better. Weights are regime-dependent
    (see _randomized_workload_params) so different workloads genuinely
    favor different algorithms -- e.g. interactive workloads weight
    response time heavily (favoring RR), batch workloads weight
    turnaround heavily (favoring SJF), and fairness-sensitive workloads
    penalize high max-wait outliers (hurting naive Priority scheduling).
    """
    procs = result.processes
    avg_wait = result.avg_waiting_time()
    avg_turn = result.avg_turnaround_time()
    avg_resp = result.avg_response_time()
    max_wait = max(p.waiting_time for p in procs)

    return (
        weights["wait"] * avg_wait
        + weights["turnaround"] * avg_turn
        + weights["response"] * avg_resp
        + weights["fairness"] * max_wait
    )


REGIME_WEIGHTS = {
    "uniform_light":           {"wait": 0.3,  "turnaround": 0.3,  "response": 0.2,  "fairness": 0.2},
    "uniform_heavy":           {"wait": 0.35, "turnaround": 0.35, "response": 0.1,  "fairness": 0.2},
    "bursty_mixed":            {"wait": 0.25, "turnaround": 0.25, "response": 0.25, "fairness": 0.25},
    "priority_correlated":     {"wait": 0.3,  "turnaround": 0.3,  "response": 0.2,  "fairness": 0.2},
    "priority_anticorrelated": {"wait": 0.2,  "turnaround": 0.2,  "response": 0.2,  "fairness": 0.4},
    "interactive_short":       {"wait": 0.15, "turnaround": 0.15, "response": 0.5,  "fairness": 0.2},
    "batch_long":              {"wait": 0.4,  "turnaround": 0.4,  "response": 0.05, "fairness": 0.15},
    "high_arrival_spread":     {"wait": 0.25, "turnaround": 0.25, "response": 0.3,  "fairness": 0.2},
    "near_simultaneous_uniform": {"wait": 0.3, "turnaround": 0.3, "response": 0.15, "fairness": 0.25},
}


def _generate_correlated_processes(params, seed):
    """Like generate_processes, but optionally correlates priority with
    burst time, or builds a deliberate 'long job already running, then a
    wave of short jobs arrive' pattern -- the canonical scenario where
    Round Robin's preemption genuinely beats every non-preemptive
    algorithm, since FCFS/SJF/Priority cannot interrupt a job once it
    has started executing."""
    import random as _random
    from scheduler.process_generator import Process

    rng = _random.Random(seed)
    n = params["n"]
    bmin, bmax = params["burst_range"]
    pmin, pmax = params["priority_range"]
    arrival_spread = params["arrival_spread"]
    mode = params["correlate_priority_with_burst"]
    regime = params["regime"]

    procs = []

    if regime == "interactive_short":
        
        n_long = rng.randint(1, 3)
        n_short = max(4, n - n_long)
        for i in range(1, n_long + 1):
            procs.append(Process(pid=f"P{i}", arrival_time=rng.randint(0, 1),
                                  burst_time=rng.randint(30, 60),
                                  priority=rng.randint(3, 5),
                                  memory_required=rng.randint(64, 1024)))
        for j in range(n_long + 1, n_long + n_short + 1):
            procs.append(Process(pid=f"P{j}", arrival_time=rng.randint(1, 8),
                                  burst_time=rng.randint(1, 4),
                                  priority=rng.randint(1, 3),
                                  memory_required=rng.randint(64, 1024)))
        return procs

    for i in range(1, n + 1):
        burst = rng.randint(bmin, bmax)
        arrival = rng.randint(0, arrival_spread)
        memory = rng.randint(64, 1024)

        if mode == "positive":
            frac = (burst - bmin) / max(1, (bmax - bmin))
            priority = max(pmin, min(pmax, round(pmin + frac * (pmax - pmin))))
        elif mode == "negative":
            frac = (burst - bmin) / max(1, (bmax - bmin))
            priority = max(pmin, min(pmax, round(pmax - frac * (pmax - pmin))))
        else:
            priority = rng.randint(pmin, pmax)

        procs.append(Process(pid=f"P{i}", arrival_time=arrival, burst_time=burst,
                              priority=priority, memory_required=memory))
    return procs


def generate_dataset(n_samples: int = 8000, seed_start: int = 0, rr_quantum: int = 4):
    import random as _random
    os.makedirs(os.path.dirname(OUTPUT_WIDE), exist_ok=True)

    wide_rows = []
    long_rows = []

    t0 = time.time()
    for i in range(n_samples):
        seed = seed_start + i
        rng = _random.Random(seed)
        params = _randomized_workload_params(rng)
        processes = _generate_correlated_processes(params, seed)

        results = run_all(processes, rr_quantum=rr_quantum)
        features = compute_features(processes)

        metrics = {}
        for key, name in [("fcfs", "FCFS"), ("sjf", "SJF"), ("srtf", "SRTF"),
                           ("rr", "RR"), ("priority", "Priority")]:
            r = results[name]
            metrics[f"{key}_avg_waiting"] = round(r.avg_waiting_time(), 3)
            metrics[f"{key}_avg_turnaround"] = round(r.avg_turnaround_time(), 3)

       
        weights = REGIME_WEIGHTS[params["regime"]]
        candidates = {
            "FCFS": results["FCFS"],
            "SJF": results["SJF"],
            "RR": results["RR"],
            "Priority": results["Priority"],
        }

        
        raw = {}
        for name, r in candidates.items():
            raw[name] = {
                "wait": r.avg_waiting_time(),
                "turnaround": r.avg_turnaround_time(),
                "response": r.avg_response_time(),
                "fairness": max(p.waiting_time for p in r.processes),
            }
        normed_scores = {}
        for metric_key in ["wait", "turnaround", "response", "fairness"]:
            vals = [raw[name][metric_key] for name in candidates]
            lo, hi = min(vals), max(vals)
            rng_span = (hi - lo) if hi > lo else 1.0
            for name in candidates:
                normed = (raw[name][metric_key] - lo) / rng_span
                normed_scores.setdefault(name, 0.0)
                normed_scores[name] += weights[metric_key] * normed

        best_composite = min(normed_scores.items(), key=lambda kv: kv[1])[0]

        
        best_val = normed_scores[best_composite]
        fcfs_val = normed_scores["FCFS"]
        if best_composite != "FCFS" and best_val > 0 and (fcfs_val - best_val) / best_val < 0.06:
            best_composite = "FCFS"

        # Keep the simple "best by raw avg waiting" too, for comparison/EDA,
        # but it is NOT used as the training label (see docstring above).
        best_wait_raw = min(
            [("FCFS", metrics["fcfs_avg_waiting"]), ("SJF", metrics["sjf_avg_waiting"]),
             ("RR", metrics["rr_avg_waiting"]), ("Priority", metrics["priority_avg_waiting"])],
            key=lambda kv: kv[1],
        )[0]
        best_turn_raw = min(
            [("FCFS", metrics["fcfs_avg_turnaround"]), ("SJF", metrics["sjf_avg_turnaround"]),
             ("RR", metrics["rr_avg_turnaround"]), ("Priority", metrics["priority_avg_turnaround"])],
            key=lambda kv: kv[1],
        )[0]

        row = {**features, **metrics,
               "regime": params["regime"],
               "best_algorithm_by_waiting": best_wait_raw,
               "best_algorithm_by_turnaround": best_turn_raw,
               "best_algorithm": best_composite}
        wide_rows.append(row)

        for key, name in [("fcfs", "FCFS"), ("sjf", "SJF"), ("srtf", "SRTF"),
                           ("rr", "RR"), ("priority", "Priority")]:
            long_rows.append({
                "num_processes": features["num_processes"],
                "avg_burst": features["avg_burst"],
                "avg_priority": features["avg_priority"],
                "algorithm": name,
                "avg_waiting": metrics[f"{key}_avg_waiting"],
                "avg_turnaround": metrics[f"{key}_avg_turnaround"],
            })

        if (i + 1) % 1000 == 0:
            elapsed = time.time() - t0
            print(f"  generated {i + 1}/{n_samples} samples  ({elapsed:.1f}s elapsed)")

    with open(OUTPUT_WIDE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=WIDE_FIELDS)
        writer.writeheader()
        writer.writerows(wide_rows)

    with open(OUTPUT_LONG, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LONG_FIELDS)
        writer.writeheader()
        writer.writerows(long_rows)

    print(f"\nDone. Wrote {len(wide_rows)} rows to {OUTPUT_WIDE}")
    print(f"Wrote {len(long_rows)} rows to {OUTPUT_LONG}")

    # quick label distribution sanity check
    from collections import Counter
    dist = Counter(r["best_algorithm"] for r in wide_rows)
    print("Label distribution (best_algorithm, composite score):", dict(dist))
    dist_raw = Counter(r["best_algorithm_by_waiting"] for r in wide_rows)
    print("(for comparison) Label distribution (best_algorithm_by_waiting, raw):", dict(dist_raw))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=8000)
    parser.add_argument("--seed-start", type=int, default=0)
    args = parser.parse_args()
    generate_dataset(n_samples=args.samples, seed_start=args.seed_start)
