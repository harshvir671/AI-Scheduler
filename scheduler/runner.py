

from typing import List, Dict
from scheduler.process_generator import Process, clone_processes
from scheduler.fcfs import run_fcfs
from scheduler.sjf import run_sjf_nonpreemptive, run_srtf
from scheduler.rr import run_rr
from scheduler.priority import run_priority
from scheduler.base import SchedulerResult


def run_all(processes: List[Process], rr_quantum: int = 4, use_aging: bool = False) -> Dict[str, SchedulerResult]:
    
    results = {
        "FCFS": run_fcfs(clone_processes(processes)),
        "SJF": run_sjf_nonpreemptive(clone_processes(processes)),
        "SRTF": run_srtf(clone_processes(processes)),
        "RR": run_rr(clone_processes(processes), quantum=rr_quantum),
        "Priority": run_priority(clone_processes(processes), use_aging=use_aging),
    }
    return results


def best_algorithm(results: Dict[str, SchedulerResult], metric: str = "avg_waiting_time") -> str:
    """Return the name of the algorithm with the lowest value of `metric`
    (avg_waiting_time or avg_turnaround_time -- lower is better for both)."""
    getter = {
        "avg_waiting_time": lambda r: r.avg_waiting_time(),
        "avg_turnaround_time": lambda r: r.avg_turnaround_time(),
        "avg_response_time": lambda r: r.avg_response_time(),
    }[metric]
    return min(results.items(), key=lambda kv: getter(kv[1]))[0]


if __name__ == "__main__":
    from scheduler.process_generator import generate_processes

    procs = generate_processes(8, seed=7)
    results = run_all(procs)
    for name, res in results.items():
        print(f"{name:10s}  avg_wait={res.avg_waiting_time():.2f}  "
              f"avg_turnaround={res.avg_turnaround_time():.2f}  "
              f"cpu_util={res.cpu_utilization():.1f}%")
    print("Best by waiting time:", best_algorithm(results))
