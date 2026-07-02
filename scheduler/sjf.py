

import heapq
from typing import List
from scheduler.process_generator import Process
from scheduler.base import SchedulerResult, GanttSlice, finalize_metrics, merge_gantt


def run_sjf_nonpreemptive(processes: List[Process]) -> SchedulerResult:
    procs = list(processes)
    n = len(procs)
    completed = 0
    current_time = 0
    idle_time = 0
    gantt: List[GanttSlice] = []
    remaining = procs[:]

    while completed < n:
        ready = [p for p in remaining if p.arrival_time <= current_time]
        if not ready:
            next_arrival = min(p.arrival_time for p in remaining)
            idle_time += next_arrival - current_time
            current_time = next_arrival
            continue

        p = min(ready, key=lambda x: (x.burst_time, x.arrival_time, x.pid))

        p.start_time = current_time
        p.response_time = p.start_time - p.arrival_time
        end_time = current_time + p.burst_time
        gantt.append(GanttSlice(pid=p.pid, start=current_time, end=end_time))
        current_time = end_time
        p.completion_time = current_time

        remaining.remove(p)
        completed += 1

    finalize_metrics(procs)
    return SchedulerResult(
        algorithm="SJF",
        processes=procs,
        gantt=gantt,
        total_time=current_time,
        idle_time=idle_time,
    )


def run_srtf(processes: List[Process]) -> SchedulerResult:
    """Preemptive SJF: always run the process with the least remaining time."""
    procs = list(processes)
    n = len(procs)
    completed = 0
    current_time = 0
    idle_time = 0
    raw_gantt: List[GanttSlice] = []

    remaining = {p.pid: p.burst_time for p in procs}
    by_pid = {p.pid: p for p in procs}

    while completed < n:
        ready = [p for p in procs if p.arrival_time <= current_time and remaining[p.pid] > 0]
        if not ready:
            future = [p.arrival_time for p in procs if remaining[p.pid] > 0]
            next_arrival = min(future)
            idle_time += next_arrival - current_time
            current_time = next_arrival
            continue

        p = min(ready, key=lambda x: (remaining[x.pid], x.arrival_time, x.pid))

        if p.start_time == -1:
            p.start_time = current_time
            p.response_time = current_time - p.arrival_time

        raw_gantt.append(GanttSlice(pid=p.pid, start=current_time, end=current_time + 1))
        remaining[p.pid] -= 1
        current_time += 1

        if remaining[p.pid] == 0:
            p.completion_time = current_time
            completed += 1

    gantt = merge_gantt(raw_gantt)
    finalize_metrics(procs)
    return SchedulerResult(
        algorithm="SRTF",
        processes=procs,
        gantt=gantt,
        total_time=current_time,
        idle_time=idle_time,
    )
