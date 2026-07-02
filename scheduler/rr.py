
from collections import deque
from typing import List
from scheduler.process_generator import Process
from scheduler.base import SchedulerResult, GanttSlice, finalize_metrics, merge_gantt


def run_rr(processes: List[Process], quantum: int = 4) -> SchedulerResult:
    procs = list(processes)
    procs_by_arrival = sorted(procs, key=lambda p: (p.arrival_time, p.pid))

    queue = deque()
    raw_gantt: List[GanttSlice] = []
    current_time = 0
    idle_time = 0
    completed = 0
    n = len(procs)

    arrival_idx = 0
    # seed time at first arrival
    if procs_by_arrival:
        current_time = procs_by_arrival[0].arrival_time

    def admit_arrivals(up_to_time):
        nonlocal arrival_idx
        while arrival_idx < n and procs_by_arrival[arrival_idx].arrival_time <= up_to_time:
            queue.append(procs_by_arrival[arrival_idx])
            arrival_idx += 1

    admit_arrivals(current_time)

    while completed < n:
        if not queue:
            if arrival_idx < n:
                next_arrival = procs_by_arrival[arrival_idx].arrival_time
                idle_time += next_arrival - current_time
                current_time = next_arrival
                admit_arrivals(current_time)
                continue
            else:
                break

        p = queue.popleft()

        if p.start_time == -1:
            p.start_time = current_time
            p.response_time = current_time - p.arrival_time

        run_for = min(quantum, p.remaining_time)
        slice_start = current_time
        current_time += run_for
        p.remaining_time -= run_for
        raw_gantt.append(GanttSlice(pid=p.pid, start=slice_start, end=current_time))

        
        admit_arrivals(current_time)

        if p.remaining_time == 0:
            p.completion_time = current_time
            completed += 1
        else:
            queue.append(p)

    gantt = merge_gantt(raw_gantt)
    finalize_metrics(procs)
    return SchedulerResult(
        algorithm=f"Round Robin (q={quantum})",
        processes=procs,
        gantt=gantt,
        total_time=current_time,
        idle_time=idle_time,
    )
