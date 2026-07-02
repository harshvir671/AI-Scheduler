

from typing import List
from scheduler.process_generator import Process
from scheduler.base import SchedulerResult, GanttSlice, finalize_metrics


def run_fcfs(processes: List[Process]) -> SchedulerResult:
    procs = sorted(processes, key=lambda p: (p.arrival_time, p.pid))
    gantt: List[GanttSlice] = []
    current_time = 0
    idle_time = 0

    for p in procs:
        if current_time < p.arrival_time:
            idle_time += p.arrival_time - current_time
            current_time = p.arrival_time

        p.start_time = current_time
        p.response_time = p.start_time - p.arrival_time
        end_time = current_time + p.burst_time
        gantt.append(GanttSlice(pid=p.pid, start=current_time, end=end_time))
        current_time = end_time
        p.completion_time = current_time

    finalize_metrics(procs)
    return SchedulerResult(
        algorithm="FCFS",
        processes=procs,
        gantt=gantt,
        total_time=current_time,
        idle_time=idle_time,
    )
