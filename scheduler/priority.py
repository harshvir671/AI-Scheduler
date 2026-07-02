

from typing import List
from scheduler.process_generator import Process
from scheduler.base import SchedulerResult, GanttSlice, finalize_metrics


def run_priority(
    processes: List[Process],
    use_aging: bool = False,
    aging_interval: int = 5,
    aging_boost: int = 1,
) -> SchedulerResult:
    """
    use_aging: if True, every `aging_interval` time units a waiting
    process's effective priority improves by `aging_boost`, preventing
    indefinite starvation of low-priority jobs.
    """
    procs = list(processes)
    n = len(procs)
    completed = 0
    current_time = 0
    idle_time = 0
    gantt: List[GanttSlice] = []
    remaining = procs[:]

    # effective_priority tracks aged priority per pid; lower = runs first
    effective_priority = {p.pid: p.priority for p in procs}
    last_seen_waiting = {p.pid: 0 for p in procs}  # time since last aging bump

    while completed < n:
        ready = [p for p in remaining if p.arrival_time <= current_time]
        if not ready:
            next_arrival = min(p.arrival_time for p in remaining)
            idle_time += next_arrival - current_time
            current_time = next_arrival
            continue

        if use_aging:
            for p in ready:
                waited = current_time - p.arrival_time
                bumps_due = waited // aging_interval
                if bumps_due > last_seen_waiting[p.pid]:
                    
                    effective_priority[p.pid] = max(1, p.priority - bumps_due * aging_boost)
                    last_seen_waiting[p.pid] = bumps_due

           
            p = min(
                ready,
                key=lambda x: (effective_priority[x.pid], -(current_time - x.arrival_time), x.pid),
            )
        else:
            p = min(
                ready,
                key=lambda x: (effective_priority[x.pid], x.arrival_time, x.pid),
            )

        p.start_time = current_time
        p.response_time = p.start_time - p.arrival_time
        end_time = current_time + p.burst_time
        gantt.append(GanttSlice(pid=p.pid, start=current_time, end=end_time))
        current_time = end_time
        p.completion_time = current_time

        remaining.remove(p)
        completed += 1

    finalize_metrics(procs)

    result = SchedulerResult(
        algorithm="Priority" + (" + Aging" if use_aging else ""),
        processes=procs,
        gantt=gantt,
        total_time=current_time,
        idle_time=idle_time,
    )
    return result


def detect_starvation(result: SchedulerResult, threshold_multiplier: float = 1.8) -> List[dict]:
    
    flagged = []
    by_priority = sorted(set(p.priority for p in result.processes))
    if len(by_priority) < 2:
        return flagged  # everyone has the same priority -- no relative starvation possible

    for p in result.processes:
        higher = [q for q in result.processes if q.priority < p.priority]
        if not higher:
            continue  # this IS the highest priority tier, can't starve relative to anyone
        higher_avg_wait = sum(q.waiting_time for q in higher) / len(higher)
        if higher_avg_wait <= 0:
            continue
        ratio = p.waiting_time / higher_avg_wait
        if ratio > threshold_multiplier and p.waiting_time > 0:
            flagged.append({
                "pid": p.pid,
                "priority": p.priority,
                "waiting_time": p.waiting_time,
                "higher_priority_avg_wait": round(higher_avg_wait, 2),
                "severity": round(ratio, 2),
            })

    flagged.sort(key=lambda x: -x["severity"])
    return flagged
