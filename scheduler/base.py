

from dataclasses import dataclass, field
from typing import List, Dict, Any
from scheduler.process_generator import Process


@dataclass
class GanttSlice:
    pid: str
    start: int
    end: int


@dataclass
class SchedulerResult:
    algorithm: str
    processes: List[Process]
    gantt: List[GanttSlice]
    total_time: int
    idle_time: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "total_time": self.total_time,
            "idle_time": self.idle_time,
            "cpu_utilization": round(self.cpu_utilization(), 2),
            "avg_waiting_time": round(self.avg_waiting_time(), 2),
            "avg_turnaround_time": round(self.avg_turnaround_time(), 2),
            "avg_response_time": round(self.avg_response_time(), 2),
            "gantt": [{"pid": g.pid, "start": g.start, "end": g.end} for g in self.gantt],
            "processes": [
                {
                    "pid": p.pid,
                    "arrival_time": p.arrival_time,
                    "burst_time": p.burst_time,
                    "priority": p.priority,
                    "memory_required": p.memory_required,
                    "start_time": p.start_time,
                    "completion_time": p.completion_time,
                    "waiting_time": p.waiting_time,
                    "turnaround_time": p.turnaround_time,
                    "response_time": p.response_time,
                }
                for p in self.processes
            ],
        }

    def avg_waiting_time(self) -> float:
        return sum(p.waiting_time for p in self.processes) / len(self.processes)

    def avg_turnaround_time(self) -> float:
        return sum(p.turnaround_time for p in self.processes) / len(self.processes)

    def avg_response_time(self) -> float:
        return sum(p.response_time for p in self.processes) / len(self.processes)

    def cpu_utilization(self) -> float:
        if self.total_time == 0:
            return 0.0
        busy = self.total_time - self.idle_time
        return (busy / self.total_time) * 100.0


def finalize_metrics(processes: List[Process]):
    """Compute waiting/turnaround time for each process after
    completion_time and start_time have been set by the algorithm."""
    for p in processes:
        p.turnaround_time = p.completion_time - p.arrival_time
        p.waiting_time = p.turnaround_time - p.burst_time
        if p.response_time == -1:
            p.response_time = p.start_time - p.arrival_time


def merge_gantt(slices: List[GanttSlice]) -> List[GanttSlice]:
    """Merge consecutive slices of the same PID (useful for RR output
    where a process might run back-to-back across quantum boundaries
    due to no other ready process)."""
    if not slices:
        return []
    merged = [slices[0]]
    for s in slices[1:]:
        last = merged[-1]
        if s.pid == last.pid and s.start == last.end:
            last.end = s.end
        else:
            merged.append(s)
    return merged
