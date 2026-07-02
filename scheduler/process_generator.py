

import random
from dataclasses import dataclass, field
from typing import List


@dataclass
class Process:
    pid: str
    arrival_time: int
    burst_time: int
    priority: int            
    memory_required: int    

    remaining_time: int = field(init=False)
    start_time: int = field(default=-1, init=False)     
    completion_time: int = field(default=-1, init=False)
    waiting_time: int = field(default=0, init=False)
    turnaround_time: int = field(default=0, init=False)
    response_time: int = field(default=-1, init=False)

    def __post_init__(self):
        self.remaining_time = self.burst_time

    def reset(self):
        """Reset simulation state so the same process set can be reused
        across multiple algorithms without regenerating random data."""
        self.remaining_time = self.burst_time
        self.start_time = -1
        self.completion_time = -1
        self.waiting_time = 0
        self.turnaround_time = 0
        self.response_time = -1


def generate_processes(
    n: int,
    max_arrival: int = 20,
    min_burst: int = 1,
    max_burst: int = 20,
    min_priority: int = 1,
    max_priority: int = 5,
    min_memory: int = 64,
    max_memory: int = 1024,
    seed: int = None,
) -> List[Process]:
    """Generate `n` random processes."""
    rng = random.Random(seed)
    processes = []
    for i in range(1, n + 1):
        p = Process(
            pid=f"P{i}",
            arrival_time=rng.randint(0, max_arrival),
            burst_time=rng.randint(min_burst, max_burst),
            priority=rng.randint(min_priority, max_priority),
            memory_required=rng.randint(min_memory, max_memory),
        )
        processes.append(p)
    return processes


def clone_processes(processes: List[Process]) -> List[Process]:
    
    return [
        Process(
            pid=p.pid,
            arrival_time=p.arrival_time,
            burst_time=p.burst_time,
            priority=p.priority,
            memory_required=p.memory_required,
        )
        for p in processes
    ]


if __name__ == "__main__":
    procs = generate_processes(5, seed=42)
    for p in procs:
        print(p.pid, p.arrival_time, p.burst_time, p.priority, p.memory_required)
