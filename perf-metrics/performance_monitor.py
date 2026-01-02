from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import time
import json
import os
import psutil
import torch
@dataclass
class ComputationalMetrics:
    operation: str
    wall_time: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: float
    tokens_processed: int
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> Dict:
        return {
            'operation': self.operation,
            'wall_time_seconds': round(self.wall_time, 4),
            'cpu_percent': round(self.cpu_percent, 2),
            'memory_mb': round(self.gpu_memory_mb, 2),
            'tokens_processed': self.tokens_processed,
            'timestamp': self.timestamp
        }

class PerfomanceMonitor:
    def __init__(self):
        self.metrics = List[ComputationalMetrics] = []
        self.process = psutil.Process()
        self.current_operation: Optional[str] = None
        self.start_time: Optional[float] = None
        self.start_memory: Optional[float] = None
        self.start_gpu_memory: Optional[float] = None

    def start_operation(self, operation_name: str):
        self.current_operation = operation_name
        self.start_time = time.time()
        self.start_memory = self.process.memory_info().rss / (1024 ** 2) # Bytes para mb
        if torch.cuda.is_available():
            self.start_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
        else:
            self.start_gpu_memory = 0.0
        
    def end_operation(self, tokens: int = 0) -> ComputationalMetrics:
        if self.current_operation is None:
            raise RuntimeError("Call start_operation() first")
        elapsed_time = time.time() - self.start_time
        end_memory = self.process.memory_info().rss / (1024 ** 2)
        memory_used = end_memory - self.start_memory
        if torch.cuda.is_available():
            end_gpu_memory = torch.cuda.memory_allocated() / (1024 ** 2)
            gpu_memory_used = end_gpu_memory = self.start_gpu_memory
        else:
            gpu_memory_used = 0.0
        
        cpu_percent = self.process.cpu_percent(interval=0.1)
        metric = ComputationalMetrics(
            operation=self.current_operation,
            wall_time=elapsed_time,
            cpu_percent=cpu_percent,
            memory_mb=memory_used,
            gpu_memory_mb=gpu_memory_used,
            tokens_processed=tokens
        )

        self.metrics.append(metric)
        self.current_operation = None
        self.start_time = None
        self.start_memory = None
        self.start_gpu_memory = None

        return metric
    
    def get_summary(self) -> Dict:
        if not self.metrics:
            return {"error": "No metrics"}
        operations = {}
        for metric in self.metrics:
            op = metric.operation
            if op not in operations:
                operations[op] = {
                    'count': 0,
                    'total_time': 0.0,
                    'total_memory': 0.0,
                    'total_gpu_memory': 0.0,
                    'total_tokens': 0
                }
            operations[op]['count'] += 1
            operations[op]['total_time'] += metric.wall_time
            operations[op]['total_memory'] += metric.memory_mb
            operations[op]['total_gpu_memory'] += metric.gpu_memory_mb
            operations[op]['total_tokens'] += metric.tokens_processed

        summary = {}
        for op, data in operations.items():
            summary[op] = {
                'count': data['count'],
                'avg_time_seconds': round(data['total_time'] / data['count'], 4),
                'total_time_seconds': round(data['total_time'], 4),
                'avg_memory_mb': round(data['total_memory'] / data['count'], 2),
                'avg_gpu_memory_mb': round(data['total_gpu_memory'] / data['count'], 2),
                'total_tokens': data['total_tokens']
            }
        
        return summary
    
    def save_metrics(self, filepath:str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        output = {
            'summary': self.get_summary(),
            'detailed_metrics': [m.to_dict for m in self.metrics],
            'total_operations': len(self.metrics),
            'recorded_at': datetime.now().isoformat()

        }
        with open(filepath, 'w') as f:
            json.dump(output, f, indent=4)

    