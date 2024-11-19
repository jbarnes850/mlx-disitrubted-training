import mlx.core as mx
import numpy as np
from typing import Iterator, Tuple, Dict, Optional, List
from pathlib import Path
import logging
import json
from dataclasses import dataclass
from src.data.dataset_config import DataConfig, BaseKnowledgeConfig

@dataclass
class DataConfig:
    batch_size: int
    sequence_length: int
    prefetch_batches: int
    streaming_chunk_size: int = 1000

class StreamingDataset:
    """Memory efficient dataset that streams from disk"""
    def __init__(
        self,
        config: DataConfig,
        world_size: int = 1,
        rank: int = 0
    ):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        
        # Initialize datasets
        self.datasets = {}
        self._initialize_datasets()
        
    def _initialize_datasets(self):
        """Initialize all configured datasets"""
        for dataset_name, weight in self.config.base_knowledge.datasets.items():
            dataset_path = self.config.data_dir / dataset_name.replace("/", "_")
            if not dataset_path.exists():
                logging.warning(f"Dataset {dataset_name} not found at {dataset_path}")
                continue
                
            self.datasets[dataset_name] = {
                "path": dataset_path,
                "weight": weight,
                "file_size": dataset_path.stat().st_size
            }
            
        if not self.datasets:
            raise ValueError("No valid datasets found")
            
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Stream data in chunks to avoid loading everything into memory"""
        while True:
            # Sample dataset based on weights
            dataset_name = np.random.choice(
                list(self.datasets.keys()),
                p=[d["weight"] for d in self.datasets.values()]
            )
            dataset = self.datasets[dataset_name]
            
            # Stream chunks
            with open(dataset["path"], 'r') as f:
                chunk = []
                for line in f:
                    if len(chunk) >= self.config.streaming_chunk_size:
                        yield self._process_chunk(chunk)
                        chunk = []
                    chunk.append(json.loads(line))
                    
                if chunk:  # Process remaining data
                    yield self._process_chunk(chunk)
                    
    def _process_chunk(self, chunk: List[Dict]) -> Dict[str, mx.array]:
        """Process a chunk of data into model inputs"""
        # This would use proper tokenization in production
        input_ids = mx.zeros((
            self.config.batch_size,
            self.config.sequence_length
        ))
        attention_mask = mx.ones_like(input_ids)
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

class PrefetchDataLoader:
    """Prefetch batches in background to optimize throughput"""
    def __init__(
        self,
        dataset: StreamingDataset,
        num_prefetch: int = 2
    ):
        self.dataset = dataset
        self.num_prefetch = num_prefetch
        
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Iterate with prefetching"""
        # Create CPU stream for data loading
        data_stream = mx.Stream(mx.cpu)
        
        # Setup prefetch queue
        prefetch_queue = []
        dataset_iter = iter(self.dataset)
        
        # Initial prefetch
        for _ in range(self.num_prefetch):
            try:
                with mx.stream(data_stream):
                    batch = next(dataset_iter)
                prefetch_queue.append(batch)
            except StopIteration:
                break
        
        # Main iteration loop
        while prefetch_queue:
            # Return current batch
            yield prefetch_queue.pop(0)
            
            # Prefetch next batch
            try:
                with mx.stream(data_stream):
                    batch = next(dataset_iter)
                prefetch_queue.append(batch)
            except StopIteration:
                continue 