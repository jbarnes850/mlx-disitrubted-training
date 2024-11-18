import mlx.core as mx
import numpy as np
from typing import Iterator, Tuple, Dict, Optional
from pathlib import Path
import logging
import json
from dataclasses import dataclass

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
        data_path: str,
        config: DataConfig,
        world_size: int = 1,
        rank: int = 0
    ):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        
        # Setup data source
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {data_path}")
            
        # Calculate shard info for distributed training
        self.shard_size = self._get_shard_size()
        self.shard_start = self.rank * self.shard_size
        self.shard_end = self.shard_start + self.shard_size
        
        logging.info(
            f"Rank {rank}/{world_size} processing shard "
            f"[{self.shard_start}:{self.shard_end}]"
        )
    
    def _get_shard_size(self) -> int:
        """Calculate size of data shard for this worker"""
        total_size = sum(1 for _ in open(self.data_path, 'r'))
        return total_size // self.world_size
    
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        """Stream data in chunks to avoid loading everything into memory"""
        chunk = []
        chunk_size = 0
        
        # Read only this worker's shard
        with open(self.data_path, 'r') as f:
            # Skip to shard start
            for _ in range(self.shard_start):
                next(f)
                
            # Process shard
            for line_num, line in enumerate(f):
                if line_num >= self.shard_size:
                    break
                    
                # Parse data
                sample = json.loads(line)
                chunk.append(sample)
                chunk_size += 1
                
                # Yield batch when chunk is full
                if chunk_size == self.config.streaming_chunk_size:
                    yield from self._process_chunk(chunk)
                    chunk = []
                    chunk_size = 0
            
            # Handle final partial chunk
            if chunk:
                yield from self._process_chunk(chunk)
    
    def _process_chunk(
        self, 
        chunk: list
    ) -> Iterator[Dict[str, mx.array]]:
        """Process a chunk of raw data into batches"""
        # Convert chunk to arrays
        input_ids = mx.array([s['input_ids'] for s in chunk])
        labels = mx.array([s['labels'] for s in chunk])
        
        # Generate batches
        for i in range(0, len(chunk), self.config.batch_size):
            batch_slice = slice(i, i + self.config.batch_size)
            yield {
                'input_ids': input_ids[batch_slice],
                'labels': labels[batch_slice]
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