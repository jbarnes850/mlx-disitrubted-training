import mlx.core as mx
from datasets import load_dataset, concatenate_datasets
from typing import Iterator, Dict, Optional, Union, List
from pathlib import Path
import logging
from dataclasses import dataclass, field
from huggingface_hub import HfApi
from transformers import AutoTokenizer
import numpy as np
import threading
from queue import Queue
import re

@dataclass
class DataConfig:
    """High-quality dataset configuration"""
    datasets: Dict[str, float] = field(default_factory=lambda: {
        # High-quality instruction data (30%)
        "HuggingFaceH4/ultrachat_200k": 0.15,  # High-quality chat
        "teknium/openhermes": 0.10,      # High-quality instruction data
        "garage-bAInd/Open-Platypus": 0.10,  # Technical QA
        
        # Code and technical (25%)
        "bigcode/starcoderdata": 0.15,         # Clean code
        "codeparrot/github-code": 0.10,        # Additional code
        
        # Books and knowledge (45%)
        "EleutherAI/pile": 0.20,               # Filtered web content
        "meta-math/MetaMathQA": 0.10     # Mathematical reasoning
    })
    quality_threshold: float = 0.8  # Increased quality threshold
    streaming: bool = True
    cache_dir: Optional[str] = "data/cache"
    max_length: int = 2048
    batch_size: int = 32
    prefetch_size: int = 2
    hf_token: Optional[str] = None
    content_filters: Dict[str, float] = field(default_factory=lambda: {
        "min_instruction_length": 8,
        "max_repetition_ratio": 0.3,
        "code_quality_threshold": 0.7
    })

class DataManager:
    """Manages dataset loading and preprocessing"""
    def __init__(
        self,
        config: DataConfig,
        tokenizer_name: str = "gpt2",
        world_size: int = 1,
        rank: int = 0
    ):
        self.config = config
        self.world_size = world_size
        self.rank = rank
        self.logger = logging.getLogger(__name__)
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            token=self.config.hf_token
        )
        
        self.prefetch_queue = Queue(maxsize=config.prefetch_size)
        self.streaming_worker = None
        
    def load_datasets(self):
        """Load and combine multiple datasets"""
        datasets = []
        weights = []
        
        for dataset_name, weight in self.config.datasets.items():
            try:
                # Load dataset with streaming
                dataset = load_dataset(
                    dataset_name,
                    streaming=self.config.streaming,
                    split="train",
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token
                )
                
                # Apply quality filtering
                if "quality_score" in dataset.features:
                    dataset = dataset.filter(
                        lambda x: x.get("quality_score", 0) >= self.config.quality_threshold
                    )
                
                # Apply dataset-specific preprocessing
                if "pg19" in dataset_name:
                    # Handle PG19's specific format
                    dataset = dataset.map(
                        lambda x: {"text": x["text"].strip()}
                    )
                elif "librispeech_lm" in dataset_name:
                    # Clean and format LibriSpeech text
                    dataset = dataset.map(
                        lambda x: {"text": self._clean_text(x["text"])}
                    )
                
                datasets.append(dataset)
                weights.append(weight)
                self.logger.info(f"Loaded dataset: {dataset_name}")
                
            except Exception as e:
                self.logger.warning(f"Failed to load dataset {dataset_name}: {str(e)}")
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w/total_weight for w in weights]
        
        # Interleave datasets with weights
        combined = datasets[0]
        if len(datasets) > 1:
            combined = combined.interleave_datasets(
                datasets[1:],
                probabilities=weights[1:],
                stopping_strategy='first_exhausted'
            )
        
        # Shard for distributed training
        if self.world_size > 1:
            combined = combined.shard(
                num_shards=self.world_size,
                index=self.rank
            )
        
        return combined
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove excessive whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        return text.strip()
    
    def preprocess_function(self, examples: Dict[str, list]) -> Dict[str, mx.array]:
        """Preprocess batch of examples"""
        # Tokenize texts
        tokenized = self.tokenizer(
            examples["text"],
            truncation=True,
            max_length=self.config.max_length,
            padding="max_length",
            return_tensors="np"
        )
        
        # Convert to MLX arrays
        return {
            "input_ids": mx.array(tokenized["input_ids"]),
            "attention_mask": mx.array(tokenized["attention_mask"]),
            "labels": mx.array(tokenized["input_ids"])
        }
    
    def start_streaming(self):
        """Initialize async data streaming"""
        self.streaming_worker = threading.Thread(
            target=self._stream_data,
            daemon=True
        )
        self.streaming_worker.start()

class StreamingDataLoader:
    """Memory-efficient data loader with prefetching"""
    def __init__(
        self,
        dataset,
        batch_size: int,
        shuffle: bool = True,
        prefetch_size: int = 2,
        drop_last: bool = False
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.prefetch_size = prefetch_size
        self.drop_last = drop_last
        
        # Create prefetch queue
        self.queue = []
        
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        # Initialize dataset iterator
        dataset_iter = iter(self.dataset)
        
        # Fill prefetch queue
        while len(self.queue) < self.prefetch_size:
            try:
                batch = self._get_batch(dataset_iter)
                self.queue.append(batch)
            except StopIteration:
                break
        
        # Main iteration loop
        while self.queue:
            # Return current batch
            yield self.queue.pop(0)
            
            # Prefetch next batch
            try:
                batch = self._get_batch(dataset_iter)
                self.queue.append(batch)
            except StopIteration:
                continue
                
    def _get_batch(self, iterator) -> Dict[str, mx.array]:
        """Get next batch from iterator"""
        batch = []
        while len(batch) < self.batch_size:
            try:
                item = next(iterator)
                batch.append(item)
            except StopIteration:
                if len(batch) == 0 or (self.drop_last and len(batch) < self.batch_size):
                    raise
                break
        
        # Stack batch items
        return {
            k: mx.stack([item[k] for item in batch])
            for k in batch[0].keys()
        }

class OptimizedDataManager(DataManager):
    """Optimized version of DataManager for multiple datasets"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = {}
        self.prefetch_thread = None
        self.prefetch_queue = Queue(maxsize=self.config.prefetch_size)
        
    def load_datasets(self):
        """Optimized dataset loading"""
        # Use dataset interleaving instead of concatenation
        datasets = []
        weights = []  # Sampling weights for each dataset
        
        total_size = 0
        for dataset_name in self.config.datasets:
            try:
                dataset = load_dataset(
                    dataset_name,
                    streaming=True,
                    split=self.config.dataset_splits.get(dataset_name, "train"),
                    cache_dir=self.config.cache_dir,
                    token=self.config.hf_token
                )
                
                # Apply quality filtering if available
                if "quality_score" in dataset.features:
                    dataset = dataset.filter(
                        lambda x: x.get("quality_score", 0) >= self.config.quality_threshold
                    )
                
                # Calculate approximate size for weighting
                sample_size = min(1000, len(dataset))
                dataset_size = len(dataset)
                total_size += dataset_size
                
                datasets.append(dataset)
                weights.append(dataset_size)
                
                self.logger.info(f"Loaded dataset: {dataset_name} (size: {dataset_size})")
                
            except Exception as e:
                self.logger.warning(f"Failed to load dataset {dataset_name}: {str(e)}")
        
        # Normalize weights
        weights = [w/total_size for w in weights]
        
        # Create interleaved dataset
        interleaved = datasets[0].shuffle()
        if len(datasets) > 1:
            interleaved = interleaved.interleave_datasets(
                datasets[1:],
                probabilities=weights,
                stopping_strategy='first_exhausted'
            )
        
        # Shard for distributed training
        if self.world_size > 1:
            interleaved = interleaved.shard(
                num_shards=self.world_size,
                index=self.rank
            )
        
        return interleaved
    
    def start_prefetch(self):
        """Start background prefetching"""
        def prefetch_worker():
            try:
                for batch in self.dataset:
                    processed = self.preprocess_function(batch)
                    self.prefetch_queue.put(processed)
            except Exception as e:
                self.logger.error(f"Prefetch error: {str(e)}")
                
        self.prefetch_thread = threading.Thread(target=prefetch_worker)
        self.prefetch_thread.daemon = True
        self.prefetch_thread.start()
    
    def get_batch(self) -> Dict[str, mx.array]:
        """Get next batch with prefetching"""
        if self.prefetch_queue.empty():
            # If queue is empty, process directly
            batch = next(self.dataset)
            return self.preprocess_function(batch)
        return self.prefetch_queue.get()

class CachedStreamingDataLoader(StreamingDataLoader):
    """Enhanced dataloader with caching and prefetching"""
    def __init__(
        self,
        dataset,
        batch_size: int,
        cache_size: int = 1000,  # Number of batches to cache
        *args,
        **kwargs
    ):
        super().__init__(dataset, batch_size, *args, **kwargs)
        self.cache_size = cache_size
        self.cache = {}
        self.cache_hits = 0
        self.total_requests = 0
        
    def __iter__(self) -> Iterator[Dict[str, mx.array]]:
        # Start prefetching
        self.dataset.start_prefetch()
        
        while True:
            try:
                batch = self.dataset.get_batch()
                
                # Update cache statistics
                self.total_requests += 1
                batch_hash = hash(batch['input_ids'].tobytes())
                
                if batch_hash in self.cache:
                    self.cache_hits += 1
                    yield self.cache[batch_hash]
                else:
                    # Cache new batch if space available
                    if len(self.cache) < self.cache_size:
                        self.cache[batch_hash] = batch
                    yield batch
                
            except StopIteration:
                break
                
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests