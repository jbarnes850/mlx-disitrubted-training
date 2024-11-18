import mlx.core as mx
import time
import json
import logging
from pathlib import Path
from src.models.unified_model import UnifiedModel, ModelConfig
from src.monitoring.dashboard import PerformanceDashboard
from typing import Dict, Any, List
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelBenchmark:
    """Comprehensive benchmark suite for MLX models"""
    def __init__(self, config_path: str = "configs/distributed_config.json"):
        self.config = self._load_config(config_path)
        self.dashboard = PerformanceDashboard(self.config["monitoring"])
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        
    def _load_config(self, config_path: str) -> dict:
        with open(config_path) as f:
            return json.load(f)
            
    async def benchmark_mmlu(self, split: str = "test") -> Dict[str, float]:
        """Benchmark on MMLU dataset"""
        logger.info("Running MMLU benchmark...")
        
        # Load MMLU dataset
        dataset = load_dataset("cais/mmlu", split=split)
        
        results = {
            "humanities": [],
            "social_sciences": [],
            "stem": [],
            "other": []
        }
        
        model = UnifiedModel(ModelConfig(**self.config["model"]))
        
        for example in dataset:
            subject = example["subject"]
            category = self._get_subject_category(subject)
            
            # Format question with choices
            prompt = self._format_mmlu_prompt(example)
            
            # Get model prediction
            inputs = self.tokenizer(prompt, return_tensors="np")
            output = model(inputs["input_ids"])
            pred = output.argmax(-1).item()
            
            # Record accuracy
            correct = pred == example["answer"]
            results[category].append(correct)
            
        # Compute averages
        return {
            category: np.mean(scores) * 100
            for category, scores in results.items()
        }
        
    async def benchmark_mmlu_pro(self, split: str = "test") -> Dict[str, float]:
        """Benchmark on MMLU-Pro dataset"""
        logger.info("Running MMLU-Pro benchmark...")
        
        # Load MMLU-Pro dataset
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split=split)
        
        results = {
            "biology": [],
            "business": [],
            "chemistry": [],
            "computer_science": [],
            "economics": [],
            "engineering": [],
            "health": [],
            "history": [],
            "law": [],
            "math": [],
            "philosophy": [],
            "physics": [],
            "psychology": [],
            "others": []
        }
        
        model = UnifiedModel(ModelConfig(**self.config["model"]))
        
        for example in dataset:
            subject = example["subject"]
            
            # Format question with 10 choices
            prompt = self._format_mmlu_pro_prompt(example)
            
            # Get model prediction
            inputs = self.tokenizer(prompt, return_tensors="np")
            output = model(inputs["input_ids"])
            pred = output.argmax(-1).item()
            
            # Record accuracy
            correct = pred == example["answer"]
            results[subject].append(correct)
            
        # Compute averages
        return {
            subject: np.mean(scores) * 100
            for subject, scores in results.items()
        }
        
    def benchmark_training(self, num_steps: int = 100) -> Dict[str, float]:
        """Benchmark training performance"""
        logger.info("Benchmarking training performance...")
        
        model = UnifiedModel(ModelConfig(**self.config["model"]))
        
        # Create dummy batch
        batch = {
            "input_ids": mx.random.randint(0, self.config["model"]["vocab_size"], (32, 512)),
            "labels": mx.random.randint(0, self.config["model"]["vocab_size"], (32, 512))
        }
        
        # Warmup
        for _ in range(5):
            loss = model(batch["input_ids"])
            mx.eval(loss)
            
        # Benchmark
        start_time = time.time()
        total_tokens = 0
        
        for _ in range(num_steps):
            loss = model(batch["input_ids"])
            mx.eval(loss)
            total_tokens += batch["input_ids"].shape[0] * batch["input_ids"].shape[1]
            
        duration = time.time() - start_time
        
        return {
            "tokens_per_second": total_tokens / duration,
            "steps_per_second": num_steps / duration,
            "latency_ms": (duration / num_steps) * 1000
        }
        
    def benchmark_inference(self, num_tokens: int = 1000) -> Dict[str, float]:
        """Benchmark inference performance"""
        logger.info("Benchmarking inference performance...")
        
        model = UnifiedModel(ModelConfig(**self.config["model"]))
        
        # Create prompt
        prompt = mx.random.randint(0, self.config["model"]["vocab_size"], (1, 32))
        
        # Warmup
        for _ in range(5):
            next(model.generate(prompt, max_length=1))
            
        # Benchmark
        start_time = time.time()
        tokens_generated = 0
        latencies = []
        
        while tokens_generated < num_tokens:
            token_start = time.time()
            token = next(model.generate(prompt, max_length=1))
            mx.eval(token)
            latencies.append(time.time() - token_start)
            tokens_generated += 1
            
        duration = time.time() - start_time
        
        return {
            "tokens_per_second": tokens_generated / duration,
            "avg_latency_ms": (sum(latencies) / len(latencies)) * 1000,
            "p90_latency_ms": sorted(latencies)[int(len(latencies) * 0.9)] * 1000
        }
        
    def _format_mmlu_prompt(self, example: Dict) -> str:
        """Format MMLU question with choices"""
        choices = ["A", "B", "C", "D"]
        prompt = f"Question: {example['question']}\n\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{choices[i]}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
        
    def _format_mmlu_pro_prompt(self, example: Dict) -> str:
        """Format MMLU-Pro question with 10 choices"""
        choices = list("ABCDEFGHIJ")
        prompt = f"Question: {example['question']}\n\n"
        for i, choice in enumerate(example['choices']):
            prompt += f"{choices[i]}. {choice}\n"
        prompt += "\nAnswer:"
        return prompt
        
    def _get_subject_category(self, subject: str) -> str:
        """Map MMLU subject to category"""
        categories = {
            "humanities": ["philosophy", "history", "literature"],
            "social_sciences": ["economics", "psychology", "sociology"],
            "stem": ["mathematics", "physics", "chemistry", "biology", "computer_science"],
            "other": ["professional_law", "professional_medicine"]
        }
        
        for category, subjects in categories.items():
            if any(s in subject.lower() for s in subjects):
                return category
        return "other"
        
    async def run_all_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Run all benchmarks"""
        results = {
            "mmlu": await self.benchmark_mmlu(),
            "mmlu_pro": await self.benchmark_mmlu_pro(),
            "training": self.benchmark_training(),
            "inference": self.benchmark_inference()
        }
        
        # Save results
        output_path = Path("benchmark_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Log comparison with other models
        self._log_model_comparison(results)
            
        return results
        
    def _log_model_comparison(self, results: Dict[str, Dict[str, float]]):
        """Log comparison with other 1B models"""
        comparisons = {
            "Llama-3.2-1B": {
                "mmlu_avg": 49.3,          # From benchmarks
                "arc_challenge": 59.4,      # Reasoning
                "hellaswag": 41.2,         # Common sense
                "gsm8k": 44.4,             # Math
                "math": 30.6,              # Math
                "bfcl_v2": 25.7,          # Tool use
                "tokens_per_sec": 5000     # Target throughput
            },
            "OpenELM-1.1B": {
                "mmlu_avg": 32.34,
                "arc_challenge": 35.58,
                "hellaswag": 41.97,
                "tokens_per_sec": 4800
            },
            "Phi-1.5": {
                "mmlu_avg": 50.0,
                "arc_challenge": 42.1,
                "hellaswag": 43.2,
                "tokens_per_sec": 4500
            }
        }
        
        # Performance targets for distributed setup
        distributed_targets = {
            "mmlu_avg": 51.0,          # Target above Phi-1.5
            "arc_challenge": 60.0,      # Target above Llama 3.2
            "hellaswag": 45.0,         # Target above competitors
            "tokens_per_sec": 8000,     # Higher with distributed
            "latency_ms": 50,          # Target low latency
            "memory_efficiency": 0.85   # Target high efficiency
        }
        
        logger.info("\nModel Comparisons:")
        for model, metrics in comparisons.items():
            logger.info(f"\n{model}:")
            for metric, value in metrics.items():
                our_value = self._get_comparable_metric(results, metric)
                diff = our_value - value
                logger.info(f"  {metric}: {our_value:.2f} vs {value:.2f} ({diff:+.2f})")
                
                # Check against distributed targets
                if metric in distributed_targets:
                    target = distributed_targets[metric]
                    if our_value < target:
                        logger.warning(
                            f"  ⚠️ {metric} below distributed target: {target:.1f}"
                        )
                    else:
                        logger.info(
                            f"  ✅ {metric} meets distributed target: {target:.1f}"
                        )

async def main():
    """Main benchmark function"""
    parser = argparse.ArgumentParser(description="MLX Model Benchmark")
    parser.add_argument("--config", default="configs/distributed_config.json")
    args = parser.parse_args()
    
    benchmark = ModelBenchmark(args.config)
    results = await benchmark.run_all_benchmarks()
    
    logger.info("\nBenchmark Results:")
    
    # MMLU Results
    logger.info("\nMMLU Results:")
    for category, score in results["mmlu"].items():
        logger.info(f"  {category}: {score:.2f}%")
    logger.info(f"  Average: {np.mean(list(results['mmlu'].values())):.2f}%")
    
    # MMLU-Pro Results
    logger.info("\nMMLU-Pro Results:")
    for subject, score in results["mmlu_pro"].items():
        logger.info(f"  {subject}: {score:.2f}%")
    logger.info(f"  Average: {np.mean(list(results['mmlu_pro'].values())):.2f}%")
    
    # Performance Results
    logger.info("\nTraining Performance:")
    for metric, value in results["training"].items():
        logger.info(f"  {metric}: {value:.2f}")
        
    logger.info("\nInference Performance:")
    for metric, value in results["inference"].items():
        logger.info(f"  {metric}: {value:.2f}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main()) 