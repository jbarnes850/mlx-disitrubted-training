#!/usr/bin/env python3

import os
import json
import logging
import argparse
from pathlib import Path
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_ci_environment() -> bool:
    """Check if we're running in CI environment"""
    return os.environ.get('CI', 'false').lower() == 'true'

def verify_model_outputs(results_dir: str, ci_mode: bool = False) -> bool:
    """Verify model outputs from distributed training"""
    try:
        # Load primary and secondary outputs
        primary_path = Path(results_dir) / "primary_outputs.json"
        secondary_path = Path(results_dir) / "secondary_outputs.json"
        
        if not primary_path.exists() or not secondary_path.exists():
            logger.error("Missing output files")
            return False
            
        with open(primary_path) as f:
            primary_results = json.load(f)
        with open(secondary_path) as f:
            secondary_results = json.load(f)
            
        # Verification criteria
        criteria = {
            "loss_threshold": 10.0 if ci_mode else 5.0,
            "gradient_norm_threshold": 10.0 if ci_mode else 5.0,
            "min_steps": 3 if ci_mode else 8
        }
        
        # Verify training progress
        if len(primary_results["losses"]) < criteria["min_steps"]:
            logger.error(f"Insufficient training steps: {len(primary_results['losses'])}")
            return False
            
        # Check loss convergence
        final_loss = np.mean(primary_results["losses"][-3:])
        if final_loss > criteria["loss_threshold"]:
            logger.error(f"Loss too high: {final_loss:.2f}")
            return False
            
        # Verify gradient synchronization
        primary_grads = primary_results["gradient_norms"]
        secondary_grads = secondary_results["gradient_norms"]
        
        if len(primary_grads) != len(secondary_grads):
            logger.error("Gradient length mismatch")
            return False
            
        grad_diff = np.mean([abs(p - s) for p, s in zip(primary_grads, secondary_grads)])
        if grad_diff > criteria["gradient_norm_threshold"]:
            logger.error(f"Gradient synchronization error: {grad_diff:.2f}")
            return False
            
        logger.info("Model output verification passed")
        logger.info(f"Final loss: {final_loss:.2f}")
        logger.info(f"Gradient sync difference: {grad_diff:.2f}")
        return True
        
    except Exception as e:
        logger.error(f"Verification failed: {str(e)}")
        return False

def verify_metrics(metrics_dir: str, ci_mode: bool = False) -> bool:
    """Verify training metrics"""
    try:
        metrics_path = Path(metrics_dir) / "training_metrics.json"
        if not metrics_path.exists():
            logger.error("Missing metrics file")
            return False
            
        with open(metrics_path) as f:
            metrics = json.load(f)
            
        # Adjust thresholds for CI
        thresholds = {
            "throughput": 100 if ci_mode else 500,  # samples/sec
            "memory_usage": 0.9 if ci_mode else 0.8,  # percentage
            "communication_time": 0.3 if ci_mode else 0.2  # percentage
        }
        
        # Verify performance metrics
        if metrics["throughput"] < thresholds["throughput"]:
            logger.error(f"Low throughput: {metrics['throughput']:.2f} samples/sec")
            return False
            
        if metrics["memory_usage"] > thresholds["memory_usage"]:
            logger.error(f"High memory usage: {metrics['memory_usage']*100:.1f}%")
            return False
            
        if metrics["communication_time"] > thresholds["communication_time"]:
            logger.error(f"High communication overhead: {metrics['communication_time']*100:.1f}%")
            return False
            
        logger.info("Performance metrics verification passed")
        logger.info(f"Throughput: {metrics['throughput']:.2f} samples/sec")
        logger.info(f"Memory usage: {metrics['memory_usage']*100:.1f}%")
        logger.info(f"Communication time: {metrics['communication_time']*100:.1f}%")
        return True
        
    except Exception as e:
        logger.error(f"Metrics verification failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Verify distributed training results")
    parser.add_argument("--results-dir", default="results", help="Directory containing training results")
    parser.add_argument("--metrics-dir", default="metrics", help="Directory containing training metrics")
    parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode with adjusted thresholds")
    args = parser.parse_args()
    
    # Check if running in CI
    ci_mode = args.ci_mode or is_ci_environment()
    if ci_mode:
        logger.info("Running verification in CI mode")
    
    # Run verifications
    model_check = verify_model_outputs(args.results_dir, ci_mode)
    metrics_check = verify_metrics(args.metrics_dir, ci_mode)
    
    if model_check and metrics_check:
        logger.info("All verifications passed")
        return 0
    else:
        logger.error("Verification failed")
        return 1

if __name__ == "__main__":
    exit(main())
