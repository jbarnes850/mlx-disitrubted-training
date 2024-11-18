import mlx.core as mx
import time
import logging
import subprocess
import argparse
from typing import Dict, Optional
import socket
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_ci_environment() -> bool:
    """Check if we're running in a CI environment"""
    return os.environ.get('CI', 'false').lower() == 'true'

def test_network_bandwidth(target_host: str) -> Optional[float]:
    """Test network bandwidth using iperf3"""
    try:
        # Skip actual bandwidth test in CI
        if is_ci_environment():
            logger.info("Skipping bandwidth test in CI environment")
            return 10.0  # Mock 10Gbps for CI

        result = subprocess.run(
            ["iperf3", "-c", target_host, "-J"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            bandwidth = data["end"]["sum_received"]["bits_per_second"] / 1e9
            logger.info(f"Network bandwidth: {bandwidth:.2f} Gbps")
            return bandwidth
    except Exception as e:
        logger.error(f"Bandwidth test failed: {str(e)}")
    return None

def test_mpi_communication(local_only: bool = False) -> Dict[str, float]:
    """Test MPI communication performance"""
    try:
        if is_ci_environment() or local_only:
            logger.info("Running in local/CI mode - using single process MPI")
            world = mx.distributed.init()
        else:
            world = mx.distributed.init()
        
        # Create smaller tensor for CI/local testing
        size = 100 if (is_ci_environment() or local_only) else 1000
        data = mx.random.normal((size, size))
        
        # Test latency and bandwidth
        start = time.time()
        iterations = 10 if (is_ci_environment() or local_only) else 100
        for _ in range(iterations):
            result = mx.distributed.all_sum(data)
            mx.eval(result)
        duration = time.time() - start
        
        metrics = {
            "bandwidth_gbps": (data.nbytes * iterations) / (duration * 1e9),
            "latency_ms": duration * (1000.0 / iterations)
        }
        logger.info(f"MPI Communication metrics: {metrics}")
        return metrics
        
    except Exception as e:
        logger.error(f"MPI communication test failed: {str(e)}")
        return {"bandwidth_gbps": 0, "latency_ms": 0}

def verify_ssh_access(target_host: str) -> bool:
    """Verify SSH access to target host"""
    try:
        if is_ci_environment():
            logger.info("Skipping SSH verification in CI environment")
            return True
            
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target_host, "echo test"],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"SSH verification failed: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test network configuration for distributed training')
    parser.add_argument('--local-only', action='store_true', help='Run only local tests')
    parser.add_argument('--target-host', type=str, help='Target host for network tests')
    args = parser.parse_args()

    success = True
    
    # Local-only mode for CI or single-machine testing
    if args.local_only or is_ci_environment():
        logger.info("Running in local-only mode")
        mpi_metrics = test_mpi_communication(local_only=True)
        success = mpi_metrics["bandwidth_gbps"] > 0
    else:
        if not args.target_host:
            logger.error("Target host required for distributed tests")
            return False
            
        # Full distributed testing
        logger.info(f"Testing connection to {args.target_host}")
        if not verify_ssh_access(args.target_host):
            logger.error("SSH verification failed")
            return False
            
        bandwidth = test_network_bandwidth(args.target_host)
        if not bandwidth or bandwidth < 1.0:  # Require at least 1Gbps
            logger.error("Insufficient network bandwidth")
            success = False
            
        mpi_metrics = test_mpi_communication()
        if mpi_metrics["bandwidth_gbps"] < 0.5:  # Require at least 500Mbps for MPI
            logger.error("Insufficient MPI communication performance")
            success = False
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())