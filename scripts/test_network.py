import mlx.core as mx
import time
import logging
import subprocess
from typing import Dict, Optional
import socket

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_network_bandwidth(target_host: str) -> Optional[float]:
    """Test network bandwidth using iperf3"""
    try:
        result = subprocess.run(
            ["iperf3", "-c", target_host, "-J"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            # Parse JSON output for bandwidth
            import json
            data = json.loads(result.stdout)
            return data["end"]["sum_received"]["bits_per_second"] / 1e9  # Convert to Gbps
    except Exception as e:
        logger.error(f"Bandwidth test failed: {str(e)}")
    return None

def test_mpi_communication() -> Dict[str, float]:
    """Test MPI communication performance"""
    world = mx.distributed.init()
    
    # Create large tensor for bandwidth test
    data = mx.random.normal((1000, 1000))
    
    # Test latency and bandwidth
    start = time.time()
    for _ in range(100):
        result = mx.distributed.all_sum(data)
        mx.eval(result)
    duration = time.time() - start
    
    return {
        "bandwidth_gbps": (data.nbytes * 100) / (duration * 1e9),
        "latency_ms": duration * 10  # Average latency per operation
    }

def verify_ssh_access(target_host: str) -> bool:
    """Verify SSH access to target host"""
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", target_host, "echo test"],
            capture_output=True
        )
        return result.returncode == 0
    except Exception:
        return False

def main():
    # Test local network
    logger.info("Testing local network...")
    local_ip = socket.gethostbyname(socket.gethostname())
    bandwidth = test_network_bandwidth(local_ip)
    if bandwidth:
        logger.info(f"Local network bandwidth: {bandwidth:.2f} Gbps")
        if bandwidth < 10:
            logger.warning("Network bandwidth below recommended 10 Gbps")
    
    # Test MPI communication
    logger.info("\nTesting MPI communication...")
    mpi_stats = test_mpi_communication()
    logger.info(f"MPI bandwidth: {mpi_stats['bandwidth_gbps']:.2f} Gbps")
    logger.info(f"MPI latency: {mpi_stats['latency_ms']:.2f} ms")
    
    # Verify SSH access
    logger.info("\nVerifying SSH access...")
    secondary_host = "secondary-device.local"
    if verify_ssh_access(secondary_host):
        logger.info(f"SSH access to {secondary_host} verified")
    else:
        logger.error(f"SSH access to {secondary_host} failed")
        
    # Overall assessment
    logger.info("\nNetwork Assessment:")
    if all([
        bandwidth and bandwidth >= 10,
        mpi_stats["bandwidth_gbps"] >= 8,
        mpi_stats["latency_ms"] < 2,
        verify_ssh_access(secondary_host)
    ]):
        logger.info("✅ Network setup meets requirements")
        return 0
    else:
        logger.error("❌ Network setup does not meet requirements")
        return 1

if __name__ == "__main__":
    exit(main()) 