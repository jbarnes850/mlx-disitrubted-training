"""
Basic MLX setup verification
"""
import mlx.core as mx
import logging
import argparse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_basic_setup() -> bool:
    """Verify basic MLX setup"""
    try:
        # Test basic MLX operation
        x = mx.array([1, 2, 3])
        y = x * 2
        logger.info("Basic MLX operation successful")
        return True
    except Exception as e:
        logger.error(f"Basic MLX verification failed: {str(e)}")
        return False

def main() -> int:
    parser = argparse.ArgumentParser(description="Verify MLX setup")
    parser.add_argument("--ci-mode", action="store_true", help="Run in CI mode")
    args = parser.parse_args()

    if verify_basic_setup():
        logger.info("✅ Basic setup verification passed")
        return 0
    else:
        logger.error("❌ Setup verification failed")
        return 1

if __name__ == "__main__":
    exit(main())