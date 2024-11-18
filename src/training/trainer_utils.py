class TrainingStabilizer:
    def __init__(self):
        self.ema = ExponentialMovingAverage(decay=0.99)
        self.loss_history = []
        
    def check_training_health(self, loss: float, step: int) -> Dict[str, Any]:
        """Monitor training stability"""
        metrics = {
            "loss_std": np.std(self.loss_history[-100:]),
            "loss_trend": self._calculate_trend(),
            "gradient_norm": self._compute_gradient_norm(),
            "learning_rate": self._get_current_lr(step)
        }
        
        # Detect training issues
        if metrics["loss_std"] > 5.0:
            self.logger.warning("High loss variance detected")
        
        return metrics 