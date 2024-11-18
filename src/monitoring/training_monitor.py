from typing import Dict, List, Tuple, DefaultDict
from collections import defaultdict

class TrainingMonitor:
    """Enhanced training monitoring"""
    def __init__(self):
        self.metrics_history = defaultdict(list)
        self.alerts = AlertManager()
        
    def track_step(self, metrics: Dict[str, float]):
        """Track training metrics"""
        # Store metrics
        for name, value in metrics.items():
            self.metrics_history[name].append(value)
            
        # Check for issues
        self._check_training_health()
        
        # Generate visualizations
        if len(self.metrics_history["loss"]) % 100 == 0:
            self._generate_training_plots() 