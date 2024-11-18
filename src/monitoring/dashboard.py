import mlx.core as mx
import psutil
import time
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import json
import threading
from collections import deque
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd

@dataclass
class DashboardConfig:
    update_interval: float = 1.0  # seconds
    history_size: int = 3600  # 1 hour at 1s intervals
    alert_thresholds: Dict[str, float] = None
    output_dir: str = "logs/dashboard"
    port: int = 8050
    enable_ui: bool = True

class MetricTracker:
    """Tracks and stores metrics with rolling history"""
    def __init__(self, max_size: int = 3600):
        self.history = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self.current = None
        
    def update(self, value: float):
        self.current = value
        self.history.append(value)
        self.timestamps.append(time.time())
        
    def get_stats(self) -> Dict[str, float]:
        if not self.history:
            return {}
        return {
            "current": self.current,
            "mean": np.mean(self.history),
            "max": np.max(self.history),
            "min": np.min(self.history),
            "std": np.std(self.history)
        }
        
    def get_history_df(self) -> pd.DataFrame:
        return pd.DataFrame({
            'timestamp': list(self.timestamps),
            'value': list(self.history)
        })

class PerformanceDashboard:
    """Real-time monitoring dashboard with visualization"""
    def __init__(self, config: DashboardConfig):
        self.config = config
        self.metrics = {
            "gpu_utilization": MetricTracker(),
            "cpu_utilization": MetricTracker(),
            "memory_usage": MetricTracker(),
            "throughput": MetricTracker(),
            "communication_overhead": MetricTracker(),
            "cache_hits": MetricTracker(),
            "loss": MetricTracker(),
            "learning_rate": MetricTracker(),
            "gradient_norm": MetricTracker(),
            "tokens_per_second": MetricTracker(),
            "inference_latency": MetricTracker()
        }
        
        # Setup logging
        self.setup_logging()
        
        # Initialize UI if enabled
        if self.config.enable_ui:
            self.setup_ui()
        
        # Start monitoring thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.start()
        
    def setup_logging(self):
        """Setup logging directory and files"""
        log_dir = Path(self.config.output_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_file = log_dir / "performance_metrics.jsonl"
        self.plot_dir = log_dir / "plots"
        self.plot_dir.mkdir(exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
    def setup_ui(self):
        """Setup Dash UI"""
        self.app = dash.Dash(__name__)
        
        self.app.layout = html.Div([
            html.H1('MLX Training Dashboard'),
            
            # Training Metrics
            html.Div([
                html.H2('Training Metrics'),
                dcc.Graph(id='training-metrics'),
                dcc.Interval(id='training-interval', interval=1000)
            ]),
            
            # System Metrics
            html.Div([
                html.H2('System Metrics'),
                dcc.Graph(id='system-metrics'),
                dcc.Interval(id='system-interval', interval=1000)
            ]),
            
            # Inference Metrics
            html.Div([
                html.H2('Inference Metrics'),
                dcc.Graph(id='inference-metrics'),
                dcc.Interval(id='inference-interval', interval=1000)
            ])
        ])
        
        self.setup_callbacks()
        
        # Start dashboard
        self.app.run_server(port=self.config.port, debug=False)
        
    def setup_callbacks(self):
        """Setup Dash callbacks"""
        @self.app.callback(
            Output('training-metrics', 'figure'),
            Input('training-interval', 'n_intervals')
        )
        def update_training_metrics(_):
            return self.create_training_plot()
            
        @self.app.callback(
            Output('system-metrics', 'figure'),
            Input('system-interval', 'n_intervals')
        )
        def update_system_metrics(_):
            return self.create_system_plot()
            
        @self.app.callback(
            Output('inference-metrics', 'figure'),
            Input('inference-interval', 'n_intervals')
        )
        def update_inference_metrics(_):
            return self.create_inference_plot()
            
    def create_training_plot(self) -> go.Figure:
        """Create training metrics plot"""
        fig = make_subplots(rows=2, cols=2)
        
        # Loss
        loss_df = self.metrics['loss'].get_history_df()
        fig.add_trace(
            go.Scatter(x=loss_df['timestamp'], y=loss_df['value'], name='Loss'),
            row=1, col=1
        )
        
        # Throughput
        throughput_df = self.metrics['tokens_per_second'].get_history_df()
        fig.add_trace(
            go.Scatter(x=throughput_df['timestamp'], y=throughput_df['value'], 
                      name='Tokens/sec'),
            row=1, col=2
        )
        
        # Learning Rate
        lr_df = self.metrics['learning_rate'].get_history_df()
        fig.add_trace(
            go.Scatter(x=lr_df['timestamp'], y=lr_df['value'], name='Learning Rate'),
            row=2, col=1
        )
        
        # Gradient Norm
        grad_df = self.metrics['gradient_norm'].get_history_df()
        fig.add_trace(
            go.Scatter(x=grad_df['timestamp'], y=grad_df['value'], 
                      name='Gradient Norm'),
            row=2, col=2
        )
        
        return fig
        
    def create_system_plot(self) -> go.Figure:
        """Create system metrics plot"""
        fig = make_subplots(rows=2, cols=2)
        
        # GPU Utilization
        gpu_df = self.metrics['gpu_utilization'].get_history_df()
        fig.add_trace(
            go.Scatter(x=gpu_df['timestamp'], y=gpu_df['value'], 
                      name='GPU Utilization'),
            row=1, col=1
        )
        
        # Memory Usage
        mem_df = self.metrics['memory_usage'].get_history_df()
        fig.add_trace(
            go.Scatter(x=mem_df['timestamp'], y=mem_df['value'], 
                      name='Memory (GB)'),
            row=1, col=2
        )
        
        # CPU Utilization
        cpu_df = self.metrics['cpu_utilization'].get_history_df()
        fig.add_trace(
            go.Scatter(x=cpu_df['timestamp'], y=cpu_df['value'], 
                      name='CPU Utilization'),
            row=2, col=1
        )
        
        # Cache Hits
        cache_df = self.metrics['cache_hits'].get_history_df()
        fig.add_trace(
            go.Scatter(x=cache_df['timestamp'], y=cache_df['value'], 
                      name='Cache Hit Rate'),
            row=2, col=2
        )
        
        return fig
        
    def create_inference_plot(self) -> go.Figure:
        """Create inference metrics plot"""
        fig = make_subplots(rows=1, cols=2)
        
        # Latency
        latency_df = self.metrics['inference_latency'].get_history_df()
        fig.add_trace(
            go.Scatter(x=latency_df['timestamp'], y=latency_df['value'], 
                      name='Latency (ms)'),
            row=1, col=1
        )
        
        # Throughput
        inf_throughput_df = self.metrics['tokens_per_second'].get_history_df()
        fig.add_trace(
            go.Scatter(x=inf_throughput_df['timestamp'], y=inf_throughput_df['value'], 
                      name='Tokens/sec'),
            row=1, col=2
        )
        
        return fig
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.running:
            try:
                self.update_metrics()
                self.check_alerts()
                time.sleep(self.config.update_interval)
            except Exception as e:
                logging.error(f"Monitoring error: {str(e)}")
                
    def update_metrics(self):
        """Update all metrics"""
        # GPU metrics
        if mx.metal.is_available():
            self.metrics["gpu_utilization"].update(
                mx.metal.get_active_memory() / mx.metal.get_memory_limit()
            )
            
        # CPU metrics
        self.metrics["cpu_utilization"].update(
            psutil.cpu_percent(interval=None)
        )
        
        # Memory metrics
        self.metrics["memory_usage"].update(
            psutil.Process().memory_info().rss / (1024**3)  # GB
        )
        
        # Log metrics
        self._log_metrics()
        
    def track_training_metrics(
        self,
        loss: float,
        throughput: float,
        learning_rate: float,
        gradient_norm: float
    ):
        """Track training-specific metrics"""
        self.metrics["loss"].update(loss)
        self.metrics["tokens_per_second"].update(throughput)
        self.metrics["learning_rate"].update(learning_rate)
        self.metrics["gradient_norm"].update(gradient_norm)
        
    def track_inference_metrics(
        self,
        latency_ms: float,
        throughput: float
    ):
        """Track inference-specific metrics"""
        self.metrics["inference_latency"].update(latency_ms)
        self.metrics["tokens_per_second"].update(throughput)
        
    def check_alerts(self):
        """Check metrics against alert thresholds"""
        if not self.config.alert_thresholds:
            return
            
        for metric, threshold in self.config.alert_thresholds.items():
            if metric in self.metrics:
                current = self.metrics[metric].current
                if current > threshold:
                    self._send_alert(metric, current, threshold)
                    
    def _send_alert(self, metric: str, value: float, threshold: float):
        """Send alert for threshold violation"""
        message = f"Alert: {metric} ({value:.2f}) exceeded threshold ({threshold:.2f})"
        logging.warning(message)
        # TODO: Implement additional alert mechanisms (email, Slack, etc.)
        
    def _log_metrics(self):
        """Log current metrics to file"""
        metrics = {
            name: tracker.get_stats()
            for name, tracker in self.metrics.items()
        }
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps({
                'timestamp': time.time(),
                'metrics': metrics
            }) + '\n')
            
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of all metrics"""
        return {
            name: tracker.get_stats()
            for name, tracker in self.metrics.items()
        }
        
    def save_plots(self):
        """Save current plots to disk"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save training plot
        self.create_training_plot().write_html(
            self.plot_dir / f"training_metrics_{timestamp}.html"
        )
        
        # Save system plot
        self.create_system_plot().write_html(
            self.plot_dir / f"system_metrics_{timestamp}.html"
        )
        
        # Save inference plot
        self.create_inference_plot().write_html(
            self.plot_dir / f"inference_metrics_{timestamp}.html"
        )
        
    def shutdown(self):
        """Clean shutdown of monitoring"""
        self.running = False
        self.monitor_thread.join()
        self.save_plots()
        logging.info("Performance monitoring stopped") 