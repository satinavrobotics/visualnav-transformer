"""
Enhanced logging system for training metrics with grouped, interpretable output.
Provides organized metric display with logical grouping and visual hierarchy.
"""

import time
from typing import Dict, List, Optional, Any
from collections import defaultdict

from visualnav_transformer.train.vint_train.logging.cli_formatter import (
    Colors, Symbols, format_number, print_section
)
from visualnav_transformer.train.vint_train.logging.logger import Logger


class MetricGroup:
    """A group of related metrics with enhanced display."""
    
    def __init__(self, name: str, symbol: str, color: str, description: str = ""):
        self.name = name
        self.symbol = symbol
        self.color = color
        self.description = description
        self.loggers: Dict[str, Logger] = {}
        
    def add_logger(self, key: str, logger: Logger):
        """Add a logger to this group."""
        self.loggers[key] = logger
        
    def display(self, epoch: int, batch: int, total_batches: int, use_latest: bool = True) -> List[str]:
        """Display all metrics in this group with enhanced formatting."""
        if not self.loggers:
            return []
            
        lines = []
        progress_pct = (batch / (total_batches - 1)) * 100 if total_batches > 1 else 100
        
        # Group header
        batch_info = f"{Colors.BRIGHT_CYAN}Batch {batch+1}/{total_batches}{Colors.RESET}"
        epoch_info = f"{Colors.BRIGHT_YELLOW}Epoch {epoch+1}{Colors.RESET}"
        progress_info = f"{Colors.GRAY}({progress_pct:.1f}%){Colors.RESET}"
        
        header = f"  {self.symbol} {epoch_info} {batch_info} {progress_info}"
        if self.description:
            header += f" {Colors.GRAY}- {self.description}{Colors.RESET}"
        lines.append(header)
        
        # Display metrics in this group
        for key, logger in self.loggers.items():
            if use_latest:
                value = logger.latest()
                moving_avg = logger.moving_average()
                total_avg = logger.average()
            else:
                value = logger.average()
                moving_avg = value
                total_avg = value
                
            # Clean metric name (remove dataset suffix)
            clean_name = key.replace('-train', '').replace('-test', '').replace('-eval', '')
            
            formatted_value = format_number(value, 4)
            formatted_moving = format_number(moving_avg, 4)
            formatted_total = format_number(total_avg, 4)
            
            metric_line = f"    {self.color}{clean_name}:{Colors.RESET} {Colors.WHITE}{formatted_value}{Colors.RESET}"
            if use_latest:
                metric_line += f" {Colors.GRAY}(avg: {formatted_moving}, total: {formatted_total}){Colors.RESET}"
            
            lines.append(metric_line)
            
        return lines


class EnhancedMetricLogger:
    """Enhanced metric logger with grouped, interpretable output."""

    def __init__(self, mode: str = "train", print_log_freq: int = 100):
        self.mode = mode
        self.print_log_freq = print_log_freq
        self.groups: Dict[str, MetricGroup] = {}
        self.all_loggers: Dict[str, Logger] = {}
        self.last_display_time = time.time()
        self.last_displayed_batch = -1  # Track last displayed batch to prevent duplicates
        self.last_displayed_epoch = -1  # Track last displayed epoch

        # Define metric groups with logical organization
        self._setup_metric_groups()

    def __contains__(self, key: str) -> bool:
        """Support 'key in logger' syntax."""
        return key in self.all_loggers

    def __getitem__(self, key: str) -> Logger:
        """Support logger[key] syntax."""
        return self.all_loggers[key]

    def __setitem__(self, key: str, logger: Logger):
        """Support logger[key] = value syntax."""
        self.add_logger(key, logger)

    def items(self):
        """Support for key, value in logger.items() syntax."""
        return self.all_loggers.items()

    def keys(self):
        """Support logger.keys() syntax."""
        return self.all_loggers.keys()

    def values(self):
        """Support logger.values() syntax."""
        return self.all_loggers.values()
        
    def _setup_metric_groups(self):
        """Setup predefined metric groups for better organization."""
        
        # Loss metrics group
        self.groups["losses"] = MetricGroup(
            name="Loss Metrics",
            symbol=Symbols.LOSS,
            color=Colors.RED,
            description="Training losses"
        )
        
        # Accuracy/Performance metrics group  
        self.groups["performance"] = MetricGroup(
            name="Performance Metrics", 
            symbol=Symbols.TARGET,
            color=Colors.GREEN,
            description="Accuracy and similarity metrics"
        )
        
        # Distance metrics group
        self.groups["distance"] = MetricGroup(
            name="Distance Metrics",
            symbol=Symbols.CHART,
            color=Colors.BLUE,
            description="Distance prediction metrics"
        )
        
        # Pose/Navigation metrics group
        self.groups["navigation"] = MetricGroup(
            name="Navigation Metrics",
            symbol=Symbols.ROCKET,
            color=Colors.MAGENTA,
            description="Pose and navigation metrics"
        )
        
    def add_logger(self, key: str, logger: Logger):
        """Add a logger and automatically assign it to the appropriate group."""
        self.all_loggers[key] = logger
        
        # Determine which group this metric belongs to
        metric_name = key.lower()
        
        if any(term in metric_name for term in ['loss', 'mse', 'mae']):
            if 'dist' in metric_name:
                self.groups["distance"].add_logger(key, logger)
            elif 'pose' in metric_name:
                self.groups["navigation"].add_logger(key, logger)
            else:
                self.groups["losses"].add_logger(key, logger)
        elif any(term in metric_name for term in ['acc', 'sim', 'cosine']):
            self.groups["performance"].add_logger(key, logger)
        elif 'dist' in metric_name:
            self.groups["distance"].add_logger(key, logger)
        elif any(term in metric_name for term in ['pose', 'nav', 'action']):
            self.groups["navigation"].add_logger(key, logger)
        else:
            # Default to losses group
            self.groups["losses"].add_logger(key, logger)
            
    def should_display(self, batch_idx: int) -> bool:
        """Check if we should display metrics based on frequency."""
        return batch_idx % self.print_log_freq == 0 and self.print_log_freq != 0
        
    def display_metrics(self, epoch: int, batch: int, total_batches: int, use_latest: bool = True):
        """Display all metrics in organized groups."""
        if not self.should_display(batch):
            return

        # Prevent duplicate displays for the same batch/epoch
        if epoch == self.last_displayed_epoch and batch == self.last_displayed_batch:
            return

        # Display each group that has metrics
        displayed_any = False
        for group_name, group in self.groups.items():
            if group.loggers:  # Only display groups that have metrics
                lines = group.display(epoch, batch, total_batches, use_latest)
                for line in lines:
                    print(line)
                displayed_any = True

        if displayed_any:
            print()  # Add spacing after metric display
            # Update last displayed batch/epoch
            self.last_displayed_batch = batch
            self.last_displayed_epoch = epoch
            
    def get_mlflow_data(self) -> Dict[str, float]:
        """Get all metric data for MLflow logging, filtering out NaN values."""
        import numpy as np
        data_log = {}
        for key, logger in self.all_loggers.items():
            value = logger.latest()
            # Only include non-NaN values in MLflow logging
            if not np.isnan(value) and np.isfinite(value):
                data_log[logger.full_name()] = value
        return data_log
        
    def get_summary_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get summary metrics organized by group for epoch summaries."""
        summary = {}
        for group_name, group in self.groups.items():
            if group.loggers:
                group_metrics = {}
                for key, logger in group.loggers.items():
                    clean_name = key.replace('-train', '').replace('-test', '').replace('-eval', '')
                    group_metrics[clean_name] = {
                        'latest': logger.latest(),
                        'average': logger.average(),
                        'moving_avg': logger.moving_average()
                    }
                if group_metrics:
                    summary[group_name] = group_metrics
        return summary


def create_enhanced_loggers(mode: str, print_log_freq: int = 100) -> EnhancedMetricLogger:
    """Create enhanced metric logger with predefined loggers."""
    enhanced_logger = EnhancedMetricLogger(mode, print_log_freq)
    
    # Create standard loggers and add them
    standard_loggers = {
        "uc_action_loss": Logger("uc_action_loss", mode, window_size=print_log_freq),
        "uc_action_waypts_cos_sim": Logger("uc_action_waypts_cos_sim", mode, window_size=print_log_freq),
        "uc_multi_action_waypts_cos_sim": Logger("uc_multi_action_waypts_cos_sim", mode, window_size=print_log_freq),
        "gc_dist_loss": Logger("gc_dist_loss", mode, window_size=print_log_freq),
        "gc_action_loss": Logger("gc_action_loss", mode, window_size=print_log_freq),
        "gc_action_waypts_cos_sim": Logger("gc_action_waypts_cos_sim", mode, window_size=print_log_freq),
        "gc_multi_action_waypts_cos_sim": Logger("gc_multi_action_waypts_cos_sim", mode, window_size=print_log_freq),
        "gc_pose_loss": Logger("gc_pose_loss", mode, window_size=print_log_freq),
    }
    
    for key, logger in standard_loggers.items():
        enhanced_logger.add_logger(key, logger)
        
    return enhanced_logger


def display_epoch_summary(enhanced_logger: EnhancedMetricLogger, epoch: int):
    """Display a comprehensive epoch summary with grouped metrics."""
    summary = enhanced_logger.get_summary_metrics()
    
    if not summary:
        return
        
    print_section(f"Epoch {epoch + 1} Summary", Symbols.STAR, Colors.BRIGHT_YELLOW)
    
    for group_name, metrics in summary.items():
        group = enhanced_logger.groups[group_name]
        print(f"  {group.symbol} {group.color}{group.name}:{Colors.RESET}")
        
        for metric_name, values in metrics.items():
            latest = format_number(values['latest'], 4)
            avg = format_number(values['average'], 4)
            print(f"    {Colors.CYAN}{metric_name}:{Colors.RESET} {Colors.WHITE}{latest}{Colors.RESET} "
                  f"{Colors.GRAY}(epoch avg: {avg}){Colors.RESET}")
        print()
