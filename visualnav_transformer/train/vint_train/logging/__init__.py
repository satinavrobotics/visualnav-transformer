"""
Logging utilities for ViNT training.
Provides enhanced CLI formatting and organized metric logging.
"""

import warnings

# Suppress PyTorch transformer warnings about nested tensors
# This warning appears when norm_first=True is used in TransformerEncoderLayer
warnings.filterwarnings(
    "ignore",
    message="enable_nested_tensor is True, but self.use_nested_tensor is False because.*",
    category=UserWarning,
    module="torch.nn.modules.transformer"
)

from .logger import Logger
from .cli_formatter import (
    Colors, Symbols, print_info, print_success, print_warning, print_error,
    print_section, print_header, format_number, format_time
)
from .enhanced_logger import (
    EnhancedMetricLogger, create_enhanced_loggers, display_epoch_summary
)

__all__ = [
    'Logger',
    'Colors', 'Symbols', 
    'print_info', 'print_success', 'print_warning', 'print_error',
    'print_section', 'print_header', 
    'format_number', 'format_time',
    'EnhancedMetricLogger', 'create_enhanced_loggers', 'display_epoch_summary'
]
