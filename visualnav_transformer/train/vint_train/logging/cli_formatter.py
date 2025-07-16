"""
Enhanced CLI formatting utilities for training output.
Provides colorful, informative, and visually appealing training progress display.
"""

import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
import sys
import os

# ANSI color codes
class Colors:
    # Basic colors
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    GRAY = '\033[90m'
    
    # Bright colors
    BRIGHT_RED = '\033[91;1m'
    BRIGHT_GREEN = '\033[92;1m'
    BRIGHT_YELLOW = '\033[93;1m'
    BRIGHT_BLUE = '\033[94;1m'
    BRIGHT_MAGENTA = '\033[95;1m'
    BRIGHT_CYAN = '\033[96;1m'
    BRIGHT_WHITE = '\033[97;1m'
    
    # Background colors
    BG_RED = '\033[101m'
    BG_GREEN = '\033[102m'
    BG_YELLOW = '\033[103m'
    BG_BLUE = '\033[104m'
    
    # Styles
    BOLD = '\033[1m'
    DIM = '\033[2m'
    UNDERLINE = '\033[4m'
    BLINK = '\033[5m'
    
    # Reset
    RESET = '\033[0m'
    END = '\033[0m'

# Unicode symbols
class Symbols:
    TRAIN = '🚂'
    EVAL = '🔍'
    SUCCESS = '✅'
    ERROR = '❌'
    WARNING = '⚠️'
    INFO = 'ℹ️'
    ROCKET = '🚀'
    CHART = '📊'
    CLOCK = '⏰'
    MEMORY = '🧠'
    GPU = '🔥'
    SAVE = '💾'
    LOAD = '📂'
    ARROW_RIGHT = '→'
    ARROW_DOWN = '↓'
    BULLET = '•'
    STAR = '⭐'
    LIGHTNING = '⚡'
    TARGET = '🎯'
    PROGRESS = '📈'
    LOSS = '📉'

def get_terminal_width():
    """Get terminal width, default to 80 if unable to determine."""
    try:
        return os.get_terminal_size().columns
    except:
        return 80

def format_time(seconds: float) -> str:
    """Format time duration in a human-readable way."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"

def format_number(num: float, precision: int = 4) -> str:
    """Format numbers with appropriate precision and scientific notation if needed."""
    if abs(num) >= 1000:
        return f"{num:.2e}"
    elif abs(num) >= 1:
        return f"{num:.{min(precision, 3)}f}"
    else:
        return f"{num:.{precision}f}"

def create_progress_bar(current: int, total: int, width: int = 30, 
                       fill_char: str = '█', empty_char: str = '░') -> str:
    """Create a visual progress bar."""
    if total == 0:
        return f"[{empty_char * width}] 0%"
    
    progress = current / total
    filled_width = int(width * progress)
    bar = fill_char * filled_width + empty_char * (width - filled_width)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}%"

def print_header(title: str, symbol: str = Symbols.ROCKET, color: str = Colors.BRIGHT_CYAN):
    """Print a formatted header."""
    width = get_terminal_width()
    title_with_symbol = f"{symbol} {title} {symbol}"
    padding = (width - len(title_with_symbol)) // 2
    border = "=" * width
    
    print(f"\n{color}{border}")
    print(f"{' ' * padding}{title_with_symbol}")
    print(f"{border}{Colors.RESET}\n")

def print_section(title: str, symbol: str = Symbols.BULLET, color: str = Colors.BRIGHT_YELLOW):
    """Print a formatted section header."""
    print(f"\n{color}{symbol} {Colors.BOLD}{title}{Colors.RESET}")
    print(f"{color}{'─' * (len(title) + 2)}{Colors.RESET}")

def print_info(message: str, symbol: str = Symbols.INFO, color: str = Colors.CYAN):
    """Print an info message."""
    print(f"{color}{symbol} {message}{Colors.RESET}")

def print_success(message: str, symbol: str = Symbols.SUCCESS, color: str = Colors.GREEN):
    """Print a success message."""
    print(f"{color}{symbol} {message}{Colors.RESET}")

def print_warning(message: str, symbol: str = Symbols.WARNING, color: str = Colors.YELLOW):
    """Print a warning message."""
    print(f"{color}{symbol} {message}{Colors.RESET}")

def print_error(message: str, symbol: str = Symbols.ERROR, color: str = Colors.RED):
    """Print an error message."""
    print(f"{color}{symbol} {message}{Colors.RESET}")

def format_metric_line(name: str, value: float, unit: str = "", 
                      color: str = Colors.WHITE, precision: int = 4) -> str:
    """Format a metric line with proper alignment."""
    formatted_value = format_number(value, precision)
    return f"  {color}{name:<25} {Colors.BRIGHT_WHITE}{formatted_value}{unit}{Colors.RESET}"

def print_metrics_table(metrics: Dict[str, float], title: str = "Metrics", 
                       symbol: str = Symbols.CHART):
    """Print metrics in a nicely formatted table."""
    print_section(title, symbol, Colors.BRIGHT_MAGENTA)
    
    for name, value in metrics.items():
        # Color code based on metric type
        if 'loss' in name.lower():
            color = Colors.RED if value > 1.0 else Colors.YELLOW
        elif 'acc' in name.lower() or 'sim' in name.lower():
            color = Colors.GREEN if value > 0.8 else Colors.YELLOW
        else:
            color = Colors.CYAN
            
        print(format_metric_line(name, value, color=color))

class TrainingProgressTracker:
    """Track and display training progress with enhanced formatting."""
    
    def __init__(self, total_epochs: int, project_name: str = "Training"):
        self.total_epochs = total_epochs
        self.project_name = project_name
        self.start_time = time.time()
        self.epoch_start_time = None
        self.current_epoch = 0
        
    def start_training(self):
        """Display training start banner."""
        print_header(f"Starting {self.project_name}", Symbols.ROCKET, Colors.BRIGHT_GREEN)
        print_info(f"Total epochs: {self.total_epochs}")
        print_info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
    def start_epoch(self, epoch: int, mode: str = "train"):
        """Display epoch start information."""
        self.current_epoch = epoch
        self.epoch_start_time = time.time()
        
        symbol = Symbols.TRAIN if mode == "train" else Symbols.EVAL
        color = Colors.BRIGHT_BLUE if mode == "train" else Colors.BRIGHT_MAGENTA
        
        progress_bar = create_progress_bar(epoch, self.total_epochs, width=40)
        elapsed_time = format_time(time.time() - self.start_time)
        
        print(f"\n{color}{'='*60}")
        print(f"{symbol} {Colors.BOLD}Epoch {epoch + 1}/{self.total_epochs} - {mode.upper()}{Colors.RESET}")
        print(f"{color}Progress: {progress_bar}")
        print(f"{Colors.GRAY}Elapsed: {elapsed_time}{Colors.RESET}")
        print(f"{color}{'='*60}{Colors.RESET}")
        
    def end_epoch(self, metrics: Dict[str, float]):
        """Display epoch completion information."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            print(f"\n{Colors.GREEN}{Symbols.SUCCESS} Epoch completed in {format_time(epoch_time)}{Colors.RESET}")
            
        # Display key metrics
        if metrics:
            key_metrics = {}
            for name, value in metrics.items():
                if any(key in name.lower() for key in ['loss', 'acc', 'sim']):
                    key_metrics[name] = value
            
            if key_metrics:
                print_metrics_table(key_metrics, "Epoch Summary", Symbols.STAR)
                
    def finish_training(self):
        """Display training completion banner."""
        total_time = time.time() - self.start_time
        print_header("Training Completed!", Symbols.SUCCESS, Colors.BRIGHT_GREEN)
        print_success(f"Total training time: {format_time(total_time)}")
        print_success(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()

def create_custom_tqdm_format():
    """Create a custom format string for tqdm progress bars."""
    return (
        f"{Colors.BRIGHT_CYAN}{{desc}}{Colors.RESET}: "
        f"{Colors.BRIGHT_WHITE}{{percentage:3.0f}}%{Colors.RESET}"
        f"{Colors.CYAN}|{{bar}}{Colors.RESET}"
        f"{Colors.BRIGHT_WHITE}| {{n_fmt}}/{{total_fmt}}{Colors.RESET} "
        f"{Colors.GRAY}[{{elapsed}}<{{remaining}}, {{rate_fmt}}{{postfix}}]{Colors.RESET}"
    )
