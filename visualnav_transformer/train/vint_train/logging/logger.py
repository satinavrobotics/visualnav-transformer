import numpy as np

# Import CLI formatter for enhanced display
try:
    from visualnav_transformer.train.vint_train.logging.cli_formatter import format_number
except ImportError:
    # Fallback if cli_formatter is not available
    def format_number(num: float, precision: int = 4) -> str:
        if abs(num) >= 1000:
            return f"{num:.2e}"
        elif abs(num) >= 1:
            return f"{num:.{min(precision, 3)}f}"
        else:
            return f"{num:.{precision}f}"


class Logger:
    def __init__(
        self,
        name: str,
        dataset: str,
        window_size: int = 10,
        rounding: int = 4,
    ):
        """
        Args:
            name (str): Name of the metric
            dataset (str): Name of the dataset
            window_size (int, optional): Size of the moving average window. Defaults to 10.
            rounding (int, optional): Number of decimals to round to. Defaults to 4.
        """
        self.data = []
        self.name = name
        self.dataset = dataset
        self.rounding = rounding
        self.window_size = window_size

    def display(self) -> str:
        latest = format_number(self.latest(), self.rounding)
        average = format_number(self.average(), self.rounding)
        moving_average = format_number(self.moving_average(), self.rounding)
        output = f"{self.full_name()}: {latest} - {self.window_size}pt moving_avg: {moving_average} - avg: {average}"
        return output

    def log_data(self, data: float):
        if not np.isnan(data):
            self.data.append(data)

    def full_name(self) -> str:
        return f"{self.name}-{self.dataset}"

    def latest(self) -> float:
        if len(self.data) > 0:
            return self.data[-1]
        return np.nan

    def average(self) -> float:
        if len(self.data) > 0:
            return np.mean(self.data)
        return np.nan

    def moving_average(self) -> float:
        if len(self.data) > self.window_size:
            return np.mean(self.data[-self.window_size :])
        return self.average()
