import mlflow
import numpy as np
from pytorch_lightning.loggers import MLFlowLogger as PLMLFlowLogger
from pytorch_lightning.utilities.rank_zero import rank_zero_only


class MLFlowLogger(PLMLFlowLogger):
    """Extended MLFlow logger with artifact logging support"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @rank_zero_only
    def log_artifact(self, name, filepath):
        """Log an artifact to MLflow"""
        try:
            # Use the underlying MLflow client to log artifacts
            mlflow.log_artifact(filepath, artifact_path=name)
        except Exception as e:
            print(f"Warning: Failed to log artifact {name}: {e}")

    @rank_zero_only
    def log_metric(self, name, value, step=None):
        """Log a single metric to MLflow with NaN/infinite value filtering"""
        if isinstance(value, str):
            print(f"Warning: Discarding metric with string value {name}={value}")
            return

        # Filter out NaN and infinite values
        if np.isnan(value) or not np.isfinite(value):
            print(f"Warning: Discarding metric with NaN/infinite value {name}={value}")
            return

        # Use the parent class method for logging metrics
        metrics = {name: value}
        self.log_metrics(metrics, step)
