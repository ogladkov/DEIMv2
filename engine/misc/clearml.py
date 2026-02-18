"""
DEIM: DETR with Improved Matching for Fast Convergence
Copyright (c) 2024 The DEIM Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
---------------------------------------------------------------------------------
ClearML integration for experiment tracking.
Based on Ultralytics YOLO ClearML integration.
"""

import os
from typing import Optional

try:
    import clearml
    from clearml import Task
    CLEARML_AVAILABLE = True
except (ImportError, AttributeError):
    CLEARML_AVAILABLE = False


class ClearMLLogger:
    """
    ClearML Logger for DEIMv2 training.

    Automatically logs hyperparameters, training metrics (loss, learning rate),
    evaluation metrics (COCO mAP), and model artifacts to ClearML.

    Usage:
        export CLEARML_API_KEY=your_api_key
        export CLEARML_API_SECRET=your_api_secret

        logger = ClearMLLogger(
            project_name='DEIMv2',
            experiment_name='deimv2_dinov3_s',
            config=config_dict,
        )
        logger.log_scalar('Loss/total', 0.5, step=100)
    """

    def __init__(
        self,
        project_name: Optional[str] = None,
        experiment_name: Optional[str] = None,
        config: Optional[dict] = None,
        enabled: bool = True,
    ):
        self.enabled = enabled and CLEARML_AVAILABLE
        self.task = None

        if not self.enabled:
            return

        try:
            api_key = os.environ.get('CLEARML_API_KEY')
            api_secret = os.environ.get('CLEARML_API_SECRET')
            clearml_server = os.environ.get('CLEARML_API_SERVER', 'https://api.clear.ml')

            if api_key and api_secret:
                os.environ['CLEARML_API_KEY'] = api_key
                os.environ['CLEARML_API_SECRET'] = api_secret
                os.environ['CLEARML_API_HOST'] = clearml_server

            self.task = Task.init(
                project_name=project_name or 'DEIMv2',
                task_name=experiment_name,
                output_uri=True,
            )

            if config:
                self.log_parameters(config)

            print(f'ClearML logging initialized. Task ID: {self.task.id}')

        except Exception as e:
            print(f'ClearML initialization failed: {e}')
            self.enabled = False
            self.task = None

    def is_enabled(self) -> bool:
        """Check if ClearML logging is active."""
        return self.enabled and self.task is not None

    def log_parameters(self, params: dict, prefix: str = '') -> None:
        """
        Log hyperparameters to ClearML.

        Args:
            params: Dictionary of parameters to log.
            prefix: Optional prefix for parameter names.
        """
        if not self.is_enabled():
            return

        try:
            flattened = self._flatten_dict(params, prefix)
            self.task.connect(flattened)
        except Exception as e:
            print(f'ClearML: failed to log parameters: {e}')

    def log_metrics(self, metrics: dict, step: int = None, epoch: int = None) -> None:
        """
        Log a dictionary of metrics to ClearML.

        Args:
            metrics: Dictionary of metric names and values.
            step: Global step (iteration).
            epoch: Epoch number (used as iteration when step is None).
        """
        if not self.is_enabled():
            return

        iteration = step if step is not None else (epoch if epoch is not None else 0)
        for k, v in metrics.items():
            try:
                if isinstance(v, (int, float)):
                    scalar = float(v)
                elif hasattr(v, 'item'):
                    scalar = float(v.item())
                else:
                    continue
                self.log_scalar(k, scalar, step=iteration)
            except Exception as e:
                print(f'ClearML: failed to log metric {k}: {e}')

    def log_scalar(self, name: str, value: float, step: int = None) -> None:
        """
        Log a single scalar value.

        Args:
            name: Metric name, e.g. 'Loss/total'. A '/' splits into title/series.
            value: Scalar value.
            step: Step number used as the x-axis iteration.
        """
        if not self.is_enabled():
            return

        try:
            if '/' in name:
                title, series = name.split('/', 1)
            else:
                title, series = name, 'value'

            self.task.get_logger().report_scalar(
                title=title,
                series=series,
                value=float(value),
                iteration=step or 0,
            )
        except Exception as e:
            print(f'ClearML: failed to log scalar {name}: {e}')

    def log_image(self, name: str, image_path: str, step: int = None) -> None:
        """Log an image to ClearML."""
        if not self.is_enabled():
            return

        try:
            self.task.get_logger().report_image(
                title=name,
                series='image',
                image=image_path,
                iteration=step or 0,
            )
        except Exception as e:
            print(f'ClearML: failed to log image {name}: {e}')

    def log_model(self, model_path: str, model_name: str = None) -> None:
        """Upload a model checkpoint as a ClearML artifact."""
        if not self.is_enabled():
            return

        try:
            name = model_name or os.path.basename(model_path)
            self.task.upload_artifact(name=name, artifact_object=model_path)
        except Exception as e:
            print(f'ClearML: failed to log model: {e}')

    def log_table(self, table_name: str, data: dict, step: int = None) -> None:
        """Log a table to ClearML."""
        if not self.is_enabled():
            return

        try:
            import pandas as pd
            df = pd.DataFrame(data)
            self.task.get_logger().report_table(
                title=table_name,
                series='table',
                table_plot=df,
                iteration=step or 0,
            )
        except Exception as e:
            print(f'ClearML: failed to log table {table_name}: {e}')

    def finish(self) -> None:
        """Close the ClearML task."""
        if self.is_enabled():
            try:
                self.task.close()
                print('ClearML task closed.')
            except Exception as e:
                print(f'ClearML: failed to close task: {e}')

    def _flatten_dict(self, d: dict, parent_key: str = '', sep: str = '/') -> dict:
        """Flatten a nested dictionary into a single-level dict."""
        items = []
        for k, v in d.items():
            new_key = f'{parent_key}{sep}{k}' if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


def setup_clearml(args, config: dict) -> Optional[ClearMLLogger]:
    """
    Create a ClearMLLogger from parsed CLI arguments and the training config.

    Args:
        args: Parsed argparse namespace.
        config: YAML config dictionary to log as hyperparameters.

    Returns:
        ClearMLLogger instance, or None when ClearML is disabled.
    """
    if not getattr(args, 'clearml', False):
        return None

    return ClearMLLogger(
        project_name=getattr(args, 'clearml_project', 'DEIMv2'),
        experiment_name=getattr(args, 'clearml_experiment', None),
        config=config,
        enabled=True,
    )


has_clearml = CLEARML_AVAILABLE
