import torch
import warnings
from typing import Any
import numpy as np


def apply_numpy_compatibility_fixes() -> None:
    """
    NumPy 2.0 compatibility fix for RecBole.
    MUST be applied before any RecBole imports.
    Add all deprecated aliases that were removed in NumPy 2.0 but RecBole still uses.
    """
    numpy_aliases = {
        'float': np.float64,
        'int': np.int64,
        'complex': np.complex128,
        'bool': np.bool_,
        'float_': np.float64,
        'int_': np.int64,
        'complex_': np.complex128,
        'bool_': np.bool_,
        'unicode_': np.str_,
        'unicode': np.str_
    }
    
    for alias, target in numpy_aliases.items():
        if not hasattr(np, alias):
            setattr(np, alias, target)


def apply_pytorch_fixes() -> None:
    """Fix PyTorch FutureWarnings via monkey patching."""
    
    # Fix torch.load weights_only warning
    _original_torch_load = torch.load
    def _patched_torch_load(*args: Any, **kwargs: Any) -> Any:
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Fix torch.cuda.amp.GradScaler deprecation  
    if hasattr(torch.cuda, 'amp') and hasattr(torch.cuda.amp, 'GradScaler'):
        _original_cuda_grad_scaler = torch.cuda.amp.GradScaler
        def _patched_cuda_grad_scaler(*args: Any, **kwargs: Any) -> Any:
            return torch.amp.GradScaler('cuda', *args, **kwargs)
        torch.cuda.amp.GradScaler = _patched_cuda_grad_scaler


def setup_warning_filters() -> None:
    """Suppress pandas FutureWarnings."""
    warnings.filterwarnings('ignore', category=FutureWarning, 
                           message='.*chained assignment.*inplace method.*')
    warnings.filterwarnings('ignore', category=FutureWarning, 
                           message='.*using.*len.*in Series.agg.*deprecated.*')


def setup_environment() -> None:
    """Apply all compatibility fixes and warning filters."""
    apply_numpy_compatibility_fixes()
    apply_pytorch_fixes()
    setup_warning_filters()


def dict2str(result_dict):
    """Convert result dict to table format string with percentage in header."""
    
    # Group metrics by type and topk values
    metrics_grouped = {}
    
    for metric_key, value in result_dict.items():
        # Parse metric name and topk (e.g., "recall@10" -> ("recall", "10"))
        if '@' in metric_key:
            metric_name, topk = metric_key.split('@')
            if metric_name not in metrics_grouped:
                metrics_grouped[metric_name] = {}
            metrics_grouped[metric_name][int(topk)] = value
        else:
            # For metrics without @k
            if 'other' not in metrics_grouped:
                metrics_grouped['other'] = {}
            metrics_grouped['other'][metric_key] = value
    
    if not metrics_grouped:
        return ""
    
    # Define the order of metrics with proper spelling
    metric_order = ['recall', 'mrr', 'ndcg', 'hit', 'precision']
    metric_display_names = {
        'recall': 'Recall',
        'mrr': 'MRR', 
        'ndcg': 'NDCG',
        'hit': 'HR',
        'precision': 'Precision'
    }
    
    # Get all unique topk values and sort them
    all_topks = set()
    for metric_data in metrics_grouped.values():
        if isinstance(metric_data, dict):
            all_topks.update(metric_data.keys())
    all_topks = sorted(all_topks)
    
    if not all_topks:
        return str(result_dict)
    
    # Create table header with percentage indicator
    header = ["Metric (%)"] + [f"@{k}" for k in all_topks]
    
    # Calculate column widths
    col_widths = [len(col) for col in header]
    
    # Collect table data and update column widths
    table_data = []
    for metric_name in metric_order:
        if metric_name in metrics_grouped:
            # Use the proper display name from metric_display_names
            display_name = metric_display_names.get(metric_name, metric_name)
            row = [display_name]
            for topk in all_topks:
                if topk in metrics_grouped[metric_name]:
                    # Convert to percentage but without % symbol
                    value_percent = metrics_grouped[metric_name][topk] * 100
                    value_str = f"{value_percent:.2f}"
                else:
                    value_str = "-"
                row.append(value_str)
            table_data.append(row)
            
            # Update column widths
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Handle other metrics not in predefined order
    for metric_name, topk_values in metrics_grouped.items():
        if metric_name not in metric_order and metric_name != 'other':
            display_name = metric_display_names.get(metric_name, metric_name.capitalize())
            row = [display_name]
            for topk in all_topks:
                if topk in topk_values:
                    # Convert to percentage but without % symbol
                    value_percent = topk_values[topk] * 100
                    value_str = f"{value_percent:.2f}"
                else:
                    value_str = "-"
                row.append(value_str)
            table_data.append(row)
            
            # Update column widths
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(cell))
    
    # Build table string
    result_lines = []
    
    # Add top border
    border_line = "+" + "+".join(["-" * (w + 2) for w in col_widths]) + "+"
    result_lines.append(border_line)
    
    # Add header
    header_line = "|" + "|".join([f" {header[i]:<{col_widths[i]}} " for i in range(len(header))]) + "|"
    result_lines.append(header_line)
    
    # Add separator
    result_lines.append(border_line)
    
    # Add data rows
    for row in table_data:
        data_line = "|" + "|".join([f" {row[i]:<{col_widths[i]}} " for i in range(len(row))]) + "|"
        result_lines.append(data_line)
    
    # Add bottom border
    result_lines.append(border_line)
    
    # Handle 'other' metrics (without @k) if any
    if 'other' in metrics_grouped:
        result_lines.append("")
        result_lines.append("Other metrics:")
        for key, value in metrics_grouped['other'].items():
            if isinstance(value, (int, float)):
                result_lines.append(f"  {key}: {value * 100:.2f}")
            else:
                result_lines.append(f"  {key}: {value}")
    
    return "\n".join(result_lines)


def set_color(log, color, highlight=True):
    """Add color to log message."""
    color_set = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        index = color_set.index(color)
    except:
        index = len(color_set) - 1
    prev_log = "\033["
    if highlight:
        prev_log += "1;3"
    else:
        prev_log += "0;3"
    prev_log += str(index) + "m"
    return prev_log + log + "\033[0m"

