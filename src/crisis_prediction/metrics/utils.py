from numbers import Number
from typing import List, Dict, Union

import numpy as np
import pandas as pd


def explode_multiple_columns_same_length(data: pd.DataFrame, columns: List[str]):
    index_name = 'index' if data.index.name is None else data.index.name
    data.index.name = 'index'
    df = data.reset_index(drop=False)
    results = [df[column].explode() for column in columns]
    length_first_elem = len(results[0])
    if not all([len(elem) == length_first_elem for elem in results]):
        raise ValueError('Not all list in columns have the same length.')
    results = pd.concat(results, axis=1)
    results = df.drop(columns=columns).join(results).set_index('index')
    results.index.name = index_name
    results = results.reindex(columns=data.columns, copy=False)
    return results


def compute_mean_metrics(metrics_full_experiment: List[Dict[str, Union[List, Number]]]):
    metrics_full_avg = {}
    for test_run_dict in metrics_full_experiment:
        for key, value in test_run_dict.items():
            if key in metrics_full_avg:
                metrics_full_avg[key].append(np.mean(value))
            else:
                metrics_full_avg[key] = [np.mean(value)]
    metrics_avg = {}
    for key in metrics_full_avg.keys():
        if key not in ['end_test', 'start_test']:
            metrics_avg[key] = np.mean(metrics_full_avg.get(key))
    return metrics_avg
