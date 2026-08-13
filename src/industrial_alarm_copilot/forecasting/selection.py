'''Validation-only forecasting model selection policy.'''

import math

import pandas as pd


MODEL_COMPLEXITY_ORDER = {
    'global_frequency_v1': 0,
    'machine_frequency_v1': 1,
    'transition_frequency_v1': 2,
    'ovr_logistic_none_v1': 3,
    'ovr_logistic_balanced_v1': 3,
    'gru_none_v1': 4,
    'gru_balanced_capped_v1': 4,
}


def select_forecast_model(
    validation_metrics: pd.DataFrame,
    macro_f1_tolerance: float,
) -> pd.Series:
    '''Select highest micro-F1 within a near-best macro-F1 tolerance band.'''
    required = {'split', 'model_version', 'micro_f1_at_k', 'macro_f1_at_k'}
    if not required.issubset(validation_metrics.columns):
        raise ValueError('validation metrics are missing selection columns')
    if set(validation_metrics['split']) != {'validation'}:
        raise ValueError('model selection may use validation rows only')
    if validation_metrics['model_version'].duplicated().any():
        raise ValueError('validation model versions must be unique')
    if not math.isfinite(macro_f1_tolerance) or macro_f1_tolerance < 0:
        raise ValueError('macro F1 tolerance must be finite and nonnegative')
    if validation_metrics[['micro_f1_at_k', 'macro_f1_at_k']].isna().any().any():
        raise ValueError('selection metrics cannot contain missing values')

    candidates = validation_metrics.copy()
    unknown_models = set(candidates['model_version']).difference(
        MODEL_COMPLEXITY_ORDER
    )
    if unknown_models:
        raise ValueError('model complexity order is undefined')
    best_macro_f1 = float(candidates['macro_f1_at_k'].max())
    eligible = candidates.loc[
        candidates['macro_f1_at_k'].ge(best_macro_f1 - macro_f1_tolerance)
    ].copy()
    eligible['complexity_rank'] = eligible['model_version'].map(
        MODEL_COMPLEXITY_ORDER
    )
    eligible = eligible.sort_values(
        ['micro_f1_at_k', 'macro_f1_at_k', 'complexity_rank', 'model_version'],
        ascending=[False, False, True, True],
        kind='stable',
    )
    selected = eligible.iloc[0].copy()
    selected['best_macro_f1'] = best_macro_f1
    selected['macro_f1_tolerance'] = macro_f1_tolerance
    selected['eligible_model_count'] = len(eligible)
    return selected
