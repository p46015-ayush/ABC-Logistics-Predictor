"""Input names and validation shared by training and prediction."""
import numpy as np
import pandas as pd

TARGET = 'Delivery_Delay'
FEATURES = ['Delivery_Distance', 'Traffic_Congestion', 'Weather_Condition',
            'Delivery_Slot', 'Driver_Experience', 'Num_Stops', 'Vehicle_Age',
            'Road_Condition_Score', 'Package_Weight', 'Fuel_Efficiency',
            'Warehouse_Processing_Time']
CATEGORICAL = ['Weather_Condition', 'Delivery_Slot']
NUMERIC = [c for c in FEATURES if c not in CATEGORICAL]

def validate_features(frame):
    """Reject invalid values; retain missing values for the fitted imputers."""
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f'Missing required columns: {missing}')
    if frame.empty:
        raise ValueError('The input file has no deliveries.')
    x = frame[FEATURES].copy()
    for c in FEATURES:
        x[c] = pd.to_numeric(x[c], errors='raise')
        if np.isinf(x[c].to_numpy(dtype=float)).any():
            raise ValueError(f'{c}: infinity is not allowed.')
    for c, allowed in {'Weather_Condition': [1, 2, 3], 'Delivery_Slot': [1, 2, 3],
                       'Traffic_Congestion': [1, 2, 3, 4, 5],
                       'Road_Condition_Score': [1, 2, 3, 4, 5]}.items():
        if not x[c].dropna().isin(allowed).all():
            raise ValueError(f'{c} must be one of {allowed}.')
    for c in NUMERIC:
        if (x[c].dropna() < 0).any():
            raise ValueError(f'{c} cannot be negative.')
    for c in ['Num_Stops']:
        if (x[c].dropna() % 1 != 0).any():
            raise ValueError(f'{c} must be an integer.')
    if (x['Fuel_Efficiency'].dropna() <= 0).any():
        raise ValueError('Fuel_Efficiency must be positive.')
    if x.isna().all(axis=1).any():
        raise ValueError('At least one row has no input values.')
    return x
