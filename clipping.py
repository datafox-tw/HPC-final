# to use Gaussian or Student-T distributions, we need to first do log transformation
# to prevent issues with 0 values, we need to add a small constant before log transformation
# but we need to know how small the constant should be
# reed the dataset and find the minimum value
import pandas as pd
import numpy as np
data = pd.read_csv('Dataset/data/ml_dataset_alpha101_volatility.csv')
# the feature is var_true_90
min_value = data['var_true_90'].min()
# print(f"The minimum value of var_true_90 is: {min_value}")
# output: The minimum value of var_true_90 is: 8.492946684691027e-37

# we want to print out the distribution of the magnitude of the values var_true_90
data['magnitude'] = data['var_true_90'].apply(lambda x: np.floor(np.log10(x)) if x > 0 else np.nan)
magnitude_counts = data['magnitude'].value_counts().sort_index()
# print("Distribution of the magnitude of var_true_90:")
# for mag, count in magnitude_counts.items():
#     print(f"10^{int(mag)}: {count} values")
# output:
"""
Distribution of the magnitude of var_true_90:
10^-37: 1 values
10^-36: 1 values
10^-19: 1 values
10^-17: 1 values
10^-16: 1 values
10^-15: 1 values
10^-14: 6 values
10^-13: 9 values
10^-12: 45 values
10^-11: 95 values
10^-10: 400 values
10^-9: 1174 values
10^-8: 3680 values
10^-7: 10907 values
10^-6: 30319 values
10^-5: 74009 values
10^-4: 81601 values
10^-3: 22120 values
10^-2: 341 values
"""

# we try to clip the values smaller than 1e-12
data['var_true_90_clipped'] = data['var_true_90'].clip(lower=1e-12)
min_value_clipped = data['var_true_90_clipped'].min()
# print(f"The minimum value of var_true_90 after clipping is: {min_value_clipped}")

# store the clipped dataset into alpha101_volatility_clipped.csv
data.to_csv('Dataset/data/ml_dataset_alpha101_volatility_clipped.csv', index=False)