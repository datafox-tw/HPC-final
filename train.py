"""
train.py: The code to train the DeepAR volatility forecasting model.

data: the datasets consist of two parts: the garch part and the alpha-101 part.
- the garch part is the autoregressive inputs, which are the past volatility values.
- the alpha-101 part is the covariate inputs, which are the alpha-101 factors.

our goal: to predict the future volatility values based on the past volatility values and the alpha-101 factors by using DeepAR structure.
"""


