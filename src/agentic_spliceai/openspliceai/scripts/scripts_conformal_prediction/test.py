import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X, y = make_regression(n_samples=1000, n_features=10, noise=0.1)
X_train, X_cal = X[:800], X[800:]
y_train, y_cal = y[:800], y[800:]

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on the calibration set
y_cal_pred = model.predict(X_cal)
errors = np.abs(y_cal_pred - y_cal)

# Determine the 95% quantile of the errors
quantile = np.quantile(errors, 0.95)

# Make a new prediction and compute the conformal interval
x_new = np.random.rand(1, 10)
y_new_pred = model.predict(x_new)
interval = (y_new_pred - quantile, y_new_pred + quantile)
print("Prediction interval:", interval)