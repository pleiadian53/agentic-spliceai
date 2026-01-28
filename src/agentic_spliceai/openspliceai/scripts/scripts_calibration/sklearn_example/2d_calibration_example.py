from sklearn.datasets import make_classification
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


probabilities = model.predict_proba(X_test)[:, 1]  # Get the probability of the positive class


from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

prob_true, prob_pred = calibration_curve(y_test, probabilities, n_bins=10)

plt.plot(prob_pred, prob_true, marker='o', linewidth=1, label='Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot (before calibration)')
plt.legend()
plt.show()

from sklearn.calibration import CalibratedClassifierCV
calibrated_model = CalibratedClassifierCV(model, method='sigmoid', cv='prefit')
calibrated_model.fit(X_train, y_train)

calibrated_probabilities = calibrated_model.predict_proba(X_test)[:, 1]

calib_prob_true, calib_prob_pred = calibration_curve(y_test, calibrated_probabilities, n_bins=10)

plt.plot(calib_prob_pred, calib_prob_true, marker='o', linewidth=1, label='Calibrated Logistic Regression')
plt.plot([0, 1], [0, 1], linestyle='--', label='Perfectly calibrated')
plt.xlabel('Mean predicted probability')
plt.ylabel('Fraction of positives')
plt.title('Calibration plot (after calibration)')
plt.legend()
plt.savefig('calibration_plot.png')
