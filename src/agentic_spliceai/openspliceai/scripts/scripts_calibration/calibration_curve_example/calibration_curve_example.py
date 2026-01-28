# Load the necessary libraries 
from sklearn.datasets import make_classification 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.calibration import calibration_curve 
import matplotlib.pyplot as plt 
  
# Generate a synthetic 3-class classification dataset 
X, y = make_classification(n_samples=1000, 
                           n_classes=3,  
                           n_features=10,  
                           n_informative=5,  
                           random_state=42) 
  
# Split the data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y,  
                                                    test_size=0.2,  
                                                    random_state=42) 
  
# Train a Logistic Regression 
clf = LogisticRegression(max_iter=12000) 
clf.fit(X_train, y_train) 
  
# Find the predicted probabilities on the testing set 
probabilities = clf.predict_proba(X_test) 
  
  
# Compute the calibration curve for each class 
calibration_curve_values = [] 
for i in range(3): 
    curve = calibration_curve(y_test == i,  
                              probabilities[:, i],  
                              n_bins=20,  
                              pos_label=True) 
    calibration_curve_values.append(curve) 
  
# Plot the calibration curves 
fig, axs = plt.subplots(1, 3, figsize=(17,5)) 
for i in range(3): 
    axs[i].plot(calibration_curve_values[i][1],  
                calibration_curve_values[i][0],  
                marker='o') 
    axs[i].plot([0, 1], [0, 1], linestyle='--') 
    axs[i].set_xlim([0, 1]) 
    axs[i].set_ylim([0, 1]) 
    axs[i].set_title(f"Class {i}", fontsize = 17) 
    axs[i].set_xlabel("Predicted probability", fontsize = 15) 
    axs[i].set_ylabel("True probability", fontsize = 15) 
plt.tight_layout() 
plt.savefig('calibration_curve_example.png')