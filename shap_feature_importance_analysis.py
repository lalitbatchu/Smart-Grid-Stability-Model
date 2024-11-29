import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
import shap

# Read data
data = pd.read_csv(r"C:\Users\ravim\Downloads\archive\smart_grid_stability_augmented.csv")
print("Data read successfully.")

# Map target variable to binary
map1 = {'unstable': 0, 'stable': 1}
data['stabf'] = data['stabf'].replace(map1)

# Shuffle data
data = data.sample(frac=1)
print("Data shuffled.")

# Split data into features and target
X = data.iloc[:, :12]
y = data.iloc[:, 13]

# Print unique values in the target variable
print("Unique values in the target variable:", data['stabf'].unique())

# Filter to include only unstable outcomes
unstable_data = data[data['stabf'] == 0]
print("Number of unstable samples:", len(unstable_data))

# Split into features and target for unstable data
X_unstable = unstable_data.iloc[:, :12]
y_unstable = unstable_data.iloc[:, 13]

# Determine the number of training samples (80% of unstable data)
train_size = int(0.8 * len(X_unstable))
print("Training size determined:", train_size)

# Create grouped features for tau, p, and g
X_grouped = pd.DataFrame({
    'tau': unstable_data[['tau1', 'tau2', 'tau3', 'tau4']].mean(axis=1),
    'p': unstable_data[['p1', 'p2', 'p3', 'p4']].sum(axis=1),
    'g': unstable_data[['g1', 'g2', 'g3', 'g4']].mean(axis=1)
})
print("Grouped features created:\n", X_grouped.describe())

# Split into training and testing sets using grouped features
X_training = X_grouped.iloc[:train_size].values
y_training = y_unstable.iloc[:train_size].values
X_testing = X_grouped.iloc[train_size:].values
y_testing = y_unstable.iloc[train_size:].values

# Standardize features
scaler = StandardScaler()
X_training = scaler.fit_transform(X_training)
X_testing = scaler.transform(X_testing)
print("Features standardized.")

# ANN initialization
classifier = Sequential()
classifier.add(Dense(units=24, kernel_initializer='uniform', activation='relu', input_dim=3))  # Changed input_dim to 3
classifier.add(Dense(units=24, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=12, kernel_initializer='uniform', activation='relu'))
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# ANN compilation
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("ANN compiled.")

# K-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=10)
for train_index, val_index in kf.split(X_training):
    x_train, x_val = X_training[train_index], X_training[val_index]
    y_train, y_val = y_training[train_index], y_training[val_index]
    classifier.fit(x_train, y_train, epochs=10, verbose=0)  # Reduced epochs for faster fitting
print("K-fold cross-validation completed.")

# Optimize SHAP values calculation
# Use a smaller sample for the background dataset
background_data = shap.sample(X_training, 100)  # Using 100 samples for background
explainer = shap.KernelExplainer(classifier.predict, background_data)
shap_values = explainer.shap_values(X_testing[:100])  # Using only the first 100 test samples for SHAP values
print("SHAP values calculated.")

# Create a DataFrame for the average absolute SHAP values
shap_abs = np.abs(shap_values)
feature_importance = pd.DataFrame(np.mean(shap_abs, axis=0), index=['tau', 'p', 'g'], columns=["Importance"])  # Updated index

# Rank features by importance
feature_importance.sort_values(by="Importance", ascending=False, inplace=True)

# Display ranked features
print("Feature Importance (Ranked from Greatest to Least for Unstable Outcomes):")
print(feature_importance)

# Optional: Plot the feature importances
shap.summary_plot(shap_values, X_testing[:100], feature_names=['tau', 'p', 'g'])  # Updated feature names
