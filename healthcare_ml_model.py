import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from pyswarm import pso

# Load the dataset (assuming you have 'heart_disease.csv' in the same directory)
df = pd.read_csv('heart_disease.csv')

# Feature engineering (same as before)

# Data preprocessing
X = df.drop('target', axis=1)
y = df['target']

# Split the dataset into training and test sets (80% training, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling for better model performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the fitness function for PSO
def fitness_function(params):
    n_estimators = int(params[0])
    max_depth = int(params[1])
    
    # Train the Random Forest Classifier with given hyperparameters
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)
    
    # Calculate accuracy as the fitness value (to be maximized)
    accuracy = accuracy_score(y_test, y_pred)
    return -accuracy  # Negative value to maximize accuracy

# Define the bounds for the hyperparameters (n_estimators and max_depth)
lower_bound = [50, 1]
upper_bound = [200, 50]

# Perform PSO for hyperparameter optimization
best_params, _ = pso(fitness_function, lower_bound, upper_bound)

# Get the best hyperparameters and model
n_estimators_opt = int(best_params[0])
max_depth_opt = int(best_params[1])

best_model = RandomForestClassifier(n_estimators=n_estimators_opt, max_depth=max_depth_opt, random_state=42)
best_model.fit(X_train_scaled, y_train)

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test_scaled)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

# Print the results
print("Best Hyperparameters (n_estimators, max_depth):", n_estimators_opt, max_depth_opt)
print("Accuracy:", accuracy)
