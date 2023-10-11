# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28, 2023
@author: erdem
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Data Inclusion
# Import the cancer data from the specified path.
data = pd.read_csv("Cancer_Data.csv")

#%%
# Remove the "Unnamed: 32" and "id" columns as they are not needed.
data.drop(["Unnamed: 32", "id"], axis=1, inplace=True)

#%%
# Separate the data into malignant (M) and benign (B) cases for visualization.
M = data[data.diagnosis == "M"]
B = data[data.diagnosis == "B"]

# Scatter Plot
# Create a scatter plot to visualize the relationship between "radius_mean" and "texture_mean."
plt.scatter(M.radius_mean, M.texture_mean, color="red", label="malignant", alpha=0.3)
plt.scatter(B.radius_mean, B.texture_mean, color="green", label="benign", alpha=0.3)
plt.xlabel("Radius Mean")
plt.ylabel("Texture Mean")
plt.legend()
plt.show()

#%%
# Assign binary labels (1 for malignant, 0 for benign) to the "diagnosis" column.
data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]
y = data.diagnosis.values
x_data = data.drop(["diagnosis"], axis=1)

# Normalization
# Normalize the features in the dataset to ensure consistent scaling.
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

# Train-Test Split
# Split the dataset into training and testing sets.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=42)

# Decision Tree Classifier
# Create a Decision Tree Classifier and fit it to the training data.
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(x_train, y_train)

# Calculate and print the accuracy score.
print("Accuracy Score:", dt.score(x_test, y_test))
