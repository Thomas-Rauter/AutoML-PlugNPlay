import pandas as pd
from sklearn.datasets import make_classification
import numpy as np


def create_ml_dataset():
    # Generate a classification dataset
    X, y = make_classification(n_samples=100, n_features=4, n_informative=4, n_redundant=0, n_classes=3, random_state=42)

    # Scale features to match the ranges: temperature (20-30), methanol concentration (0.5-5), induction time (1-24), pH (5-7)
    X_scaled = np.copy(X)
    X_scaled[:, 0] = X[:, 0] * 5 + 25  # Scale to 20-30
    X_scaled[:, 1] = X[:, 1] * 2.25 + 2.75  # Scale to 0.5-5
    X_scaled[:, 2] = X[:, 2] * 11.5 + 12.5  # Scale to 1-24
    X_scaled[:, 3] = X[:, 3] + 6  # Scale to 5-7

    # Map numeric labels to categorical
    label_map = {0: 'low', 1: 'medium', 2: 'high'}
    labels = [label_map[label] for label in y]

    # Create DataFrame
    df = pd.DataFrame(X_scaled, columns=['Temperature', 'Methanol Concentration', 'Induction Time', 'pH'])
    df['Expression Level'] = labels

    return df


# Create DataFrame
local_df = create_ml_dataset()

# Save to CSV file
local_df.to_csv('ml_protein_expression_data.csv', index=False)

print("ML DataFrame created and saved as 'ml_protein_expression_data.csv'")
