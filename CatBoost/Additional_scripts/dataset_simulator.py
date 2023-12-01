import pandas as pd
import numpy as np
import random


# Function to create a DataFrame with the specified columns and data
def create_dataframe():
    # Define realistic ranges for each parameter
    temperature_range = (20, 30)  # in degrees Celsius
    methanol_concentration_range = (0.5, 5)  # in percent
    induction_time_range = (1, 24)  # in hours
    pH_range = (5, 7)  # pH scale

    # Initialize an empty DataFrame
    df = pd.DataFrame(columns=['Temperature', 'Methanol Concentration', 'Induction Time', 'pH', 'Expression Level'])

    # Generate 50 rows of data
    for _ in range(1000):
        temperature = np.random.uniform(*temperature_range)
        methanol_concentration = np.random.uniform(*methanol_concentration_range)
        induction_time = np.random.uniform(*induction_time_range)
        pH = np.random.uniform(*pH_range)
        expression_level = random.choice(['high', 'medium', 'low'])

        # Create a new row DataFrame
        new_row = pd.DataFrame([{'Temperature': temperature,
                                 'Methanol Concentration': methanol_concentration,
                                 'Induction Time': induction_time,
                                 'pH': pH,
                                 'Expression Level': expression_level}])

        # Concatenate the new row with the existing DataFrame
        df = pd.concat([df, new_row], ignore_index=True)

    return df


# Create DataFrame
local_df = create_dataframe()

# Save to CSV file
local_df.to_csv('protein_expression_data.csv', index=False)

print("DataFrame created and saved as 'protein_expression_data.csv'")
