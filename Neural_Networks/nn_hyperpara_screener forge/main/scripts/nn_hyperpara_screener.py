#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse
import sys
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape, A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import Image
from reportlab.pdfgen import canvas
import os
import numpy as np
import glob
import optuna
import subprocess
import torch.optim as optim


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Section, where all the functions and classes are stored. All the function calls and class instantiations are below this section.


def round_to_three_custom(fn1_num):
    """
    Description:
        Modify a floating-point number to have at most three non-zero digits after the decimal point,
        including handling numbers in scientific notation.

    Input:
        fn1_num (float): The number to be modified.

    Output:
        float: The modified number with at most three non-zero digits after the decimal point.

    Function-code:
        fn1_
    """

    if isinstance(fn1_num, (float, np.float64, np.float32)):
        # Check if the number is in scientific notation
        if 'e' in str(fn1_num):
            fn1_num = format(fn1_num, '.3e')  # Convert to scientific notation with 3 decimal places
            return fn1_num
        else:
            fn1_num_str = str(fn1_num)
            if '.' in fn1_num_str:
                fn1_whole, fn1_decimal = fn1_num_str.split('.')  # Splitting the number into whole and decimal parts
                fn1_non_zero_count = 0
                fn1_found_non_zero_digit_before_dot = False

                for fn1_digit in fn1_whole:  # Loop over the whole number part
                    if fn1_digit != '0':
                        fn1_found_non_zero_digit_before_dot = True

                for fn1_local_i, fn1_digit in enumerate(fn1_decimal):
                    if fn1_digit != '0':
                        fn1_non_zero_count += 1

                    if fn1_non_zero_count == 3:
                        # Keeping up to the third non-zero digit and truncating the rest
                        fn1_new_decimal = fn1_decimal[:fn1_local_i + 1]
                        return float(fn1_whole + '.' + fn1_new_decimal)

                    if fn1_local_i == 2 and fn1_found_non_zero_digit_before_dot:
                        fn1_new_decimal = fn1_decimal[:fn1_local_i + 1]
                        return float(fn1_whole + '.' + fn1_new_decimal)

                return float(fn1_num_str)  # Return the original number if less than 3 non-zero digits
            else:
                return int(fn1_num_str)  # Return the original number if no decimal part
    else:
        return fn1_num


def create_config():
    """
    Description:
        This function creates a configuration file named 'nn_hyperpara_screener_config.csv' for neural network hyperparameters.
        If the file does not exist, it is created with headers and initial values for a starter model.

    Input:
        None

    Output:
        nn_hyperpara_screener_config.csv file

    Function-code:
        fn2_
    """

    fn2_filename = 'nn_hyperpara_screener_config.csv'
    if not os.path.exists(fn2_filename):
        with open(fn2_filename, 'w') as fn2_file:
            fn2_headers = ['model_name', 'neurons', 'acti_fun', 'acti_fun_out', 'epochs', 'batch_size', 'noise',
                           'optimizer', 'alpha', 'decay_rate', 'lamda', 'dropout', 'psi', 'cost_fun']
            fn2_file.write(','.join(fn2_headers))
            fn2_starter_model = ["\nModel_1", '"10,1"', "ReLU", "Linear", 100, 64, 0, "Adam,0.9,0.99", 0.01, 0, 0.001, 0, 1, "MSELoss"]
            fn2_file.write(','.join([str(item) for item in fn2_starter_model]))


def parse_command_line_args(fn3_arguments):
    """
    Description:
        Parse command line arguments for train_set_name, and dev_set_name, dev_set_size.

    Input:
        fn3_arguments (list): A list of command line arguments.

    Output:
        float: The local_dev_set_size as a fraction.
        str: The local_train_set_name.
        str: The local_dev_set_name.

    Function-code:
        fn3_
    """

    if len(fn3_arguments) == 1 and fn3_arguments[0] == 'help':

        print("Usage: python3 nn_hyperpara_screener.py train_dataset dev_dataset")
        print("Options: python3 nn_hyperpara_screener.py help        -->     Prints info messages to the terminal and terminate script.")
        print("Options: python3 nn_hyperpara_screener.py config                                -->     Creates config file for manually exploring the hyperparameters and terminate script.")
        print("--plots   -->   Show loss plots during training; --optuna=10   -->   runs Optuna optimizer instead of manually tuning hyperparameters. 10 is the amount of trials for the Optuna study.")
        print("##################################################################################################################################################################\n")
        print("Further info about the script:")
        print("Input: train-dataset, dev-dataset, and respective config file (with list of models or ranges for Optuna) or only train-dataset and config file in the case of k-fold cross-validation")
        print("Output: A single Pdf-report about the performance of the models in the case of a list of models or the best model in the case of Optuna and .pth files of the models in the first case")
        print("This script takes as command line arguments the filename of the train set and the filename of the dev set located in current working dir (cwd).")
        print("Additionally, it requires the file nn_hyperpara_screener_config.csv to be present in the cwd.")
        print("It outputs 1 pdf-file, called nn_hyperpara_screener_output__ followed by the current date-time.")
        print("This pdf contains the hyperparameters of the different models and their performance on the dev set, togehter with loss-plots.")
        print("The purpose of this script is to simplify and streamline the process of 'screening' for the best set of hyperparameters for a neural network, on a given dataset.")

        fn3_available_activation_functions = {
            "Linear": "Identity function, g(z) = z. Used when no transformation is desired.",
            "ReLU": "Rectified Linear Unit, g(z) = max(0, z). Common in hidden layers; helps with gradient flow.",
            "LeakyReLU": "Leaky ReLU, g(z) = z if z > 0, else alpha * z. Addresses dying ReLU problem.",
            "PReLU": "Parametric ReLU, g(z) = z if z > 0, else alpha * z. Alpha is a learnable parameter.",
            "ELU": "Exponential Linear Unit, g(z) = alpha * (e^z - 1) if z < 0, else z. Aims to combine benefits of ReLU and sigmoid.",
            "GELU": "Gaussian Error Linear Unit, g(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)). Useful in Transformer models.",
            "SELU": "Scaled Exponential Linear Unit, g(z) = scale * [max(0, z) + min(0, alpha * (e^z - 1))]. Self-normalizing property.",
            "CELU": "Continuously Differentiable Exponential Linear Units, g(z) = max(0, z) + min(0, alpha * (exp(z/alpha) - 1)). Variant of ELU.",
            "CoLU": "Custom-made, g(z) = z / (1 - z^-(z + e^z)). Beneficial for deep networks.",
            "Softplus": "Softplus, g(z) = log(1 + e^z). Smooth approximation of ReLU.",
            "Swish": "Swish, g(z) = z * sigmoid(z). Balances ReLU and sigmoid properties.",
            "Hardswish": "Hard Swish, g(z) = z * ReLU6(z + 3) / 6. Efficient approximation of Swish.",
            "Sigmoid": "Sigmoid, g(z) = 1 / (1 + e^(-z)). Good for binary classification.",
            "Tanh": "g(z) = (e^z - e^(-z)) / (e^z + e^(-z)) Very similar to the sigmoid function, for classification.",
            "ArcTan": "Custom function, also for classification",
            "Softmax": "Softmax, g(z_i) = e^(z_i) / sum(e^(z_j) for j in all outputs. For multiclass classification problems.)",
            "Mish": "Both for regression and classification. Its smooth and non-monotonic."
        }

        print("\nAvailable activation functions:")
        for fn3_function, fn3_description in fn3_available_activation_functions.items():
            print(f"- {fn3_function}: {fn3_description}")

        fn3_available_optimizers = {
            "Adam": "Adaptive Moment Estimation, combines the advantages of AdaGrad and RMSprop. Effective for a wide range of problems, especially in deep learning. Start with this one.",
            "SGD": "Stochastic Gradient Descent, simple and effective, often used with momentum. Good for general purposes.",
            "RMSprop": "Root Mean Square Propagation, adapts learning rate based on recent gradients. Useful for recurrent neural networks.",
            "Adagrad": "Adaptive Gradient Algorithm, adjusts learning rate per parameter. Good for sparse data.",
            "Adadelta": "An extension of AdaGrad that reduces its aggressive learning rate. Suited for large datasets.",
            "Nesterov-SGD": "Stochastic Gradient Descent with Nesterov momentum. A momentum-based optimizer that has a lookahead property. Good for convex functions.",
            "LBFGS": "Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm, suitable for large-scale problems. Not commonly used for deep learning.",
            "AdamW": "Variant of Adam with improved weight decay regularization. Good for tasks where generalization is important.",
            "Adamax": "A variant of Adam based on the infinity norm. Can be more stable than Adam in some cases.",
        }

        print("\nAvailable Optimizers:")
        for fn3_optimizer, fn3_description in fn3_available_optimizers.items():
            print(f"- {fn3_optimizer}: {fn3_description}")

        fn3_available_cost_functions = {
            "MSELoss": "Mean Squared Error Loss, used for regression tasks. It measures the average squared difference between the estimated values and the actual value.",
            "CrossEntropyLoss": "Cross-Entropy Loss, used for classification tasks. It combines LogSoftmax and NLLLoss in one single class.",
            "BCELoss": "Binary Cross-Entropy Loss, used for binary classification tasks. Measures the loss for a binary classification problem with a probability output.",
            "BCEWithLogitsLoss": "Combines a Sigmoid layer and the BCELoss in one single class. More numerically stable than using a plain Sigmoid followed by a BCELoss.",
            "NLLLoss": "Negative Log Likelihood Loss, used in conjunction with log-softmax layer for classification tasks. It is useful to train a classification problem with C classes.",
            "SmoothL1Loss": "Smooth L1 Loss (also known as Huber Loss), used for regression tasks. It is less sensitive to outliers than the MSELoss and often used in reinforcement learning.",
            "L1Loss": "L1 Loss, measures the mean absolute error (MAE) between each element in the input and target. Used for regression tasks.",
            "PoissonNLLLoss": "Poisson Negative Log Likelihood Loss, used for count data and for modelling count-based distributions.",
            "KLDivLoss": "Kullback-Leibler divergence Loss, used when the output of the model is a probability distribution and measures how one probability distribution diverges from a second, expected probability distribution.",
            "MarginRankingLoss": "Creates a criterion that measures the loss given inputs x1, x2, two 1D mini-batch Tensors, and a label 1D mini-batch tensor y with values (1 or -1).",
            "HingeEmbeddingLoss": "Used for learning nonlinear embeddings or semi-supervised learning tasks. The loss is used for 'embedding' inputs into a lower-dimensional space.",
            "CosineEmbeddingLoss": "Measures the cosine distance between two tensors. Commonly used in tasks like learning semantic similarity between sentences."
        }

        print("\nAvailable cost-functions:")
        for fn3_cost_function, fn3_description in fn3_available_cost_functions.items():
            print(f"- {fn3_cost_function}: {fn3_description}")

        fn3_supported_file_extensions = {
            ".csv": "Comma separated value file.",
            ".json": "Add description.",
            ".xlsx": "Add description.",
            ".parquet": "Add description.",
            ".hdf": "Add description.",
            ".h5": "Add description."
        }

        print("\nSupported file extensions for the datafiles:")
        for fn3_file_extension, fn3_description in fn3_supported_file_extensions.items():
            print(f"- {fn3_file_extension}: {fn3_description}")

        sys.exit()
    elif len(fn3_arguments) == 1 and fn3_arguments[0] == 'config':
        create_config()
        sys.exit()

    fn3_parser = argparse.ArgumentParser(description='Process the development set size and file names.')
    fn3_parser.add_argument('train_set_name', type=str, help='Name of the train set file')
    fn3_parser.add_argument('dev_set_name', type=str, help='Name of the dev set file')

    # Optional arguments
    fn3_parser.add_argument('--plots', action='store_true', help='Optional argument to show plots')
    fn3_parser.add_argument('--optuna', nargs='?', const=50, type=int, default=False, help='Optional argument to run Optuna with a specified number')

    fn3_args = fn3_parser.parse_args(fn3_arguments)

    fn3_train_set_name = fn3_args.train_set_name
    fn3_dev_set_name = fn3_args.dev_set_name
    fn3_show_plots = fn3_args.plots
    fn3_run_optuna = fn3_args.optuna is not False
    fn3_trials = fn3_args.optuna if fn3_args.optuna is not False else None

    return fn3_train_set_name, fn3_dev_set_name, fn3_show_plots, fn3_run_optuna, fn3_trials


def create_timestamp():
    """
    Description:
        Generate a timestamp string with the current date and time.

    Input:
        None

    Output:
        str: A string representing the current date and time in the format "day-month-year_hour-minute-second".

    Function-code:
        fn4_
    """

    fn4_now = datetime.now()                                             # Get the current time
    fn4_precise_time = fn4_now.strftime("%d-%m-%Y_%H-%M-%S")             # Format the time as "day-month-year_hour-minute-second"
    return fn4_precise_time


def delete_pth_files():
    """
    Description:
        Deletes all .pth files in a specific subdirectory within the output directory.

    Input:
        None

    Output:
        None

    Function-code:
        fn5_
    """
    # Updated directory containing .pth files
    fn5_main_directory = '../output/nn_hyperpara_screener__output'
    fn5_sub_directory = 'model_pth_files'
    fn5_directory_path = os.path.join(fn5_main_directory, fn5_sub_directory)

    # Construct the full path to the directory
    fn5_full_directory_path = os.path.join(os.getcwd(), fn5_directory_path)

    # Check if the directory exists, and if not, return without doing anything
    if not os.path.exists(fn5_full_directory_path):
        return

    # List all .pth files in the directory
    fn5_pth_files = glob.glob(os.path.join(fn5_full_directory_path, '*.pth'))

    # Loop through the files and delete them
    for fn5_file_path in fn5_pth_files:
        try:
            os.remove(fn5_file_path)
        except OSError as e:
            print(f"Error deleting file {fn5_file_path}: {e}")


def read_and_process_config(fn6_nr_examples):
    """
    Description:
        Reads the configuration file for neural network hyperparameters, fills missing values with defaults,
        and enforces specific data types for each column.

    Input:
        fn6_nr_examples (int): The number of examples to be used as the default batch size in the configuration.

    Output:
        DataFrame: A pandas DataFrame containing the processed configuration settings for neural network training.

    Function-code:
        fn6_
    """

    fn6_default_values = {
        'neurons': '10,1',
        'acti_fun': 'ReLU',
        'acti_fun_out': 'Linear',
        'epochs': 100,
        'batch_size': fn6_nr_examples,
        'noise': 0,
        'optimizer': 'Adam,0.9,0.099,10e-8',
        'alpha': 0.01,
        'decay_rate': 0,
        'lamda': 0.001,
        'dropout': 0,
        'psi': 1,
        'cost_fun': 'MSELoss',
    }

    fn6_data_types = {
        'model_name': str,
        'neurons': str,
        'acti_fun': str,
        'acti_fun_out': str,
        'epochs': int,
        'batch_size': int,
        'noise': float,
        'optimizer': str,
        'alpha': float,
        'decay_rate': float,
        'lamda': float,
        'dropout': float,
        'psi': float,
        'cost_fun': str,
    }

    # Read the CSV file
    fn6_config = pd.read_csv('../input/nn_hyperpara_screener_config.csv')

    # Replace NaN values with defaults
    for fn6_column, fn6_default in fn6_default_values.items():
        if fn6_column in fn6_config.columns:
            fn6_config[fn6_column].fillna(fn6_default, inplace=True)

    # Enforce data types
    for fn6_column, fn6_dtype in fn6_data_types.items():
        if fn6_column in fn6_config.columns:
            fn6_config[fn6_column] = fn6_config[fn6_column].astype(fn6_dtype)

    return fn6_config


def check_config_columns(fn7_df, fn7_required_columns):
    """
    Description:
        Checks if the columns in a DataFrame match a set of required columns, taking into account exact matches and similar (potentially misspelled) column names.
        The function identifies missing and extra columns, and provides a detailed report of discrepancies. It exits the program if discrepancies are found.

    Input:
        fn7_df (DataFrame): The DataFrame to be checked.
        fn7_required_columns (dict): A dictionary with keys representing the required column names.

    Output:
        None: The function does not return a value but prints discrepancies and exits if any are found.

    Function-code:
        fn7_
    """

    def is_similar(fn8_column_a, fn8_column_b):
        """
        Function-code:
            fn8_
        """

        # Simple function to check if two column names are similar
        return fn8_column_a.lower() == fn8_column_b.lower() or fn8_column_a in fn8_column_b or fn8_column_b in fn8_column_a

    # Identify exact missing and extra columns
    fn7_exact_missing_columns = set(fn7_required_columns.keys()) - set(fn7_df.columns)
    fn7_exact_extra_columns = set(fn7_df.columns) - set(fn7_required_columns.keys())

    # Initialize lists for similar but potentially misspelled columns
    fn7_similar_missing_columns = []
    fn7_similar_extra_columns = []

    # Check for similar columns
    for fn7_column in list(fn7_exact_missing_columns):
        for fn7_df_column in fn7_df.columns:
            if is_similar(fn7_column, fn7_df_column):
                fn7_similar_missing_columns.append((fn7_df_column, fn7_column))  # Note the order change here
                fn7_exact_missing_columns.remove(fn7_column)
                fn7_exact_extra_columns.discard(fn7_df_column)
                break

    for fn7_column in list(fn7_exact_extra_columns):
        for fn7_required_column in fn7_required_columns.keys():
            if is_similar(fn7_column, fn7_required_column):
                fn7_similar_extra_columns.append((fn7_column, fn7_required_column))  # Keeping this as it is
                fn7_exact_extra_columns.remove(fn7_column)
                fn7_exact_missing_columns.discard(fn7_required_column)
                break

    # Print the results
    if fn7_exact_missing_columns or fn7_exact_extra_columns or fn7_similar_missing_columns or fn7_similar_extra_columns:
        if fn7_exact_missing_columns:
            print("Missing columns:", ', '.join(fn7_exact_missing_columns))
        if fn7_exact_extra_columns:
            print("Extra columns:", ', '.join(fn7_exact_extra_columns))
        if fn7_similar_missing_columns:
            print("Mispelled or similar missing columns:", ', '.join([f"{x[0]} (similar to {x[1]})" for x in fn7_similar_missing_columns]))
        if fn7_similar_extra_columns:
            print("Mispelled or similar extra columns:", ', '.join([f"{x[0]} (similar to {x[1]})" for x in fn7_similar_extra_columns]))

        print("Error: DataFrame does not have the exact required columns.")
        sys.exit(1)


def check_dataframe_columns(fn9_df):
    """
    Description:
        Checks if the given DataFrame contains a specific set of columns
        and no others. It also checks if the data in specific columns meets
        certain criteria:
        - 'nr_neurons': Must contain only ints separated by commas.
        - 'alpha': Must contain only legal ints or floats.
        - 'nr_epochs' and 'batch_size': Must contain only ints.
        - 'lamda': Must contain only legal floats.
        - 'acti_fun': Must contain only letters.
        - 'psi': Must contain only legal ints or floats.

        If any condition is not met, the function prints an error message
        along with the contents of the offending cell and terminates the script.

    Input:
        fn9_df (DataFrame): The DataFrame to be checked.

    Output:
        Outputs an error message and terminates the script if discrepancies are found, or confirms successful validation otherwise.

    Funtion-code:
        fn9_
    """

    fn9_required_columns_nn = {
        'model_name': lambda fn9_column: all(isinstance(fn9_element, str) for fn9_element in fn9_column) and len(set(fn9_column)) == len(fn9_column),
        'neurons': lambda fn9_x: pd.isna(fn9_x) or all(fn9_i.strip().isdigit() for fn9_i in str(fn9_x).split(',')),
        'acti_fun': lambda fn9_x: pd.isna(fn9_x) or fn9_x.isalpha(),
        'acti_fun_out': lambda fn9_x: pd.isna(fn9_x) or fn9_x.isalpha(),
        'epochs': lambda fn9_x: pd.isna(fn9_x) or (isinstance(fn9_x, int) and str(fn9_x).isdigit()),
        'batch_size': lambda fn9_x: pd.isna(fn9_x) or (isinstance(fn9_x, int) and str(fn9_x).isdigit()),
        'noise': lambda fn9_x: pd.isna(fn9_x) or (isinstance(fn9_x, (int, float)) and fn9_x >= 0),
        'optimizer': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, str),
        'alpha': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, (int, float)),
        'decay_rate': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, (int, float)),
        'lamda': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, float),
        'dropout': lambda fn9_x: pd.isna(fn9_x) or (isinstance(fn9_x, (int, float)) and 0 <= fn9_x < 1),
        'psi': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, (int, float)),
        'cost_fun': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, str),
        'seed': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, (int, float)),
    }

    check_config_columns(fn9_df, fn9_required_columns_nn)

    # Checks if the column model_name just contains unique values (no two rows should have the same name)
    fn9_model_name_column = fn9_df.columns[0]

    # Find indices of rows with duplicate values in the first column
    fn9_duplicate_rows = fn9_df[fn9_df[fn9_model_name_column].duplicated(keep=False)].index

    # Print the result
    if not fn9_duplicate_rows.empty:
        print(f"Rows with duplicate values in the column model_name: {list(fn9_duplicate_rows)}")
        sys.exit()

    # Check each column for its specific criteria
    for fn9_col, fn9_check in fn9_required_columns_nn.items():
        if fn9_check:
            for fn9_index, fn9_value in fn9_df[fn9_col].items():
                if not fn9_check(fn9_value):
                    print(f"Error: Data in column '{fn9_col}', row {fn9_index}, does not meet the specified criteria. Offending data: {fn9_value}")
                    sys.exit(1)

    print("All required columns are present and data meets all specified criteria.")


def load_datasets(fn10_train_set_name, fn10_dev_set_name):
    """
    Description:
        Load datasets based on file extension.

    Input:
        fn10_train_set_name (str): File path for the training dataset.
        fn10_dev_set_name (str): File path for the development dataset.

    Output:
        tuple: A tuple containing two pandas DataFrames, one for the training set and one for the development set.

    Function-code:
        fn10_
    """

    def load_dataset(fn11_file_name):
        """
        Description:
            Loads a dataset from a specified file path, supporting multiple file formats. The function determines the file extension and uses the appropriate pandas function to read the data.

        Input:
            fn11_file_name (str): File path for the dataset, supporting formats like CSV, JSON, XLSX, Parquet, HDF5.

        Output:
            DataFrame: A pandas DataFrame containing the data from the specified file.

        Function-code:
            fn11_
        """

        # Full path construction
        fn11_full_path = os.path.join("../input", fn11_file_name)

        # Determine the file extension
        fn11_file_ext = os.path.splitext(fn11_full_path)[1].lower()

        # Load the dataset based on the file extension
        if fn11_file_ext == '.csv':
            return pd.read_csv(fn11_full_path)
        elif fn11_file_ext == '.json':
            return pd.read_json(fn11_full_path)
        elif fn11_file_ext == '.xlsx':
            return pd.read_excel(fn11_full_path)
        elif fn11_file_ext == '.parquet':
            return pd.read_parquet(fn11_full_path)
        elif fn11_file_ext == '.hdf' or fn11_file_ext == '.h5':
            return pd.read_hdf(fn11_full_path)
        else:
            raise ValueError(f"Unsupported file type: {fn11_file_ext}")

    # Load the datasets
    fn10_train_set_data = load_dataset(fn10_train_set_name)
    fn10_dev_set_data = load_dataset(fn10_dev_set_name)

    return fn10_train_set_data, fn10_dev_set_data


def assign_hyperparameters_from_config(fn12_pandas_df, fn12_row_nr, fn12_amount_of_rows):
    """
    Description:
        Extracts and processes hyperparameters for a neural network model from a given DataFrame row.
        The function maps DataFrame columns to hyperparameter variable names, handles special data parsing
        for certain parameters (like neuron counts and optimizer settings), and validates some hyperparameters
        against provided constraints (such as batch size and dropout values).

    Input:
        fn12_pandas_df (DataFrame): The DataFrame containing hyperparameter configurations.
        fn12_row_nr (int): The row number in the DataFrame to extract hyperparameters from.
        fn12_amount_of_rows (int): The total number of rows in the dataset, used for batch size validation.

    Output:
        dict: A dictionary of processed and validated hyperparameters for use in a neural network model.

    Function-code:
        fn12_
    """

    fn12_variable_names = ['model_name', 'nr_neurons_str', 'activation_function_type', 'acti_fun_out', 'nr_epochs',
                           'batch_size', 'noise_stddev', 'optimizer_type', 'learning_rate', 'decay_rate', 'lamda', 'dropout',
                           'psi_value', 'cost_function','random_seed']
    fn12_column_names = ['model_name', 'neurons', 'acti_fun', 'acti_fun_out', 'epochs', 'batch_size',
                         'noise', 'optimizer', 'alpha', 'decay_rate', 'lamda', 'dropout', 'psi', 'cost_fun','seed']

    fn12_hyperparams = {}

    for fn12_var_name, fn12_col_name in zip(fn12_variable_names, fn12_column_names):
        fn12_value = fn12_pandas_df[fn12_col_name].iloc[fn12_row_nr]

        if fn12_var_name == 'nr_neurons_str':  # Special treatment for nr_neurons
            fn12_neuron_list = [int(fn12_neuron) for fn12_neuron in fn12_value.split(',')]
            fn12_nr_output_neurons = fn12_neuron_list.pop()
            fn12_hyperparams['nr_neurons_hidden_layers'] = fn12_neuron_list
            fn12_hyperparams['nr_neurons_output_layer'] = fn12_nr_output_neurons
        elif fn12_var_name == 'optimizer_type':
            fn12_split_values = fn12_value.split(',')
            fn12_optimizer = fn12_split_values[0]
            fn12_optim_add_params = [float(fn12_item) for fn12_item in fn12_split_values[1:]]
            fn12_hyperparams['optimizer_type'] = fn12_optimizer
            fn12_hyperparams['optim_add_params'] = fn12_optim_add_params
        else:
            fn12_hyperparams[fn12_var_name] = fn12_value

    if fn12_hyperparams['batch_size'] > fn12_amount_of_rows:
        fn12_hyperparams['batch_size'] = fn12_amount_of_rows
        fn12_pandas_df.loc[fn12_row_nr, "batch_size"] = fn12_amount_of_rows

    if fn12_hyperparams['dropout'] >= 1 or fn12_hyperparams['dropout'] < 0:
        print("Error: dropout must be smaller than 1 and greater than 0!")
        sys.exit()

    return fn12_hyperparams


class NNmodel(nn.Module):
    """
    Description:
        The NNmodel class defines a customizable neural network architecture using PyTorch's nn.Module.
        It allows for creating a sequential model with varied numbers of layers, customizable neuron counts in each layer,
        choice of activation functions, and the option to include dropout for regularization.
        The class supports batch normalization and is adaptable for different types of neural network tasks,
        such as regression or classification.
    """

    def __init__(self, fn13_input_size, fn13_nr_neurons, fn13_activation_function, fn13_nr_output_neurons, fn13_acti_fun_out, fn13_dropout):
        """
        Description:
            Initializes the NNmodel class by constructing a neural network with specified layers, batch normalization, activation functions, and dropout.
            The network is built as a sequence of layers, starting with a linear transformation, followed by batch normalization, activation, and dropout,
            and ending with an output layer.

        Input:
            fn13_input_size (int): The size of the input layer.
            fn13_nr_neurons (list): A list of the number of neurons in each hidden layer.
            fn13_activation_function (function): The activation function for the hidden layers.
            fn13_nr_output_neurons (int): The number of neurons in the output layer.
            fn13_acti_fun_out (function): The activation function for the output layer.
            fn13_dropout (float): Dropout rate for regularization.

        Output:
            None: This method does not return a value but initializes the layers of the neural network.

        Function-code:
            fn13_
        """

        super(NNmodel, self).__init__()
        layers = []
        for fn13_neurons in fn13_nr_neurons:
            layers.append(nn.Linear(fn13_input_size, fn13_neurons))         # The linear function is the "z = weight_vector * feature_vector + bias" part
            layers.append(nn.BatchNorm1d(fn13_neurons))                     # Batch normalization on the z (before the activation function is applied).
            layers.append(fn13_activation_function)                         # This is then the "g(z)" part.
            layers.append(nn.Dropout(fn13_dropout))                         # Dropout layer, this adds dropout regularization to the network, if fn13_dropout is bigger than 0.
            fn13_input_size = fn13_neurons

        layers.append(nn.Linear(fn13_input_size, fn13_nr_output_neurons))   # Output layer(s). For regression, there is usually only one output layer with no activation function.
        layers.append(fn13_acti_fun_out)                                    # If fn13_acti_fun_out is 'linear', it will not do any harm, then just nothing happens.
        self.model = nn.Sequential(*layers)

    def forward(self, fn14_x):
        """
        Description:
            Passes the input tensor through the sequential neural network model and returns the output tensor,
            representing the model's predictions or outputs.

        Input:
            fn14_x (Tensor): The input tensor to be fed through the neural network.

        Output:
            Tensor: The output tensor obtained after passing the input through the neural network.

        Function-code:
            fn14_
        """

        return self.model(fn14_x)  # Passes the input x through the model and returns the result (y-predictions)


class ArcTanActivation(torch.nn.Module):
    """
    Description:
        A custom PyTorch activation function module implementing the ArcTan (arctangent) function.
        This module can be used as an activation function in neural network models within PyTorch's framework.
    """

    def forward(self, fn15_input):
        """
        Description:
            Computes the arctangent of the input tensor element-wise, applying the ArcTan activation function.

        Input:
            fn15_input (Tensor): The input tensor to which the ArcTan activation function will be applied.

        Output:
            Tensor: The output tensor after applying the ArcTan activation function element-wise.

        Function-code:
            fn15_
        """

        return torch.atan(fn15_input)


class Mish(torch.nn.Module):
    """
    Description:
        A custom PyTorch activation function module implementing the Mish function.
        Mish is a smooth, self-regularized, and non-monotonic activation function which
        has been found to improve neural network performance in certain scenarios.
    """

    def forward(self, fn16_input):
        """
        Description:
            Applies the Mish activation function to the input tensor. Mish is defined as
            x * tanh(softplus(x)), which helps in maintaining a smooth gradient flow.

        Input:
            fn16_input (Tensor): The input tensor to which the Mish activation function will be applied.

        Output:
            Tensor: The output tensor after applying the Mish activation function element-wise.

        Function-code:
            fn16_
        """
        return fn16_input * torch.tanh(F.softplus(fn16_input))


class Hardswish(torch.nn.Module):
    """
    Description:
        A custom PyTorch activation function module implementing the Hardswish function.
        Hardswish is an efficient, hardware-friendly approximation of the Swish activation function,
        commonly used in deep learning models for faster computation with similar performance.
    """

    def forward(self, fn17_input):
        """
        Description:
            Applies the Hardswish activation function to the input tensor. Hardswish is computed as
            x * relu6(x + 3) / 6, providing a piecewise linear approximation to Swish, which is faster to compute.

        Input:
            fn17_input (Tensor): The input tensor to which the Hardswish activation function will be applied.

        Output:
            Tensor: The output tensor after applying the Hardswish activation function element-wise.

        Function-code:
            fn17_
        """
        return fn17_input * F.relu6(fn17_input + 3) / 6


class CoLUActivation(nn.Module):
    """
    Description:
        A custom PyTorch activation function module implementing the CoLU (Complementary Log-U) function.
        This activation function is designed to provide a novel approach to handling neural network activations,
        potentially offering benefits in specific types of neural network architectures.
    """

    def forward(self, fn18_input):
        """
        Description:
            Applies the CoLU activation function to the input tensor. The CoLU function is defined as
            x / (1 - x^(-x + exp(x))), which introduces a unique transformation to the input values.

        Input:
            fn18_input (Tensor): The input tensor to which the CoLU activation function will be applied.

        Output:
            Tensor: The output tensor after applying the CoLU activation function element-wise.

        Function-code:
            fn18_
        """
        return fn18_input / (1 - torch.pow(fn18_input, -(fn18_input + torch.exp(fn18_input))))


def get_activation_function(fn19_acti_fun_type):
    """
    Description:
        Returns the corresponding PyTorch activation function based on the given type.

    Input:
        local_acti_fun_type (str): The name of the activation function.

    Output:
        nn.Module: The PyTorch activation function object.
        If the specified activation function is not recognized, the function
        prints an error message and exits the script.

    Function-code:
        fn19_
    """

    if fn19_acti_fun_type == "Linear":                 # Basically means no activation function
        return nn.Identity()                            # Linear, g(z) = z
    elif fn19_acti_fun_type == "ReLU":                 # All the ReLU functions should only be used in the hidden layers.
        return nn.ReLU()                                # ReLU, g(z) = max(0, z)
    elif fn19_acti_fun_type == "LeakyReLU":            # In case of dead neurons
        return nn.LeakyReLU()                           # Leaky ReLU, g(z) = z if z > 0, else alpha * z, alpha is fixed
    elif fn19_acti_fun_type == "PReLU":
        return nn.PReLU()                               # PReLU, g(z) = z if z > 0, else alpha * z, alpha is a hyperparameter
    elif fn19_acti_fun_type == "ELU":                  # ELU = exponential linear units
        return nn.ELU()                                 # ELU, g(z) = alpha * (e^z - 1) if z < 0, else z, alpha is fixed
    elif fn19_acti_fun_type == "GELU":                 # Gaussian error linear units.
        return nn.GELU()                                # GELU, g(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3))
    elif fn19_acti_fun_type == "SELU":
        return nn.SELU()                                # SELU, g(z) = scale * [max(0, z) + min(0, alpha * (e^z - 1))]
    elif fn19_acti_fun_type == "CELU":
        return nn.CELU()                                # CeLU, g(z) = max(0, z) + min(0, alpha * (exp(z/alpha) - 1))
    elif fn19_acti_fun_type == "CoLU":                 # Beneficial for deep networks.
        return CoLUActivation()                         # CoLU, g(z) = z / (1 - z^-(z + e^z))
    elif fn19_acti_fun_type == "Softplus":             # Smooth approximation of ReLU
        return nn.Softplus()                            # Softplus, g(z) = log(1 + e^z)
    elif fn19_acti_fun_type == "Swish":
        return nn.SiLU()                                # Swish, g(z) = z * sigmoid(z)
    elif fn19_acti_fun_type == "Hardswish":            # Approximation of the Swish activation function that is computationally more efficient.
        return Hardswish()                              # Hard Swish, g(z) = z * ReLU6(z + 3) / 6
    elif fn19_acti_fun_type == "Sigmoid":              # Sigmoid and Tanh are better for classification.
        return nn.Sigmoid()                             # Sigmoid, g(z) = 1 / (1 + e^(-z))
    elif fn19_acti_fun_type == "Tanh":                 # Very similar to the sigmoid function, for classification.
        return nn.Tanh()                                # Tanh, g(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    elif fn19_acti_fun_type == "ArcTan":               # Custom function build with class.
        return ArcTanActivation()                       # ArcTan, g(z) = arctan(z)
    elif fn19_acti_fun_type == "Softmax":              # For multiclass classification problems.
        return nn.Softmax(dim=1)                        # Softmax, g(z_i) = e^(z_i) / sum(e^(z_j) for j in all outputs)
    elif fn19_acti_fun_type == "Mish":                 # Both for regression and classification. Its smooth and non-monotonic.
        return Mish()                                   # Mish, g(z) = z * tanh(softplus(z)) = z * tanh(log(1 + e^z))
    else:
        print("Error: No activation function was specified in the config file!")
        sys.exit()


def create_initial_weights(fn20_input_size, fn20_n_neurons, fn20_nr_output_neurons, fn20_acti_fun_type, fn20_psi):
    """
    Description:
        Creates initial weights and biases for a neural network given the input size,
        the number of neurons in each hidden layer, and the output size. Weights are
        initialized using a custom formula, and biases are initialized to a small constant value.

    Input:
        fn20_input_size (int): The number of features in the input data.
        fn20_n_neurons (list of int): A list containing the number of neurons in each hidden layer.
        fn20_output_size (int, optional): The number of neurons in the output layer.

    Output:
        weights (list of Tensor): A list of weight matrices, where each matrix corresponds
                                  to the weights for one layer in the network.
        biases (list of Tensor): A list of bias vectors, where each vector corresponds
                                 to the biases for one layer in the network.

    Function-code:
        fn20_
    """

    fn20_weights = []
    fn20_biases = []

    # Start with the input size
    fn20_layer_input_size = fn20_input_size

    # Create weights and biases for each hidden layer
    for fn20_neurons in fn20_n_neurons:
        # Custom weight initialization
        if fn20_acti_fun_type == "ReLU":
            fn20_w = torch.randn(fn20_layer_input_size, fn20_neurons) * np.sqrt(fn20_psi * 2 / fn20_layer_input_size)
        elif fn20_acti_fun_type == "Tanh":
            fn20_w = torch.randn(fn20_layer_input_size, fn20_neurons) * np.sqrt(fn20_psi * 1 / fn20_layer_input_size)      # Xavier initialization
        else:
            fn20_w = torch.randn(fn20_layer_input_size, fn20_neurons) * np.sqrt(fn20_psi * 1 / fn20_layer_input_size)

        fn20_weights.append(fn20_w)

        # Bias initialization
        fn20_b = torch.Tensor(fn20_neurons).fill_(0.01)
        fn20_biases.append(fn20_b)

        # The output of this layer is the input to the next layer
        fn20_layer_input_size = fn20_neurons

    # Custom weight initialization for the output layer
    fn20_w = torch.randn(fn20_n_neurons[-1], fn20_nr_output_neurons) * np.sqrt(1/fn20_n_neurons[-1])
    fn20_weights.append(fn20_w)

    fn20_b = torch.Tensor(fn20_nr_output_neurons).fill_(0.01)
    fn20_biases.append(fn20_b)

    return fn20_weights, fn20_biases


def apply_stored_weights(fn21_model, fn21_weights, fn21_biases):
    """
    Description:
        Applies stored weights and biases to each linear layer in a given model.
        This function iterates through each layer of the provided model. If the layer is an instance of nn.Linear
        (a linear layer), the function sets the layer's weights and biases to the stored values provided. This is
        particularly useful for initializing a model with a specific set of weights and biases, ensuring consistency
        in model initialization across different instances or runs.

        Note: This function assumes that the number of tensors in fn21_weights and fn21_biases matches the number of linear
        layers in fn21_model. It applies each set of weights and biases to the corresponding linear layer based on their
        order in the model.

    Input:
        fn21_model (nn.Module): The neural network model to which the weights and biases will be applied.
                                 The model should be an instance of nn.Module or a subclass thereof.
        fn21_weights (List[Tensor]): A list of PyTorch tensors representing the weights for each linear layer in the model.
                                      The order of tensors in the list should match the order of linear layers in the model.
        fn21_biases (List[Tensor]): A list of PyTorch tensors representing the biases for each linear layer in the model.
                                     Similar to local_weights, the order should match the linear layers' order.

    Output:
        The function does not return any value; it modifies the model in-place.

    Function-code:
        fn21_
    """

    fn21_linear_layer_count = 0
    for fn21_layer in fn21_model.model:
        if isinstance(fn21_layer, nn.Linear):
            # Transpose the weight matrix to match [out_features, in_features] (this is to backtranspose the matrix, it is somehow in the wrong format).
            fn21_transposed_weight = fn21_weights[fn21_linear_layer_count].T
            fn21_layer.weight.data = fn21_transposed_weight
            fn21_layer.bias.data = fn21_biases[fn21_linear_layer_count]
            fn21_linear_layer_count += 1


def get_optimizer(fn22_model_parameters, fn22_optimizer_type, fn22_learning_rate, fn22_lamda, fn22_additional_params):
    """
    Description:
        Creates and returns a PyTorch optimizer object based on the specified type and arguments.

    Input:
        fn22_model_parameters: The parameters of the model to be optimized.
        fn22_optimizer_type (str): Type of the optimizer (e.g., 'Adam', 'RMSprop', 'SGD', etc.).
        fn22_learning_rate (float): Strength of decay.
        fn22_additional_params (list): Additional parameters specific to the optimizer type.

    Output:
        An instance of the specified optimizer initialized with the given parameters and arguments.

    Function-code:
        fn22_
    """

    fn22_supported_optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "Adadelta": optim.Adadelta,
        "Nesterov-SGD": lambda fn23_params, fn23_lr, **fn23_kwargs: optim.SGD(fn23_params, fn23_lr, **fn23_kwargs, nesterov=True),
        "LBFGS": optim.LBFGS,
        "AdamW": optim.AdamW,
        "Adamax": optim.Adamax
    }

    fn22_optimizer_class = fn22_supported_optimizers.get(fn22_optimizer_type)
    if not fn22_optimizer_class:
        raise ValueError(f"Unsupported optimizer type: {fn22_optimizer_type}")

    # Build the keyword arguments for the optimizer (lr and weight_decay are the consensus keywords for all those optimizers).
    # weight_decay = L2 regularization. No other form of regularization is directly supported for the PyTorch optimizers.
    fn22_kwargs = {
        'lr': fn22_learning_rate,
        'weight_decay': fn22_lamda,
    }

    # **dict(zip(['betas', 'momentum', 'dampening', 'alpha', 'tolerance_grad', 'tolerance_change', 'max_iter', 'history_size', 'eps'], local_additional_params))

    if fn22_optimizer_type == 'Adam':
        # Assuming the first two elements in local_additional_params are beta1 and beta2
        fn22_betas = (fn22_additional_params[0], fn22_additional_params[1])
        fn22_kwargs['betas'] = fn22_betas           # betas is a keyword for the Adam optimizer

        fn22_epsilon = fn22_additional_params[2]
        fn22_kwargs['eps'] = fn22_epsilon           # eps is the keyword for the Adam optimizer

    fn22_optimizer = fn22_optimizer_class(fn22_model_parameters, **fn22_kwargs)
    return fn22_optimizer


def get_criterion(fn24_cost_function):
    """
    Description:
        Dynamically retrieves and returns a loss function from the PyTorch nn module based on a given string.

    Input:
        fn24_cost_function (str): Name of the loss function as a string. This should correspond to a class name in torch.nn.

    Output:
        fn24_criterion: An instance of the specified loss function class from torch.nn.

        Raises:
            SystemExit: If the specified loss function is not found in torch.nn or if no loss function is provided.

    Function-code:
        fn24_
    """

    if isinstance(fn24_cost_function, str):
        fn24_loss_class = getattr(nn, fn24_cost_function, None)  # Dynamically get the loss class from the nn module

        if fn24_loss_class is not None:                           # Check if the loss class was successfully retrieved
            fn24_criterion = fn24_loss_class()                   # Instantiate the loss class
            return fn24_criterion
        else:
            print(f"Error: '{fn24_cost_function}' is not a valid loss function in torch.nn.")
            sys.exit(1)
    else:
        print("Error: No cost function was specified in the config file!")
        sys.exit(1)


def prepare_model_training(fn25_hyperparams, fn25_train_dev_dataframes, fn25_input_size):
    """
    Description:
        Prepares a neural network for training by setting up the model, optimizer, criterion, and data loaders.

    Input:
        - fn25_activation_function_type: Type of the activation function.
        - fn25_acti_fun_out: Output activation function type.
        - fn25_input_size: Input size of the neural network.
        - fn25_nr_neurons: Number of neurons.
        - fn25_nr_output_neurons: Number of output neurons.
        - fn25_psi_value: Psi value for initializing weights.
        - fn25_optimizer_type: Type of optimizer to use.
        - fn25_learning_rate: Learning rate for the optimizer.
        - fn25_lamda: Lambda value for the optimizer.
        - fn25_optim_add_params: Additional parameters for the optimizer.
        - fn25_cost_function: Cost function for the model.
        - fn25_x_train: Training data.
        - fn25_y_train: Training labels.
        - fn25_x_dev: Development data.
        - fn25_y_dev: Development labels.
        - fn25_batch_size: Batch size for data loading.

    Output:
        fn25_model, fn25_optimizer, fn25_criterion, fn25_train_loader, fn25_dev_loader, fn25_x_dev_tensor, fn25_y_dev_tensor

    Function-code:
        fn25_
    """

    fn25_nr_neurons = fn25_hyperparams['nr_neurons_hidden_layers']
    fn25_nr_output_neurons = fn25_hyperparams['nr_neurons_output_layer']
    fn25_activation_function_type = fn25_hyperparams['activation_function_type']
    fn25_acti_fun_out = fn25_hyperparams['acti_fun_out']
    fn25_batch_size = fn25_hyperparams['batch_size']
    fn25_optimizer_type = fn25_hyperparams['optimizer_type']
    fn25_optim_add_params = fn25_hyperparams['optim_add_params']
    fn25_learning_rate = fn25_hyperparams['learning_rate']
    fn25_lamda = fn25_hyperparams['lamda']
    fn25_dropout = fn25_hyperparams['dropout']
    fn25_psi_value = fn25_hyperparams['psi_value']
    fn25_cost_function = fn25_hyperparams['cost_function']

    fn25_random_seed = fn25_hyperparams['random_seed']
    torch.manual_seed(fn25_random_seed)
    np.random.seed(fn25_random_seed)

    fn25_x_train = fn25_train_dev_dataframes[0]
    fn25_y_train = fn25_train_dev_dataframes[1]
    fn25_x_dev = fn25_train_dev_dataframes[2]
    fn25_y_dev = fn25_train_dev_dataframes[3]

    # Get activation functions
    fn25_activation_function = get_activation_function(fn25_activation_function_type)
    fn25_acti_fun_output = get_activation_function(fn25_acti_fun_out)

    # Initialize weights and biases
    fn25_init_weights, fn25_init_biases = create_initial_weights(fn25_input_size, fn25_nr_neurons, fn25_nr_output_neurons,
                                                                 fn25_activation_function_type, fn25_psi_value)

    # Create the model
    fn25_model = NNmodel(fn25_input_size, fn25_nr_neurons, fn25_activation_function, fn25_nr_output_neurons,
                         fn25_acti_fun_output, fn25_dropout)
    apply_stored_weights(fn25_model, fn25_init_weights, fn25_init_biases)

    fn25_model_parameters = fn25_model.parameters()
    fn25_optimizer = get_optimizer(fn25_model_parameters, fn25_optimizer_type, fn25_learning_rate, fn25_lamda, fn25_optim_add_params)

    fn25_criterion = get_criterion(fn25_cost_function)

    # Convert arrays to tensors
    fn25_x_train_tensor = torch.tensor(fn25_x_train, dtype=torch.float32)  # float32 is common for nn, because they provide a good balance between precision and computational efficiency.
    fn25_y_train_tensor = torch.tensor(fn25_y_train, dtype=torch.float32)
    fn25_x_dev_tensor = torch.tensor(fn25_x_dev, dtype=torch.float32)
    fn25_y_dev_tensor = torch.tensor(fn25_y_dev, dtype=torch.float32)

    fn25_batch_size = int(fn25_batch_size)

    # Define a DataLoader
    # Both the train data (x) and the labels (y) are combined into a 'TensorDataset', which makes it easier to iterate over the data during training.
    fn25_train_dataset = TensorDataset(fn25_x_train_tensor, fn25_y_train_tensor)
    # Without shuffling, the rows are always in the same order and the batches are always identical (which is suboptimal for learning).
    fn25_train_loader = DataLoader(fn25_train_dataset, batch_size=fn25_batch_size, shuffle=True, drop_last=True)  # Drop the last batch if there are not enough examples for the batch size left
    fn25_dev_dataset = TensorDataset(fn25_x_dev_tensor, fn25_y_dev_tensor)
    fn25_dev_loader = DataLoader(fn25_dev_dataset, batch_size=fn25_batch_size, shuffle=False, drop_last=True)  # Drop the last batch if there are not enough examples for the batch size left

    return fn25_model, fn25_optimizer, fn25_criterion, fn25_train_loader, fn25_dev_loader, fn25_x_dev_tensor, fn25_y_dev_tensor


def train_and_optionally_plot(fn26_model_to_train, fn26_training_loader, fn26_epochs_num, fn26_training_optimizer, fn26_loss_criterion, fn26_x_dev_data, fn26_y_dev_data, fn26_noise_stddev, fn26_learning_rate, fn26_decay_rate, fn26_inside_optuna=True, fn26_name_of_model='MyModel', fn26_timestamp='now', fn26_show_plots=False, fn26_plot_every_epochs=10):
    """
    Description:
        Trains the model and plots the training and development loss.

    Input:
        model_to_train: The neural network model to be trained.
        training_loader: DataLoader for the training data.
        epochs_num: Number of epochs for training.
        training_optimizer: Optimizer used for training.
        loss_criterion: Loss function.
        x_dev_data: Tensor for development/validation features.
        y_dev_data: Tensor for development/validation labels.
        name_of_model: String that represents the name of the model (decided by the user).
        plot_every_epochs: Frequency of updating the plot (default: 10 epochs).

    Output:

    Function-code:
        fn26_
    """

    fn26_training_history = None
    fn26_ax = None
    fn26_record_low_train_loss = None
    fn26_record_low_dev_loss = None
    fn26_start_time = None
    fn26_fig = None

    if not fn26_inside_optuna:
        # Initialize the plot
        fn26_fig, fn26_ax = plt.subplots(figsize=(10, 4))
        if fn26_show_plots:
            plt.show(block=False)  # Open the plot window

        # Add horizontal grid lines
        fn26_ax.yaxis.grid(True)  # Add horizontal grid lines
        fn26_ax.set_axisbelow(True)  # Ensure grid lines are below other plot elements

        # Training loop
        fn26_training_history = {'loss_train': [], 'loss_dev': []}  # Keeps track of the losses over each epoch.

        # Initialize record low variables
        fn26_record_low_train_loss = float('inf')
        fn26_record_low_dev_loss = float('inf')

        fn26_start_time = datetime.now()

    fn26_loss_train = None

    for fn26_epoch in range(fn26_epochs_num):  # Goes from 0 to epochs_num - 1.
        # Calculate the adjusted learning rate
        fn26_adjusted_lr = fn26_learning_rate / (1 + fn26_decay_rate * fn26_epoch)

        # Update the learning rate in the optimizer
        for fn26_param_group in fn26_training_optimizer.param_groups:
            fn26_param_group['lr'] = fn26_adjusted_lr

        for fn26_batch_x, fn26_batch_y in fn26_training_loader:  # The training_loader always delivers a new batch.
            fn26_training_optimizer.zero_grad()  # Resets the gradients of the model parameters.

            # Apply Gaussian noise to the batch (if the hyperparameter fn26_noise_stddev is set to 0, there is no Gaussian noise added).
            fn26_noise = torch.randn_like(fn26_batch_x) * fn26_noise_stddev  # noise_stddev is the standard deviation of the noise
            fn26_noisy_batch_x = fn26_batch_x + fn26_noise

            fn26_predictions = fn26_model_to_train(fn26_noisy_batch_x).squeeze()  # Forward pass
            fn26_loss_train = fn26_loss_criterion(fn26_predictions, fn26_batch_y)  # Compute the loss
            fn26_loss_train.backward()  # Backpropagation
            fn26_training_optimizer.step()  # Update weights

        with torch.no_grad():
            fn26_model_to_train.eval()
            fn26_predictions = fn26_model_to_train(fn26_x_dev_data).squeeze()  # Forward pass on dev set
            fn26_loss_dev = fn26_loss_criterion(fn26_predictions, fn26_y_dev_data)

        if not fn26_inside_optuna:  # When run while using Optuna, do not plot anything.
            # Record training and validation loss
            fn26_training_history['loss_train'].append(fn26_loss_train.item())
            fn26_training_history['loss_dev'].append(fn26_loss_dev.item())

            if (fn26_epoch + 1) % fn26_plot_every_epochs == 0 or fn26_epoch == fn26_epochs_num - 1:
                fn26_ax.clear()
                fn26_ax.plot(fn26_training_history['loss_train'], label='Train set loss')
                fn26_ax.plot(fn26_training_history['loss_dev'], label='Dev set loss')
                fn26_ax.set_title(f"{fn26_name_of_model} - Epoch: {fn26_epoch + 1}", fontsize=16, fontweight='bold')
                fn26_ax.set_xlabel('Epochs', fontsize=14)
                fn26_ax.set_ylabel('Loss', fontsize=14)
                fn26_ax.set_xlim(0, fn26_epochs_num)

                fn26_ax.yaxis.grid(True)
                fn26_ax.set_axisbelow(True)
                fn26_ax.legend()

                # Update record lows if current losses are lower
                fn26_current_train_loss = fn26_training_history['loss_train'][-1]
                fn26_current_dev_loss = fn26_training_history['loss_dev'][-1]

                if fn26_current_train_loss < fn26_record_low_train_loss:
                    fn26_record_low_train_loss = fn26_current_train_loss

                if fn26_current_dev_loss < fn26_record_low_dev_loss:
                    fn26_record_low_dev_loss = fn26_current_dev_loss

                fn26_ax.text(0.35, 0.93, f'Train-loss = {round(fn26_current_train_loss, 2)} (Lowest: {round(fn26_record_low_train_loss, 2)})\n'
                             f'Dev-loss  = {round(fn26_current_dev_loss, 2)} (Lowest: {round(fn26_record_low_dev_loss, 2)})',
                             transform=fn26_ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

                # Add the current runtime info.
                fn26_end_time = datetime.now()
                fn26_runtime = fn26_end_time - fn26_start_time
                fn26_hours, fn26_remainder = divmod(fn26_runtime.total_seconds(), 3600)
                fn26_minutes, fn26_seconds = divmod(fn26_remainder, 60)
                fn26_runtime_text = f"Runtime: {int(fn26_hours)}h {int(fn26_minutes)}m {int(fn26_seconds)}s"

                # Plot the runtime text
                fn26_ax.text(0.35, 0.73, fn26_runtime_text, transform=fn26_ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

                if fn26_show_plots:  # Only show the plots when the user decides so.
                    plt.draw()
                    plt.pause(0.1)

                if fn26_epoch == fn26_epochs_num - 1:

                    # Main directory and subdirectory for saving model .pth files
                    fn26_main_directory = '../output/nn_hyperpara_screener__output'
                    fn26_sub_directory = 'model_pth_files'

                    # Full path for saving the model includes both the main directory and subdirectory
                    fn26_save_directory = os.path.join(fn26_main_directory, fn26_sub_directory)

                    # Check if the directory exists, and if not, create it
                    if not os.path.exists(fn26_save_directory):
                        os.makedirs(fn26_save_directory)

                    # Create the full path for saving the model, including the timestamp
                    fn26_model_save_path = os.path.join(fn26_save_directory, f'{fn26_name_of_model}__{fn26_timestamp}.pth')

                    # Save the model's state dictionary
                    torch.save(fn26_model_to_train.state_dict(), fn26_model_save_path)

                    plt.close(fn26_fig)
                    return fn26_fig

    if fn26_inside_optuna:
        # Evaluate on the development data
        with torch.no_grad():
            fn26_model_to_train.eval()
            fn26_predictions = fn26_model_to_train(fn26_x_dev_data).squeeze()  # Forward pass on dev set
            fn26_loss_dev = fn26_loss_criterion(fn26_predictions, fn26_y_dev_data)

            # #################
            # Convert tensors to numpy arrays for DataFrame
            predictions_numpy = fn26_predictions.cpu().numpy()
            y_dev_numpy = fn26_y_dev_data.cpu().numpy()

            # Create a DataFrame
            results_df = pd.DataFrame({'Predictions': predictions_numpy, 'ActualValues': y_dev_numpy})

            # Generate a unique filename using the current timestamp
            fn26_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            fn26_filename = f"predictions_vs_actuals_{fn26_timestamp}.csv"

            # Save the DataFrame to a CSV file
            results_df.to_csv(os.path.join(os.getcwd(), fn26_filename), index=False)
            #
            # #############

            # Calculate percentage error
            global global_study_mean_percentage_error
            global_study_mean_percentage_error = calculate_mean_percent_error(fn26_predictions, fn26_y_dev_data)

            return fn26_loss_dev


def calculate_mean_percent_error(fn27_predictions, fn27_y_dev_tensor):
    """
    Description:
        Calculates the mean percentage error between the predicted values and the actual values.
        This function is useful for evaluating the performance of a regression model.

    Input:
        fn27_predictions (Tensor): Predicted values from the model.
        fn27_y_dev_tensor (Tensor): Actual values (ground truth).

    Output:
        float: The mean percentage error calculated from the predictions and actual values.

    Function-code:
        fn27_
    """

    # Ensure both tensors are of the same type and device
    fn27_predictions = fn27_predictions.to(fn27_y_dev_tensor.device).type(fn27_y_dev_tensor.dtype)

    # Calculate the percent error for each element
    fn27_percent_error = torch.abs(fn27_predictions - fn27_y_dev_tensor) / torch.abs(fn27_y_dev_tensor) * 100
    # print("Absolute Differences:", absolute_differences)

    # Calculate the mean of these percent errors
    fn27_mean_percent_error = round(torch.mean(fn27_percent_error).item(), 2)

    # #################
    # # Convert tensors to numpy arrays for DataFrame
    # predictions_numpy = fn27_predictions.cpu().numpy()
    # y_dev_numpy = fn27_y_dev_tensor.cpu().numpy()
    #
    # # Create a DataFrame
    # results_df = pd.DataFrame({'Predictions': predictions_numpy, 'ActualValues': y_dev_numpy})
    #
    # # Generate a unique filename using the current timestamp
    # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # filename = f"predictions_vs_actuals_{timestamp}.csv"
    #
    # # Save the DataFrame to a CSV file
    # results_df.to_csv(os.path.join(os.getcwd(), filename), index=False)
    #
    # #############

    return fn27_mean_percent_error


def evaluate_model(fn28_model, fn28_x_dev_norm, fn28_y_dev_data, fn28_model_index, fn28_config_df):
    """
    Description:
        Evaluates the trained model on the development set, calculates the mean percent error, and generates comparisons between predictions and actual labels for the first ten instances. This provides insights into the model's prediction accuracy and a sample of individual predictions.
        The function operates in the following steps:
        1. Converts the development set features and labels into PyTorch tensors.
        2. Sets the model to evaluation mode and generates predictions for the development set.
        3. Calculates the mean percent error between predictions and actual labels and updates this in the provided DataFrame.
        4. Generates a list of comparison strings between the model's predictions and the actual labels for the first ten instances.

    Input:
        fn28_model_eval: The trained neural network model to be evaluated.
        fn28_x_dev_norm: Normalized features for the development set as a NumPy array.
        fn28_y_dev_data: Actual labels for the development set as a NumPy array.
        fn28_model_index: An identifier for the model, used for indexing within the configuration DataFrame.
        fn28_config_df: A DataFrame to record the evaluation results, specifically the mean percent error.

    Output:
        fn28_comparisons: A list of strings, each containing a comparison of predicted and actual values for an individual instance from the development set (limited to the first ten instances).

    Function-code:
        fn28_
    """

    fn28_config_df = fn28_config_df.copy()      # Create a copy of the df, because otherwise, it is a reference and is modified in place.

    # Convert NumPy arrays to PyTorch tensors
    fn28_x_dev_tensor = torch.tensor(fn28_x_dev_norm, dtype=torch.float32)
    fn28_y_dev_tensor = torch.tensor(fn28_y_dev_data, dtype=torch.float32)

    # Evaluate the model
    with torch.no_grad():
        fn28_model.eval()
        fn28_predictions = fn28_model(fn28_x_dev_tensor).squeeze()

        # Calculate % error (for nice and intuitive reporting)
        fn28_mean_percent_error = calculate_mean_percent_error(fn28_predictions, fn28_y_dev_tensor)
        if '%err' not in fn28_config_df.columns:
            fn28_config_df['%err'] = None
        fn28_config_df.at[fn28_model_index, '%err'] = fn28_mean_percent_error
        print(f"Mean % error: {fn28_mean_percent_error} %")

    fn28_comparisons = []

    for fn28_i in range(10):
        fn28_pred = fn28_predictions[fn28_i].item()
        fn28_rounded_pred = round_to_three_custom(fn28_pred)
        fn28_actual = fn28_y_dev_tensor[fn28_i].item()
        fn28_rounded_actual = round_to_three_custom(fn28_actual)
        fn28_comparison = f"Prediction: {fn28_rounded_pred}, Actual: {fn28_rounded_actual}"
        fn28_comparisons.append(fn28_comparison)

    return fn28_comparisons, fn28_config_df


def pandas_df_to_pdf(fn29_dataframe, fn29_timestamp, fn29_figure_filenames, fn29_filename_dataset, fn29_nr_features, fn29_nr_examples, fn29_examples_model_predictions):
    """
    Description:
        Converts a pandas DataFrame into a PDF report. This function sorts the DataFrame based on a specified column,
        prepares plots, tables, and additional textual information, and compiles them into a PDF. It is particularly
        useful for creating detailed reports from DataFrame data, including visualizations.

    Input:
        fn29_dataframe (DataFrame): The DataFrame to be converted into a PDF report.
        fn29_local_timestamp (str): A timestamp string used to name the PDF file.
        fn29_figure_filenames (list): A list of filenames (strings) for figures to be included in the report.
        fn29_filename_dataset (str): The filename of the dataset used.
        fn29_nr_features (int): The number of features in the dataset.
        fn29_nr_examples (int): The number of examples in the dataset.
        fn29_examples_model_predictions (dict): A dictionary containing model predictions to be included in the report.

    Output:
        None: The function generates a PDF file as an output and does not return a value.

    Function-code:
        fn29_
    """

    fn29_filename = 'nn_hyperpara_screener output     ' + fn29_timestamp + '.pdf'

    # Sort the DataFrame in descending order based on 'mean_absolute_error'
    fn29_sorted_dataframe = fn29_dataframe.sort_values(by='%err', ascending=True)

    # Create a list of expected filenames based on sorted_dataframe's 'model_name'
    fn29_expected_filenames = [f"{fn29_row['model_name']}.png" for fn29_index, fn29_row in fn29_sorted_dataframe.iterrows()]

    # Reorder figure_filenames to match the sorted order
    fn29_figure_filenames = [fn29_filename for fn29_filename in fn29_expected_filenames if fn29_filename in fn29_figure_filenames]

    # Define main directory and subdirectory
    fn29_main_directory = '../output/nn_hyperpara_screener__output'
    fn29_sub_directory = 'manual_results'

    # Create full path for the subdirectory
    fn29_sub_directory_path = os.path.join(fn29_main_directory, fn29_sub_directory)

    # Check if the subdirectory exists, if not, create it
    if not os.path.exists(fn29_sub_directory_path):
        os.makedirs(fn29_sub_directory_path)

    fn29_pdf_full_path = os.path.join(fn29_sub_directory_path, fn29_filename)

    # Create a PDF document with ReportLab
    fn29_pdf = SimpleDocTemplate(fn29_pdf_full_path, pagesize=landscape(A4), topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    fn29_elements = []

    # Prepare title
    fn29_styles = getSampleStyleSheet()
    fn29_title = fn29_filename.split('.')[0]
    fn29_title_paragraph = Paragraph(fn29_title, fn29_styles['Title'])

    # Add title to elements
    fn29_elements.append(fn29_title_paragraph)

    # Add extra space after title
    fn29_elements.append(Spacer(1, 0.5 * inch))  # Increase space after title (second number, the one after the ,)

    # Add a line that states the dataset used.
    fn29_dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    fn29_dataset_info = f"Filename dataset: {fn29_filename_dataset}"
    fn29_dataset_paragraph = Paragraph(fn29_dataset_info, fn29_dataset_paragraph_style)
    fn29_elements.append(fn29_dataset_paragraph)

    # Add a line that states the number of features for this dataset.
    fn29_dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    fn29_dataset_info = f"Nr. of features: {fn29_nr_features}"
    fn29_dataset_paragraph = Paragraph(fn29_dataset_info, fn29_dataset_paragraph_style)
    fn29_elements.append(fn29_dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    fn29_dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    fn29_dataset_info = f"Nr. of examples train set: {fn29_nr_examples}"
    fn29_dataset_paragraph = Paragraph(fn29_dataset_info, fn29_dataset_paragraph_style)
    fn29_elements.append(fn29_dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    fn29_dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=13, spaceBefore=15, spaceAfter=10)
    fn29_dataset_info = f"Explanation of the short forms of the column names:"
    fn29_dataset_paragraph = Paragraph(fn29_dataset_info, fn29_dataset_paragraph_style)
    fn29_elements.append(fn29_dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    fn29_dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=11, spaceBefore=10, spaceAfter=10)
    fn29_dataset_info = f"alpha = learning rate, noise = amount of Gaussian noise, psi = parameter of weight initialization, %err = mean % error on dev set;"
    fn29_dataset_paragraph = Paragraph(fn29_dataset_info, fn29_dataset_paragraph_style)
    fn29_elements.append(fn29_dataset_paragraph)

    # Add a spacer
    fn29_elements.append(Spacer(1, 0.2 * inch))  # Add more space between title and table

    # Prepare data for the table (including column headers)
    fn29_data = [fn29_sorted_dataframe.columns.tolist()] + fn29_sorted_dataframe.values.tolist()

    fn29_additional_row = ['', 'Hyperparameters', '', '', '', '', '', '', '', '', '', '', '', '']  # Fewer cells than total columns
    fn29_data.insert(0, fn29_additional_row)

    # Create a table with the data
    fn29_table = Table(fn29_data)

    # Find the index of '%err' column that contains the % error.
    fn29_mae_index = fn29_sorted_dataframe.columns.tolist().index('%err')

    fn29_font_size = 8

    # Define the style for the table
    fn29_style = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), fn29_font_size),  # Fontsize for all rows of the table.

        ('SPAN', (1, 0), (12, 0)),  # Span the large cell across columns 1 to 11
        ('SPAN', (13, 0), (13, 0)),  # Span for the last cell in the first row

        # Center align content in the large cell and the small cell afterward
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),  # Center alignment for the large cell
        ('ALIGN', (13, 0), (13, 0), 'CENTER'),  # Center alignment for the small cell afterward

        # Basic formatting for the very first row
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),  # Light background color for first row
        ('GRID', (0, 0), (-1, 0), 1, colors.black),  # Grid for the first row

        # Apply styles to the second row (the original header row, now at index 1)
        ('BACKGROUND', (0, 1), (-1, 1), colors.grey),  # Header row background color
        ('TEXTCOLOR', (0, 1), (-1, 1), colors.whitesmoke),  # Header row text color
        ('ALIGN', (0, 1), (-1, 1), 'CENTER'),
        ('FONTNAME', (0, 1), (-1, 1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 1), (-1, 1), 12),

        # Different background color for the first column in the second row
        ('BACKGROUND', (0, 1), (0, 1), colors.darkblue),

        # Styles for the rest of the table starting from the third row (index 2)
        ('BACKGROUND', (0, 2), (-1, -1), colors.beige),
        ('GRID', (0, 1), (-1, -1), 1, colors.black),

        # Different background color for the first column starting from the third row
        ('BACKGROUND', (0, 2), (0, -1), colors.lightblue),

        # Additional styles for 'mean_absolute_error' column, now starting from the third row (index 2)
        ('BACKGROUND', (fn29_mae_index, 2), (fn29_mae_index, -1), colors.lightcoral),  # Column cells background color
        ('BACKGROUND', (fn29_mae_index, 1), (fn29_mae_index, 1), colors.darkred)  # Header cell background color
    ])

    # Apply the style to the table
    fn29_table.setStyle(fn29_style)

    # Add the table to the elements that will be written to the PDF
    fn29_elements.append(fn29_table)

    # Add a spacer
    fn29_elements.append(Spacer(1, 0.3 * inch))  # Add more space between value table and loss-plots.

    # Define the styles
    fn29_stylesheet = getSampleStyleSheet()

    # Get the keys in the correct order from the DataFrame
    fn29_keys_in_order = fn29_sorted_dataframe['model_name'].tolist()
    fn29_sorted_local_examples_model_predictions = {fn29_key: fn29_examples_model_predictions[fn29_key] for fn29_key in fn29_keys_in_order if fn29_key in fn29_examples_model_predictions}

    fn29_keys = list(fn29_sorted_local_examples_model_predictions.keys())

    # Number of keys to print beside each other in a row
    fn29_keys_per_row = 4

    # Initialize an index to keep track of the keys
    fn29_key_index = 0

    while fn29_key_index < len(fn29_keys):
        # Create a list to store key paragraphs for this row
        fn29_key_paragraphs = []

        # Create a list to store array paragraphs for this row
        fn29_array_paragraphs = []

        # Iterate through keys for the current row
        for fn29_i in range(fn29_key_index, min(fn29_key_index + fn29_keys_per_row, len(fn29_keys))):
            fn29_key = fn29_keys[fn29_i]

            # Add a line for the key
            fn29_key_line = Paragraph(f"<b><font size=12>{fn29_key}:</font></b><br/><br/>", fn29_stylesheet['Normal'])
            fn29_key_paragraphs.append(fn29_key_line)

            # Add a line for the value
            fn29_array_to_display = fn29_examples_model_predictions[fn29_key]
            fn29_array_paragraph = Paragraph("<br/>".join(fn29_array_to_display), fn29_stylesheet['Normal'])
            fn29_array_paragraphs.append(fn29_array_paragraph)

        # Create a table with key-value pairs for the current row
        fn29_kv_table = Table([fn29_key_paragraphs, fn29_array_paragraphs])

        # Optionally, add some styling to the kv table
        fn29_kv_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))

        # Add the kv table to elements
        fn29_elements.append(fn29_kv_table)

        # Increment the key index for the next row
        fn29_key_index += fn29_keys_per_row

        # Add a spacer to separate rows
        fn29_elements.append(Spacer(1, 0.2 * inch))  # Adjust the spacing as needed

    # Assuming figure_filenames is a list of image file names
    # Process images in pairs
    for fn29_i in range(0, len(fn29_figure_filenames), 2):
        # Create an empty row
        fn29_row = []

        for fn29_j in range(2):
            fn29_index = fn29_i + fn29_j
            if fn29_index < len(fn29_figure_filenames):
                # Load and resize image
                fn29_img = Image(fn29_figure_filenames[fn29_index])
                fn29_img.drawHeight = fn29_img.drawHeight * 0.45
                fn29_img.drawWidth = fn29_img.drawWidth * 0.45
                fn29_row.append(fn29_img)
            else:
                # Add an empty cell if no image left
                fn29_row.append('')

        # Corrected line to handle string elements
        fn29_col_widths = [fn29_img.drawWidth if isinstance(fn29_img, Image) else 0 for fn29_img in fn29_row]

        # Create a table for each row
        fn29_table = Table([fn29_row], colWidths=fn29_col_widths)

        # Add the table to elements
        fn29_elements.append(fn29_table)
        fn29_elements.append(Spacer(1, 0.5 * inch))  # Space after each row

    # Build the PDF
    fn29_pdf.build(fn29_elements)

    # Optionally, delete the temporary image files
    for fn29_img_filename in fn29_figure_filenames:
        os.remove(fn29_img_filename)


def optuna_output_to_pdf(fn30_study, fn30_mean_percent_error, fn30_timestamp, fn30_cost_function, fn30_random_seed):
    """
    Description:
        Converts the results of an Optuna optimization study into a detailed PDF report. This function generates a summary
        of the study, including the number of trials, mean percentage error, best hyperparameters, and visualizations
        of the optimization process. The PDF report provides a comprehensive overview of the study's outcomes and
        is useful for analyzing and presenting the results of the hyperparameter optimization.

    Input:
        fn29_study (Study): The Optuna study object containing the results of the optimization.
        fn29_mean_percent_error (float): The mean percentage error of the best trial in the study.
        fn29_timestamp (str): Timestamp used for naming the PDF file.
        fn29_cost_function (str): The cost function used in the study.

    Output:
        None: The function generates a PDF file as an output and does not return a value.

    Function-code:
        fn30_
    """

    fn30_file_path = "../output/nn_hyperpara_screener__output/optuna_results/optuna_report__{}.pdf".format(fn30_timestamp)
    os.makedirs(os.path.dirname(fn30_file_path), exist_ok=True)
    subprocess.run(["xdotool", "key", "F5"])                    # This is needed, otherwise you risk a dir which just says its loading the files (which is permanent, it will never load)

    # Initialize PDF in A4 portrait orientation
    fn30_c = canvas.Canvas(fn30_file_path, pagesize=A4)
    fn30_width, fn30_height = A4

    # Define margin and position for the header
    fn30_left_margin = 50
    fn30_top_position = fn30_height - 50

    # Add Header on the first page
    fn30_c.setFont("Helvetica-Bold", 16)
    fn30_c.drawString(fn30_left_margin, fn30_top_position, f"NN Optuna Optimization Report  {fn30_timestamp}")

    fn30_c.setFont("Helvetica", 12)
    fn30_c.drawString(fn30_left_margin, fn30_top_position - 30, f"Number of finished trials: {len(fn30_study.trials)}")

    fn30_c.setFont("Helvetica", 12)
    fn30_mean_percent_error = round_to_three_custom(fn30_mean_percent_error)
    fn30_c.drawString(fn30_left_margin, fn30_top_position - 60, f"Mean % error on the dev-set for the best hyperparameters found by the Optuna study: {fn30_mean_percent_error}")

    # Round best trial parameters if they are floats
    fn30_rounded_params = {fn30_k: round_to_three_custom(fn30_v) for fn30_k, fn30_v in fn30_study.best_trial.params.items()}

    # Set font for the header of the best hyperparameters section
    fn30_c.setFont("Helvetica-Bold", 12)
    fn30_header_position = fn30_top_position - 100  # Adjust position for the header

    # Add a line for the header of the best hyperparameters
    fn30_c.drawString(fn30_left_margin, fn30_header_position, "Best hyperparameters found by the Optuna study:")

    # Write each parameter on a separate line
    fn30_c.setFont("Helvetica", 12)
    fn30_line_height = 20
    fn30_param_start_position = fn30_top_position - 120  # Starting position for parameters

    fn30_i = -1  # Set a default value for fn30_i
    for fn30_i, (fn30_k, fn30_v) in enumerate(fn30_rounded_params.items()):
        # Check if the key is 'optimizer' and value is 'Adam'
        if fn30_k == "optimizer" and fn30_v == "Adam":
            fn30_additional_string = " (beta_1 = 0.9, beta_2 = 0.999, epsilon = 10e-8)"
            fn30_c.drawString(fn30_left_margin, fn30_param_start_position - (fn30_i * fn30_line_height), f"{fn30_k} : {fn30_v}{fn30_additional_string}")
        else:
            fn30_c.drawString(fn30_left_margin, fn30_param_start_position - (fn30_i * fn30_line_height), f"{fn30_k} : {fn30_v}")

    # Print the random seed line after the loop
    fn30_c.drawString(fn30_left_margin, fn30_param_start_position - ((fn30_i + 1) * fn30_line_height), f"Random seed : {fn30_random_seed}")

    fn30_param_lines = len(fn30_rounded_params)

    # Calculate position for 'Copy and paste:' line
    fn30_copy_paste_position = fn30_param_start_position - (fn30_param_lines * fn30_line_height) - fn30_line_height - 50

    # Add a line for 'Copy and paste:' in bold
    fn30_c.setFont("Helvetica-Bold", 12)  # Set font to bold for 'Copy and paste:'
    fn30_c.drawString(fn30_left_margin, fn30_copy_paste_position, "Copy and paste:")

    # Convert to list of strings (only values)
    fn30_param_values = [str(v) for v in fn30_rounded_params.values()]

    # Remove the first element from the list
    fn30_num_hidden_layers = int(fn30_param_values[0])
    fn30_param_values = fn30_param_values[1:]  # Keeps all elements of the list except the first one
    fn30_param_values.insert(0, 'Optuna_model')
    fn30_param_values.append(fn30_cost_function)

    fn30_iterations = 8 + fn30_num_hidden_layers     # The Optimizer is at that pos, and " also need to be added there, because the params of the optimizer should be in one field in the .csv file

    for fn30_i in range(fn30_iterations):
        if fn30_i == 1:
            fn30_param_values[fn30_i] = '"' + fn30_param_values[fn30_i]         # Add an " before the first neuron number.
        elif fn30_i == fn30_num_hidden_layers + 1:
            fn30_param_values[fn30_i] = fn30_param_values[fn30_i] + '"'         # Add an " after the last neuron number
        elif fn30_i == fn30_iterations - 1:
            fn30_param_values[fn30_i] = '"' + fn30_param_values[fn30_i]

    fn30_param_values[-6] = fn30_param_values[-6] + '"'       # Add an " after the last parameter of the optimizer (or after the optimizer itself).

    # Join the strings without spaces after commas
    fn30_values_str = ','.join(fn30_param_values)

    fn30_values_str += f",{fn30_random_seed}"          # Add the random seed at the end

    if 'Adam' in fn30_values_str:                       # Add the beta parameters of Adam, because these are not optimized by Optuna.
        # Find the index where 'Adam' ends
        index_after_adam = fn30_values_str.find('Adam') + len('Adam')
        # Insert the substring ',0.9,0.999' after 'Adam'
        fn30_values_str = fn30_values_str[:index_after_adam] + ',0.9,0.999,10e-8' + fn30_values_str[index_after_adam:]

    # Set font and draw the string
    fn30_c.setFont("Helvetica", 4)  # Set back to regular font for the values
    fn30_c.drawString(fn30_left_margin, fn30_copy_paste_position - fn30_line_height, fn30_values_str)

    # Start figures from the second page
    fn30_c.showPage()

    # List of plot filenames
    fn30_plot_files = [
        "optimization_history.png",
        "param_importances.png",
        "edf.png"
    ]

    # Image dimensions
    fn30_img_width = 500
    fn30_img_height = 250  # Adjusted height to fit three images per page

    # Add plots to PDF, three on each page
    for fn30_i, fn30_plot_file in enumerate(fn30_plot_files):
        if fn30_i % 3 == 0 and fn30_i > 0:  # Create a new page after every three plots
            fn30_c.showPage()

        fn30_full_path = os.path.join(os.getcwd(), fn30_plot_file)
        if os.path.exists(fn30_full_path):
            # Calculate image Y position (top, middle, bottom image)
            fn30_img_y_position = fn30_top_position - 220 - (fn30_i % 3) * (fn30_img_height + 5)

            fn30_c.drawImage(fn30_full_path, fn30_left_margin, fn30_img_y_position, width=fn30_img_width, height=fn30_img_height, preserveAspectRatio=True)

            os.remove(fn30_full_path)    # Delete plot file
        else:
            print(f"Error: Plot file not found: {fn30_full_path}")
            sys.exit()

    # Save the PDF
    fn30_c.save()


def parse_optuna_hyperparameter_ranges(fn31_csv_file):
    """
    Description:
        Parses a CSV file to extract hyperparameter ranges for Optuna optimization.

        The function reads a CSV file where the first column from the second row onwards contains the names of hyperparameters,
        and the second column contains their values. It processes each row to determine the type of hyperparameter
        (e.g., list of categorical options, numerical range, single value) and formats it accordingly for use in hyperparameter optimization.
        The function supports:
        - Categorical hyperparameters specified as a comma-separated string.
        - Numerical ranges specified as a min-max pair, separated by a comma.
        - Single numerical or categorical values.
        - Special handling for 'batch_size' to interpret it as a single value or a list of values.

    Input:
        fn31_csv_file: A string representing the path to the CSV file containing hyperparameter configurations.

    Output:
        fn31_hyperparameters: A dictionary where keys are hyperparameter names and values are the respective
                           hyperparameter configurations (ranges, lists of options, or single values).

    Function-code:
        fn31_
    """

    fn31_df = pd.read_csv(fn31_csv_file, header=None, skiprows=1)
    fn31_hyperparameters = {}
    for fn31_index, fn31_row in fn31_df.iterrows():
        fn31_name = fn31_row[0]
        fn31_value = str(fn31_row[1])

        if fn31_name == 'batch_size':
            if ',' in fn31_value:  # It's a list of batch sizes
                fn31_hyperparameters[fn31_name] = [int(fn31_val.strip()) for fn31_val in fn31_value.split(',')]
            else:  # It's a single batch size
                fn31_hyperparameters[fn31_name] = int(fn31_value)
        elif fn31_value[0].isalpha():  # It's a list of categorical options
            if ',' in fn31_value:
                fn31_hyperparameters[fn31_name] = fn31_value.split(',')
            else:
                fn31_hyperparameters[fn31_name] = fn31_value
        elif fn31_value[0].isdigit():  # It's a numerical range
            if ',' in fn31_value:
                fn31_min_val, fn31_max_val = map(float, fn31_value.split(','))
                fn31_hyperparameters[fn31_name] = (fn31_min_val, fn31_max_val)
            else:
                fn31_hyperparameters[fn31_name] = int(fn31_value)

    return fn31_hyperparameters


def run_optuna_study(fn32_train_dev_dataframes, fn32_timestamp, fn32_n_trials, fn32_input_size):
    """
    Description:
        Conducts an Optuna hyperparameter optimization study for a neural network model. This function sets up
        and executes an Optuna study to find the best hyperparameters for the given model architecture and data.
        It also generates and saves visualization plots for the study and compiles a detailed PDF report of the results.

    Input:
        fn32_train_dev_dataframes (tuple of DataFrames): Tuple containing training and development datasets.
        fn32_timestamp (str): A timestamp string used for naming output files.
        fn32_n_trials (int): The number of trials to run in the Optuna study.
        fn32_input_size (int): The size of the input layer for the neural network model.

    Output:
        dict: The best hyperparameters found by the Optuna study.

    Function-code:
        fn32_
    """

    def objective(fn33_trial, fn33_train_dev_dataframes, fn33_input_size, fn33_hyperparameter_ranges, fn33_random_seed):
        """
        Description:
            The objective function for the Optuna study, defining how the model should be trained and evaluated.
            It suggests hyperparameters, builds a model with those hyperparameters, and calculates the loss on the
            development set. The function is called by the Optuna study for each trial.

        Input:
            fn33_trial (Trial): The current Optuna trial instance.
            fn33_train_dev_dataframes (tuple of DataFrames): Tuple containing training and development datasets.
            fn33_input_size (int): The size of the input layer for the neural network model.

        Output:
            float: The loss of the model on the development set for the current set of hyperparameters.

        Function-code:
            fn33_
        """

        # Hyperparameters to be tuned by Optuna
        # Hyperparameter range for the number of layers
        fn33_num_layers = fn33_trial.suggest_int('num_hidden_layers', *fn33_hyperparameter_ranges['num_hidden_layers'])

        # Dynamically creating hyperparameters for each layer's neuron count
        fn33_nr_neurons = []
        fn33_num_neurons = 1
        for fn33_i in range(1, fn33_num_layers + 1):         # So that the first hidden layer has the number 1 (nr. 0 could be confused with input layer)
            fn33_num_neurons = fn33_trial.suggest_int(f'neurons_hidden_layer_{fn33_i}', *fn33_hyperparameter_ranges['nr_neurons_hidden_layers'])
        fn33_nr_neurons.append(fn33_num_neurons)
        fn33_nr_output_neurons = fn33_trial.suggest_int('num_neurons_output_layer', *fn33_hyperparameter_ranges['num_neurons_output_layer'])

        fn33_acti_fun_type = fn33_trial.suggest_categorical('acti_fun', fn33_hyperparameter_ranges['acti_fun_hidden_layers'])
        fn33_acti_fun_out_type = fn33_trial.suggest_categorical('acti_fun_out', fn33_hyperparameter_ranges['acti_fun_output_layer'])
        fn33_nr_epochs = fn33_trial.suggest_int('nr_epochs', *fn33_hyperparameter_ranges['nr_epochs'])
        fn33_batch_size = fn33_trial.suggest_categorical('batch_size', fn33_hyperparameter_ranges['batch_size'])
        fn33_noise = fn33_trial.suggest_float('noise', *fn33_hyperparameter_ranges['gaussian_noise'])

        # local_optimizer_column_list = hyperparameter_ranges['optimizer'].split(',')
        fn33_optimizer_type = fn33_trial.suggest_categorical('optimizer', fn33_hyperparameter_ranges['optimizer'])

        # Conditional hyperparameters for the optimizer
        fn33_optim_add_params = None
        fn33_momentum = None

        if fn33_optimizer_type == 'Adam':
            fn33_beta1 = 0.9                    # Fixed value for beta1 (Andrew Ng almost never tunes these)
            fn33_beta2 = 0.999                  # Fixed value for beta2
            fn33_epsilon = 10e-8
            fn33_optim_add_params = [fn33_beta1, fn33_beta2, fn33_epsilon]
        elif fn33_optimizer_type == 'SGD':
            fn33_momentum = fn33_trial.suggest_float('momentum', 0.5, 0.9)
        else:
            print(f"Error: Optimizer name {fn33_optimizer_type} not valid!")
            sys.exit()

        fn33_learning_rate = fn33_trial.suggest_float('alpha', *fn33_hyperparameter_ranges['alpha'], log=True)
        fn33_decay_rate = fn33_trial.suggest_float('decay_rate', *fn33_hyperparameter_ranges['decay_rate'], log=True)
        fn33_lamda = fn33_trial.suggest_float('lamda', *fn33_hyperparameter_ranges['lamda'], log=True)
        fn33_dropout = fn33_trial.suggest_float('dropout', *fn33_hyperparameter_ranges['dropout'])
        fn33_cost_function = fn33_hyperparameter_ranges['cost_function']
        fn33_psi = fn33_trial.suggest_float('psi', *fn33_hyperparameter_ranges['psi'])

        fn33_hyperparams = {
            'nr_neurons_hidden_layers': fn33_nr_neurons,
            'nr_neurons_output_layer': fn33_nr_output_neurons,
            'activation_function_type': fn33_acti_fun_type,
            'acti_fun_out': fn33_acti_fun_out_type,
            'batch_size': fn33_batch_size,
            'optimizer_type': fn33_optimizer_type,
            'optim_add_params': fn33_optim_add_params if fn33_optimizer_type == 'Adam' else [fn33_momentum],
            'learning_rate': fn33_learning_rate,
            'lamda': fn33_lamda,
            'dropout': fn33_dropout,
            'psi_value': fn33_psi,
            'cost_function': fn33_cost_function,
            'random_seed': fn33_random_seed,
        }

        fn33_model, fn33_optimizer, fn33_criterion, fn33_train_loader, fn33_dev_loader, fn33_x_dev_tensor, fn33_y_dev_tensor = prepare_model_training(fn33_hyperparams, fn33_train_dev_dataframes, fn33_input_size)

        fn33_loss_dev = train_and_optionally_plot(fn33_model, fn33_train_loader, fn33_nr_epochs, fn33_optimizer, fn33_criterion, fn33_x_dev_tensor, fn33_y_dev_tensor, fn33_noise, fn33_learning_rate, fn33_decay_rate)

        return fn33_loss_dev

    fn32_study = optuna.create_study(direction='minimize')

    fn32_hyperparameter_ranges = parse_optuna_hyperparameter_ranges('../input/nn_hyperpara_screener_optuna_ranges.csv')
    fn32_random_seed = fn32_hyperparameter_ranges.get('random_seed', 999)
    fn32_study.optimize(lambda trial: objective(trial, fn32_train_dev_dataframes, fn32_input_size, fn32_hyperparameter_ranges, fn32_random_seed), fn32_n_trials)

    best_hyperparams = fn32_study.best_trial.params
    print("\n\nBest hyperparameters: ", best_hyperparams, "\n")

    #################################################################################
    # Generate a range of plots that Optuna has to offer
    fn32_fig = optuna.visualization.plot_optimization_history(fn32_study)
    fn32_fig.write_image("optimization_history.png")

    fn32_fig = optuna.visualization.plot_param_importances(fn32_study)
    fn32_fig.write_image("param_importances.png")

    fn32_fig = optuna.visualization.plot_edf(fn32_study)
    fn32_fig.write_image("edf.png")
    #################################################################################

    fn32_hyperparameter_ranges = parse_optuna_hyperparameter_ranges('../input/nn_hyperpara_screener_optuna_ranges.csv')
    fn32_cost_function = fn32_hyperparameter_ranges['cost_function']
    optuna_output_to_pdf(fn32_study, global_study_mean_percentage_error, fn32_timestamp, fn32_cost_function, fn32_random_seed)       # Generate a report for the Optuna study.

    return fn32_study.best_trial.params


# End of the section where all the functions and classes are stored.
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Section with the function calls and class instantiations and the rest of the code.

# Get the command line arguments (excluding the script name)
cmd_args = sys.argv[1:]

# Call the parse_command_line_args function with the command line arguments
train_set_name, dev_set_name, show_plots, run_optuna, nr_trials = parse_command_line_args(cmd_args)

train_set_data, dev_set_data = load_datasets(train_set_name, dev_set_name)          # Load the dataset and assign it to variables

x_train = train_set_data.iloc[:, :-1].values
y_train = train_set_data.iloc[:, -1].values

x_dev = dev_set_data.iloc[:, :-1].values
y_dev = dev_set_data.iloc[:, -1].values

train_dev_dataframes = [x_train, y_train, x_dev, y_dev]

# Create a "timestamp". This is a string with the day and the exact time at the start of the execution and will label the files and serve as "code".
timestamp = create_timestamp()

input_size = x_train.shape[1]  # Automatically set input_size based on the number of features (number of columns)
amount_of_rows = x_train.shape[0]

if run_optuna:                          # Instead of manually tuning, the optimizer Optuna does it automatically.
    global_study_mean_percentage_error = None
    best_trial_parameter = run_optuna_study(train_dev_dataframes, timestamp, nr_trials, input_size)
    os.system('play -nq -t alsa synth 1 sine 600')  # Give a notification sound when done.
    sys.exit()

delete_pth_files()                          # Deletes all .pth files from the dir nn_hyperpara_screener_model_pth_files (so they do not pile up during screening)

config = read_and_process_config(amount_of_rows)        # Get the hyperparameter info from the config file.

check_dataframe_columns(config)                         # Check if the columns of the config file are correctly set up

nr_of_models = config.shape[0]                              # Get the number of rows = number of models to test

loss_vs_epoch_figures = []                                  # In this list, all the figures of the model trainings are stored.

examples_model_predictions = {}                             # This hash stored all the arrays of the 10 examples of the model predictions.

##################################################
# Section where it loops over the models and trains them.

for model_nr in range(nr_of_models):

    # This function assigns the values from the config file about the hyperparameters to the respective variables
    hyperparams = assign_hyperparameters_from_config(config, model_nr, amount_of_rows)

    model_name = hyperparams['model_name']
    nr_epochs = hyperparams['nr_epochs']
    noise_stddev = hyperparams['noise_stddev']
    learning_rate = hyperparams['learning_rate']
    decay_rate = hyperparams['decay_rate']

    # The print statement below is to separate the different terminal sections for the models.
    print("\n\n############################################################################################\n")
    print(f"Model: {model_name}\n")         # To assign potential terminal messages to a model.

    model, optimizer, criterion, train_loader, dev_loader, x_dev_tensor, y_dev_tensor = prepare_model_training(hyperparams, train_dev_dataframes, input_size)

    inside_optuna = False
    fig = train_and_optionally_plot(model, train_loader, nr_epochs, optimizer, criterion, x_dev_tensor, y_dev_tensor, noise_stddev, learning_rate, decay_rate, inside_optuna, model_name, timestamp, show_plots)
    loss_vs_epoch_figures.append(fig)

    ten_examples_model_predictions, config = evaluate_model(model, x_dev, y_dev, model_nr, config)
    examples_model_predictions[model_name] = ten_examples_model_predictions                    # Store the predictions in a hash for the pdf.

##################################################

figure_filenames_list = []  # To store filenames of saved figures

for i, fig in enumerate(loss_vs_epoch_figures):
    model_name = config.iloc[i]['model_name']
    img_filename = f"{model_name}.png"
    fig.savefig(img_filename, bbox_inches='tight')
    figure_filenames_list.append(img_filename)

pandas_df_to_pdf(config, timestamp, figure_filenames_list, train_set_name, input_size, amount_of_rows, examples_model_predictions)

os.system('play -nq -t alsa synth 1 sine 600')                                      # Give a notification sound when done.
