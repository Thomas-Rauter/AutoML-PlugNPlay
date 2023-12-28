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
from copy import deepcopy
import torch.optim as optim


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Section, where all the functions and classes are stored. All the function calls and class instantiations are below this section.


def round_to_three_custom(fn1_num):
    """
    Description:
        Modify a floating-point number to have at most three non-zero digits after the decimal point.

    Input:
        fn1_num (float): The number to be modified.

    Output:
        float: The modified number with at most three non-zero digits after the decimal point.

    Function-code:
        fn1_
    """

    if isinstance(fn1_num, (float, np.float64, np.float32)):            # To avoid errors when this is applied to a mixed list.
        fn1_num_str = str(fn1_num)
        if '.' in fn1_num_str:
            fn1_whole, fn1_decimal = fn1_num_str.split('.')             # Splitting the number into whole and decimal parts
            fn1_non_zero_count = 0
            fn1_found_non_zero_digit_before_dot = False

            for fn1_local_i, fn1_digit in enumerate(fn1_whole):         # Loop over the decimal digits (starting from the . on leftwards)
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
                    fn1_new_decimal = fn1_decimal[:fn1_local_i]
                    return float(fn1_whole + '.' + fn1_new_decimal)

            return float(fn1_num_str)       # Return the original number if less than 3 non-zero digits
        else:
            return int(fn1_num_str)         # Return the original number if no decimal part
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
                           'optimizer', 'alpha', 'lamda', 'dropout', 'psi', 'cost_fun']
            fn2_file.write(','.join(fn2_headers))
            fn2_starter_model = ["\nModel_1", '"10,1"', "ReLU", "Linear", 100, 64, 0, "Adam,0.9,0.99", 0.01, 0.001, 0, 1, "MSELoss"]
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
        except Exception as e:
            pass  # Optionally handle errors silently or log them


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
        'optimizer': 'Adam,0.9,.099',
        'alpha': 0.01,
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
        'lamda': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, float),
        'dropout': lambda fn9_x: pd.isna(fn9_x) or (isinstance(fn9_x, (int, float)) and 0 <= fn9_x < 1),
        'psi': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, (int, float)),
        'cost_fun': lambda fn9_x: pd.isna(fn9_x) or isinstance(fn9_x, str),
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


def assign_hyperparameters_from_config(local_pandas_df, local_row_nr, local_amount_of_rows):
    local_variable_names = ['model_name', 'nr_neurons_str', 'activation_function_type', 'acti_fun_out', 'nr_epochs',
                            'batch_size', 'noise_stddev', 'optimizer_type', 'learning_rate', 'lamda', 'dropout',
                            'psi_value', 'cost_function']
    local_column_names = ['model_name', 'neurons', 'acti_fun', 'acti_fun_out', 'epochs', 'batch_size',
                          'noise', 'optimizer', 'alpha', 'lamda', 'dropout', 'psi', 'cost_fun']

    local_hyperparams = {}

    for local_var_name, local_col_name in zip(local_variable_names, local_column_names):
        local_value = local_pandas_df[local_col_name].iloc[local_row_nr]

        if local_var_name == 'nr_neurons_str':  # Special treatment for nr_neurons
            local_neuron_list = [int(local_neuron) for local_neuron in local_value.split(',')]
            local_nr_output_neurons = local_neuron_list.pop()
            local_hyperparams['nr_neurons_hidden_layers'] = local_neuron_list
            local_hyperparams['nr_neurons_output_layer'] = local_nr_output_neurons
        elif local_var_name == 'optimizer_type':
            local_split_values = local_value.split(',')
            local_optimizer = local_split_values[0]
            local_optim_add_params = [float(local_item) for local_item in local_split_values[1:]]
            local_hyperparams['optimizer_type'] = local_optimizer
            local_hyperparams['optim_add_params'] = local_optim_add_params
        else:
            local_hyperparams[local_var_name] = local_value

    if local_hyperparams['batch_size'] > local_amount_of_rows:
        local_hyperparams['batch_size'] = local_amount_of_rows
        local_pandas_df.loc[local_row_nr, "batch_size"] = local_amount_of_rows

    if local_hyperparams['dropout'] >= 1 or local_hyperparams['dropout'] < 0:
        print("Error: dropout must be smaller than 1 and greater than 0!")
        sys.exit()

    return local_hyperparams


class NNmodel(nn.Module):
    def __init__(self, local_input_size, local_nr_neurons, local_activation_function, local_nr_output_neurons, local_acti_fun_out, local_dropout):
        super(NNmodel, self).__init__()
        layers = []
        for neurons in local_nr_neurons:
            layers.append(nn.Linear(local_input_size, neurons))      # The linear function is the "z = weight_vector * feature_vector + bias" part
            layers.append(nn.BatchNorm1d(neurons))                   # Batch normalization on the z (before the activation function is applied).
            layers.append(local_activation_function)                 # This is then the "g(z)" part.
            layers.append(nn.Dropout(local_dropout))                 # Dropout layer, this adds dropout regularization to the network, if local_dropout is bigger than 0.
            local_input_size = neurons

        layers.append(nn.Linear(local_input_size, local_nr_output_neurons))                # Output layer(s). For regression, there is usually only one output layer with no activation function.
        layers.append(local_acti_fun_out)                                                  # If local_acti_fun_out is 'linear', it will not do any harm, then just nothing happens.
        self.model = nn.Sequential(*layers)

    def forward(self, local_x):
        return self.model(local_x)                                   # Passes the input x through the model and returns the result (y-predictions)


class ArcTanActivation(torch.nn.Module):
    def forward(self, local_input):
        return torch.atan(local_input)


class Mish(torch.nn.Module):
    def forward(self, local_input):
        return local_input * torch.tanh(F.softplus(local_input))


class Hardswish(torch.nn.Module):
    def forward(self, local_input):
        return local_input * F.relu6(local_input + 3) / 6


class CoLUActivation(nn.Module):
    def forward(self, local_input):
        return local_input / (1 - torch.pow(local_input, -(local_input + torch.exp(local_input))))


def get_activation_function(local_acti_fun_type):
    """
    Returns the corresponding PyTorch activation function based on the given type.

    Parameters:
    local_acti_fun_type (str): The name of the activation function.

    Returns:
    nn.Module: The PyTorch activation function object.

    If the specified activation function is not recognized, the function
    prints an error message and exits the script.
    """

    if local_acti_fun_type == "Linear":                 # Basically means no activation function
        return nn.Identity()                            # Linear, g(z) = z
    elif local_acti_fun_type == "ReLU":                 # All the ReLU functions should only be used in the hidden layers.
        return nn.ReLU()                                # ReLU, g(z) = max(0, z)
    elif local_acti_fun_type == "LeakyReLU":            # In case of dead neurons
        return nn.LeakyReLU()                           # Leaky ReLU, g(z) = z if z > 0, else alpha * z, alpha is fixed
    elif local_acti_fun_type == "PReLU":
        return nn.PReLU()                               # PReLU, g(z) = z if z > 0, else alpha * z, alpha is a hyperparameter
    elif local_acti_fun_type == "ELU":                  # ELU = exponential linear units
        return nn.ELU()                                 # ELU, g(z) = alpha * (e^z - 1) if z < 0, else z, alpha is fixed
    elif local_acti_fun_type == "GELU":                 # Gaussian error linear units.
        return nn.GELU()                                # GELU, g(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3))
    elif local_acti_fun_type == "SELU":
        return nn.SELU()                                # SELU, g(z) = scale * [max(0, z) + min(0, alpha * (e^z - 1))]
    elif local_acti_fun_type == "CELU":
        return nn.CELU()                                # CeLU, g(z) = max(0, z) + min(0, alpha * (exp(z/alpha) - 1))
    elif local_acti_fun_type == "CoLU":                 # Beneficial for deep networks.
        return CoLUActivation()                         # CoLU, g(z) = z / (1 - z^-(z + e^z))
    elif local_acti_fun_type == "Softplus":             # Smooth approximation of ReLU
        return nn.Softplus()                            # Softplus, g(z) = log(1 + e^z)
    elif local_acti_fun_type == "Swish":
        return nn.SiLU()                                # Swish, g(z) = z * sigmoid(z)
    elif local_acti_fun_type == "Hardswish":            # Approximation of the Swish activation function that is computationally more efficient.
        return Hardswish()                              # Hard Swish, g(z) = z * ReLU6(z + 3) / 6
    elif local_acti_fun_type == "Sigmoid":              # Sigmoid and Tanh are better for classification.
        return nn.Sigmoid()                             # Sigmoid, g(z) = 1 / (1 + e^(-z))
    elif local_acti_fun_type == "Tanh":                 # Very similar to the sigmoid function, for classification.
        return nn.Tanh()                                # Tanh, g(z) = (e^z - e^(-z)) / (e^z + e^(-z))
    elif local_acti_fun_type == "ArcTan":               # Custom function build with class.
        return ArcTanActivation()                       # ArcTan, g(z) = arctan(z)
    elif local_acti_fun_type == "Softmax":              # For multiclass classification problems.
        return nn.Softmax(dim=1)                        # Softmax, g(z_i) = e^(z_i) / sum(e^(z_j) for j in all outputs)
    elif local_acti_fun_type == "Mish":                 # Both for regression and classification. Its smooth and non-monotonic.
        return Mish()                                   # Mish, g(z) = z * tanh(softplus(z)) = z * tanh(log(1 + e^z))
    else:
        print("Error: No activation function was specified in the config file!")
        sys.exit()


def create_initial_weights(local_input_size, n_neurons, local_nr_output_neurons, local_acti_fun_type, psi):
    """
    Creates initial weights and biases for a neural network given the input size,
    the number of neurons in each hidden layer, and the output size. Weights are
    initialized using a custom formula, and biases are initialized to a small constant value.

    Parameters:
    local_input_size (int): The number of features in the input data.
    n_neurons (list of int): A list containing the number of neurons in each hidden layer.
    local_output_size (int, optional): The number of neurons in the output layer.

    Returns:
    weights (list of Tensor): A list of weight matrices, where each matrix corresponds
                              to the weights for one layer in the network.
    biases (list of Tensor): A list of bias vectors, where each vector corresponds
                             to the biases for one layer in the network.
    """

    weights = []
    biases = []

    # Start with the input size
    layer_input_size = local_input_size

    # Create weights and biases for each hidden layer
    for neurons in n_neurons:
        # Custom weight initialization
        if local_acti_fun_type == "ReLU":
            local_w = torch.randn(layer_input_size, neurons) * np.sqrt(psi * 2 / layer_input_size)
        elif local_acti_fun_type == "Tanh":
            local_w = torch.randn(layer_input_size, neurons) * np.sqrt(psi * 1 / layer_input_size)      # Xavier initialization
        else:
            local_w = torch.randn(layer_input_size, neurons) * np.sqrt(psi * 1 / layer_input_size)

        weights.append(local_w)

        # Bias initialization
        local_b = torch.Tensor(neurons).fill_(0.01)
        biases.append(local_b)

        # The output of this layer is the input to the next layer
        layer_input_size = neurons

    # Custom weight initialization for the output layer
    local_w = torch.randn(n_neurons[-1], local_nr_output_neurons) * np.sqrt(1/n_neurons[-1])
    weights.append(local_w)

    local_b = torch.Tensor(local_nr_output_neurons).fill_(0.01)
    biases.append(local_b)

    return weights, biases


def apply_stored_weights(local_model, local_weights, local_biases):
    """
    Applies stored weights and biases to each linear layer in a given model.

    This function iterates through each layer of the provided model. If the layer is an instance of nn.Linear
    (a linear layer), the function sets the layer's weights and biases to the stored values provided. This is
    particularly useful for initializing a model with a specific set of weights and biases, ensuring consistency
    in model initialization across different instances or runs.

    Parameters:
    local_model (nn.Module): The neural network model to which the weights and biases will be applied.
                             The model should be an instance of nn.Module or a subclass thereof.
    local_weights (List[Tensor]): A list of PyTorch tensors representing the weights for each linear layer in the model.
                                  The order of tensors in the list should match the order of linear layers in the model.
    local_biases (List[Tensor]): A list of PyTorch tensors representing the biases for each linear layer in the model.
                                 Similar to local_weights, the order should match the linear layers' order.

    The function does not return any value; it modifies the model in-place.

    Note: This function assumes that the number of tensors in local_weights and local_biases matches the number of linear
    layers in local_model. It applies each set of weights and biases to the corresponding linear layer based on their
    order in the model.
    """

    linear_layer_count = 0
    for layer in local_model.model:
        if isinstance(layer, nn.Linear):
            # Transpose the weight matrix to match [out_features, in_features] (this is to backtranspose the matrix, it is somehow in the wrong format).
            transposed_weight = local_weights[linear_layer_count].T
            layer.weight.data = transposed_weight
            layer.bias.data = local_biases[linear_layer_count]
            linear_layer_count += 1


def get_optimizer(local_model_parameters, local_optimizer_type, local_learning_rate, local_lamda, local_additional_params):
    """
    Creates and returns a PyTorch optimizer object based on the specified type and arguments.

    Args:
        local_model_parameters: The parameters of the model to be optimized.
        local_optimizer_type (str): Type of the optimizer (e.g., 'Adam', 'RMSprop', 'SGD', etc.).
        local_learning_rate (float): Strength of decay.
        local_additional_params (list): Additional parameters specific to the optimizer type.

    Returns:
        An instance of the specified optimizer initialized with the given parameters and arguments.
    """
    local_supported_optimizers = {
        "Adam": optim.Adam,
        "SGD": optim.SGD,
        "RMSprop": optim.RMSprop,
        "Adagrad": optim.Adagrad,
        "Adadelta": optim.Adadelta,
        "Nesterov-SGD": lambda params, lr, **kwargs: optim.SGD(params, lr, **kwargs, nesterov=True),
        "LBFGS": optim.LBFGS,
        "AdamW": optim.AdamW,
        "Adamax": optim.Adamax
    }

    local_optimizer_class = local_supported_optimizers.get(local_optimizer_type)
    if not local_optimizer_class:
        raise ValueError(f"Unsupported optimizer type: {local_optimizer_type}")

    # Build the keyword arguments for the optimizer (lr and weight_decay are the consensus keywords for all those optimizers).
    # weight_decay = L2 regularization. No other form of regularization is directly supported for the PyTorch optimizers.
    local_kwargs = {
        'lr': local_learning_rate,
        'weight_decay': local_lamda,
    }

    # **dict(zip(['betas', 'momentum', 'dampening', 'alpha', 'tolerance_grad', 'tolerance_change', 'max_iter', 'history_size', 'eps'], local_additional_params))

    if local_optimizer_type == 'Adam':
        # Assuming the first two elements in local_additional_params are beta1 and beta2
        betas = (local_additional_params[0], local_additional_params[1])
        local_kwargs['betas'] = betas

    local_optimizer = local_optimizer_class(local_model_parameters, **local_kwargs)
    return local_optimizer


def get_criterion(local_cost_function):
    """
    Dynamically retrieves and returns a loss function from the PyTorch nn module based on a given string.

    Args:
        local_cost_function (str): Name of the loss function as a string. This should correspond to a class name in torch.nn.

    Returns:
        local_criterion: An instance of the specified loss function class from torch.nn.

    Raises:
        SystemExit: If the specified loss function is not found in torch.nn or if no loss function is provided.
    """
    if isinstance(local_cost_function, str):
        local_loss_class = getattr(nn, local_cost_function, None)  # Dynamically get the loss class from the nn module

        if local_loss_class is not None:                           # Check if the loss class was successfully retrieved
            local_criterion = local_loss_class()                   # Instantiate the loss class
            return local_criterion
        else:
            print(f"Error: '{local_cost_function}' is not a valid loss function in torch.nn.")
            sys.exit(1)
    else:
        print("Error: No cost function was specified in the config file!")
        sys.exit(1)


def prepare_model_training(local_hyperparams, local_train_dev_dataframes, local_input_size):
    """
    Prepares a neural network for training by setting up the model, optimizer, criterion, and data loaders.

    Parameters:
    - local_activation_function_type: Type of the activation function.
    - local_acti_fun_out: Output activation function type.
    - local_input_size: Input size of the neural network.
    - local_nr_neurons: Number of neurons.
    - local_nr_output_neurons: Number of output neurons.
    - local_psi_value: Psi value for initializing weights.
    - local_optimizer_type: Type of optimizer to use.
    - local_learning_rate: Learning rate for the optimizer.
    - local_lamda: Lambda value for the optimizer.
    - local_optim_add_params: Additional parameters for the optimizer.
    - local_cost_function: Cost function for the model.
    - local_x_train: Training data.
    - local_y_train: Training labels.
    - local_x_dev: Development data.
    - local_y_dev: Development labels.
    - local_batch_size: Batch size for data loading.

    Returns:
    A tuple containing the model, optimizer, criterion, train loader, and dev loader.
    """

    local_nr_neurons = local_hyperparams['nr_neurons_hidden_layers']
    local_nr_output_neurons = local_hyperparams['nr_neurons_output_layer']
    local_activation_function_type = local_hyperparams['activation_function_type']
    local_acti_fun_out = local_hyperparams['acti_fun_out']
    local_batch_size = local_hyperparams['batch_size']
    local_optimizer_type = local_hyperparams['optimizer_type']
    local_optim_add_params = local_hyperparams['optim_add_params']
    local_learning_rate = local_hyperparams['learning_rate']
    local_lamda = local_hyperparams['lamda']
    local_dropout = local_hyperparams['dropout']
    local_psi_value = local_hyperparams['psi_value']
    local_cost_function = local_hyperparams['cost_function']

    local_x_train = local_train_dev_dataframes[0]
    local_y_train = local_train_dev_dataframes[1]
    local_x_dev = local_train_dev_dataframes[2]
    local_y_dev = local_train_dev_dataframes[3]

    # Get activation functions
    local_activation_function = get_activation_function(local_activation_function_type)
    local_acti_fun_output = get_activation_function(local_acti_fun_out)

    # Initialize weights and biases
    local_init_weights, local_init_biases = create_initial_weights(local_input_size, local_nr_neurons, local_nr_output_neurons,
                                                                   local_activation_function_type, local_psi_value)

    # Create the model
    local_model = NNmodel(local_input_size, local_nr_neurons, local_activation_function, local_nr_output_neurons,
                          local_acti_fun_output, local_dropout)
    apply_stored_weights(local_model, local_init_weights, local_init_biases)

    local_model_parameters = local_model.parameters()
    local_optimizer = get_optimizer(local_model_parameters, local_optimizer_type, local_learning_rate, local_lamda, local_optim_add_params)

    local_criterion = get_criterion(local_cost_function)

    # Convert arrays to tensors
    local_x_train_tensor = torch.tensor(local_x_train, dtype=torch.float32)  # float32 is common for nn, because they provide a good balance between precision and computational efficiency.
    local_y_train_tensor = torch.tensor(local_y_train, dtype=torch.float32)
    local_x_dev_tensor = torch.tensor(local_x_dev, dtype=torch.float32)
    local_y_dev_tensor = torch.tensor(local_y_dev, dtype=torch.float32)

    local_batch_size = int(local_batch_size)

    # Define a DataLoader
    # Both the train data (x) and the labels (y) are combined into a 'TensorDataset', which makes it easier to iterate over the data during training.
    local_train_dataset = TensorDataset(local_x_train_tensor, local_y_train_tensor)
    # Without shuffling, the rows are always in the same order and the batches are always identical (which is suboptimal for learning).
    local_train_loader = DataLoader(local_train_dataset, batch_size=local_batch_size, shuffle=True, drop_last=True)  # Drop the last batch if there are not enough examples for the batch size left
    local_dev_dataset = TensorDataset(local_x_dev_tensor, local_y_dev_tensor)
    local_dev_loader = DataLoader(local_dev_dataset, batch_size=local_batch_size, shuffle=False, drop_last=True)  # Drop the last batch if there are not enough examples for the batch size left

    return local_model, local_optimizer, local_criterion, local_train_loader, local_dev_loader, local_x_dev_tensor, local_y_dev_tensor


def train_and_optionally_plot(model_to_train, training_loader, epochs_num, training_optimizer, loss_criterion, x_dev_data, y_dev_data, local_noise_stddev, local_inside_optuna=True, name_of_model='MyModel', local_timestamp='now', local_show_plots=False, plot_every_epochs=10):
    """
    Trains the model and plots the training and development loss.

    Args:
    model_to_train: The neural network model to be trained.
    training_loader: DataLoader for the training data.
    epochs_num: Number of epochs for training.
    training_optimizer: Optimizer used for training.
    loss_criterion: Loss function.
    x_dev_data: Tensor for development/validation features.
    y_dev_data: Tensor for development/validation labels.
    name_of_model: String that represents the name of the model (decided by the user).
    plot_every_epochs: Frequency of updating the plot (default: 10 epochs).
    """

    if not local_inside_optuna:
        # Initialize the plot
        local_fig, ax = plt.subplots(figsize=(10, 4))
        if local_show_plots:
            plt.show(block=False)  # Open the plot window

        # Add horizontal grid lines
        ax.yaxis.grid(True)  # Add horizontal grid lines
        ax.set_axisbelow(True)  # Ensure grid lines are below other plot elements

        # Training loop
        training_history = {'loss_train': [], 'loss_dev': []}  # Keeps track of the losses over each epoch.

        # Initialize record low variables
        record_low_train_loss = float('inf')
        record_low_dev_loss = float('inf')

        start_time = datetime.now()

    for epoch in range(epochs_num):  # Goes from 0 to epochs_num - 1.
        for batch_x, batch_y in training_loader:  # The training_loader always delivers a new batch.
            training_optimizer.zero_grad()  # Resets the gradients of the model parameters.

            # Apply Gaussian noise to the batch (if the hyperparameter local_noise_stddev is set to 0, there is no Gaussian noise added).
            noise = torch.randn_like(batch_x) * local_noise_stddev  # noise_stddev is the standard deviation of the noise
            noisy_batch_x = batch_x + noise

            local_predictions = model_to_train(noisy_batch_x).squeeze()     # Forward pass
            loss_train = loss_criterion(local_predictions, batch_y)         # Compute the loss
            loss_train.backward()                                           # Backpropagation
            training_optimizer.step()                                       # Update weights

        with torch.no_grad():
            model_to_train.eval()
            local_predictions = model_to_train(x_dev_data).squeeze()  # Forward pass on dev set
            local_loss_dev = loss_criterion(local_predictions, y_dev_data)

        if not local_inside_optuna:                                         # When run while using Optuna, do not plot anything.
            # Record training and validation loss
            training_history['loss_train'].append(loss_train.item())
            training_history['loss_dev'].append(local_loss_dev.item())

            if (epoch + 1) % plot_every_epochs == 0 or epoch == epochs_num - 1:
                ax.clear()
                ax.plot(training_history['loss_train'], label='Train set loss')
                ax.plot(training_history['loss_dev'], label='Dev set loss')
                ax.set_title(f"{name_of_model} - Epoch: {epoch + 1}", fontsize=16, fontweight='bold')
                ax.set_xlabel('Epochs', fontsize=14)
                ax.set_ylabel('Loss', fontsize=14)
                ax.set_xlim(0, epochs_num)

                ax.yaxis.grid(True)
                ax.set_axisbelow(True)
                ax.legend()

                # Update record lows if current losses are lower
                current_train_loss = training_history['loss_train'][-1]
                current_dev_loss = training_history['loss_dev'][-1]

                if current_train_loss < record_low_train_loss:
                    record_low_train_loss = current_train_loss

                if current_dev_loss < record_low_dev_loss:
                    record_low_dev_loss = current_dev_loss

                ax.text(0.35, 0.93, f'Train-loss = {round(current_train_loss, 2)} (Lowest: {round(record_low_train_loss, 2)})\n'
                        f'Dev-loss  = {round(current_dev_loss, 2)} (Lowest: {round(record_low_dev_loss, 2)})',
                        transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

                # Add the current runtime info.
                end_time = datetime.now()
                runtime = end_time - start_time
                hours, remainder = divmod(runtime.total_seconds(), 3600)
                minutes, seconds = divmod(remainder, 60)
                runtime_text = f"Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s"

                # Plot the runtime text
                ax.text(0.35, 0.73, runtime_text, transform=ax.transAxes, fontsize=12, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.5))

                if local_show_plots:                # Only show the plots when the user decides so.
                    plt.draw()
                    plt.pause(0.1)

                if epoch == epochs_num - 1:

                    # Main directory and subdirectory for saving model .pth files
                    main_directory = '../output/nn_hyperpara_screener__output'
                    sub_directory = 'model_pth_files'

                    # Full path for saving the model includes both the main directory and subdirectory
                    save_directory = os.path.join(main_directory, sub_directory)

                    # Check if the directory exists, and if not, create it
                    if not os.path.exists(save_directory):
                        os.makedirs(save_directory)

                    # Create the full path for saving the model, including the timestamp
                    model_save_path = os.path.join(save_directory, f'{name_of_model}__{local_timestamp}.pth')

                    # Save the model's state dictionary
                    torch.save(model_to_train.state_dict(), model_save_path)

                    plt.close(local_fig)
                    return local_fig

    if local_inside_optuna:
        # Evaluate on the development data
        with torch.no_grad():
            model_to_train.eval()
            local_predictions = model_to_train(x_dev_data).squeeze()  # Forward pass on dev set
            local_loss_dev = loss_criterion(local_predictions, y_dev_data)

            # Calculate percentage error
            percentage_errors = torch.abs((local_predictions - y_dev_data) / y_dev_data) * 100

            # Handle cases where the actual value is 0 to avoid division by zero
            percentage_errors[y_dev_data == 0] = torch.abs(local_predictions - y_dev_data)[y_dev_data == 0] * 100

            # Calculate mean percentage error
            global study_mean_percentage_error  # Export by assigning to a global variable (because the return goes to the optuna module and is not easily accessible)
            study_mean_percentage_error = torch.mean(percentage_errors).item()  # Convert to Python scalar
            study_mean_percentage_error = round(study_mean_percentage_error, 2)

            return local_loss_dev


def calculate_mean_percent_error(local_predictions, local_y_dev_tensor):
    # Ensure both tensors are of the same type and device
    local_predictions = local_predictions.to(local_y_dev_tensor.device).type(local_y_dev_tensor.dtype)

    # Calculate the percent error for each element
    local_percent_error = torch.abs(local_predictions - local_y_dev_tensor) / torch.abs(local_y_dev_tensor) * 100
    # print("Absolute Differences:", absolute_differences)

    # Calculate the mean of these percent errors
    local_mean_percent_error = round(torch.mean(local_percent_error).item(), 2)
    # print("Percent Errors:", local_percent_error)

    return local_mean_percent_error


def evaluate_model(model_eval, x_dev_norm, y_dev_data, model_index, config_df):
    """
    Evaluates the trained model on the development set, calculates the mean percent error, and generates comparisons between predictions and actual labels for the first ten instances. This provides insights into the model's prediction accuracy and a sample of individual predictions.

    Args:
    model_eval: The trained neural network model to be evaluated.
    x_dev_norm: Normalized features for the development set as a NumPy array.
    y_dev_data: Actual labels for the development set as a NumPy array.
    model_index: An identifier for the model, used for indexing within the configuration DataFrame.
    config_df: A DataFrame to record the evaluation results, specifically the mean percent error.

    Returns:
    local_comparisons: A list of strings, each containing a comparison of predicted and actual values for an individual instance from the development set (limited to the first ten instances).

    The function operates in the following steps:
    1. Converts the development set features and labels into PyTorch tensors.
    2. Sets the model to evaluation mode and generates predictions for the development set.
    3. Calculates the mean percent error between predictions and actual labels and updates this in the provided DataFrame.
    4. Generates a list of comparison strings between the model's predictions and the actual labels for the first ten instances.
    """

    # Convert NumPy arrays to PyTorch tensors
    local_x_dev_tensor = torch.tensor(x_dev_norm, dtype=torch.float32)
    local_y_dev_tensor = torch.tensor(y_dev_data, dtype=torch.float32)

    # Evaluate the model
    with torch.no_grad():
        model_eval.eval()
        local_predictions = model_eval(local_x_dev_tensor).squeeze()

        # Calculate % error (for nice and intuitive reporting)
        mean_percent_error = calculate_mean_percent_error(local_predictions, local_y_dev_tensor)
        if '%err' not in config_df.columns:
            config_df['%err'] = None
        config_df.at[model_index, '%err'] = mean_percent_error

    local_comparisons = []

    for local_i in range(10):
        local_pred = local_predictions[local_i].item()
        local_rounded_pred = round_to_three_custom(local_pred)
        local_actual = local_y_dev_tensor[local_i].item()
        local_rounded_actual = round_to_three_custom(local_actual)
        local_comparison = f"Prediction: {local_rounded_pred}, Actual: {local_rounded_actual}"
        local_comparisons.append(local_comparison)

    return local_comparisons


def pandas_df_to_pdf(dataframe, local_timestamp, figure_filenames, filename_dataset, nr_features, nr_examples, local_examples_model_predictions):
    filename = 'nn_hyperpara_screener output     ' + local_timestamp + '.pdf'

    # Sort the DataFrame in descending order based on 'mean_absolute_error'
    sorted_dataframe = dataframe.sort_values(by='%err', ascending=True)

    # Create a list of expected filenames based on sorted_dataframe's 'model_name'
    expected_filenames = [f"{row['model_name']}.png" for index, row in sorted_dataframe.iterrows()]

    # Reorder figure_filenames to match the sorted order
    figure_filenames = [filename for filename in expected_filenames if filename in figure_filenames]

    # Define main directory and subdirectory
    main_directory = '../output/nn_hyperpara_screener__output'
    sub_directory = 'manual_results'

    # Create full path for the subdirectory
    sub_directory_path = os.path.join(main_directory, sub_directory)

    # Check if the subdirectory exists, if not, create it
    if not os.path.exists(sub_directory_path):
        os.makedirs(sub_directory_path)

    pdf_full_path = os.path.join(sub_directory_path, filename)

    # Create a PDF document with ReportLab
    pdf = SimpleDocTemplate(pdf_full_path, pagesize=landscape(A4), topMargin=0.5 * inch, bottomMargin=0.5 * inch)
    elements = []

    # Prepare title
    styles = getSampleStyleSheet()
    title = filename.split('.')[0]
    title_paragraph = Paragraph(title, styles['Title'])

    # Add title to elements
    elements.append(title_paragraph)

    # Add extra space after title
    elements.append(Spacer(1, 0.5 * inch))  # Increase space after title (second number, the one after the ,)

    # Add a line that states the dataset used.
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    dataset_info = f"Filename dataset: {filename_dataset}"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that states the number of features for this dataset.
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    dataset_info = f"Nr. of features: {nr_features}"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=12, spaceBefore=10, spaceAfter=10)
    dataset_info = f"Nr. of examples train set: {nr_examples}"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=13, spaceBefore=15, spaceAfter=10)
    dataset_info = f"Explanation of the short forms of the column names:"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that states the number of examples for this dataset.
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=11, spaceBefore=10, spaceAfter=10)
    dataset_info = f"alpha = learning rate, noise = amount of Gaussian noise, psi = parameter of weight initialization, %err = mean % error on dev set;"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a spacer
    elements.append(Spacer(1, 0.2 * inch))  # Add more space between title and table

    # Prepare data for the table (including column headers)
    local_data = [sorted_dataframe.columns.tolist()] + sorted_dataframe.values.tolist()

    additional_row = ['', 'Hyperparameters', '', '', '', '', '', '', '', '', '', '', '', '']  # Fewer cells than total columns
    local_data.insert(0, additional_row)

    # Create a table with the data
    table = Table(local_data)

    # Find the index of 'mean_absolute_error' column
    mae_index = sorted_dataframe.columns.tolist().index('%err')

    font_size = 8

    # Define the style for the table
    style = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), font_size),  # Fontsize for all rows of the table.

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
        ('BACKGROUND', (mae_index, 2), (mae_index, -1), colors.lightcoral),  # Column cells background color
        ('BACKGROUND', (mae_index, 1), (mae_index, 1), colors.darkred)  # Header cell background color
    ])

    # Apply the style to the table
    table.setStyle(style)

    # Add the table to the elements that will be written to the PDF
    elements.append(table)

    # Add a spacer
    elements.append(Spacer(1, 0.3 * inch))  # Add more space between value table and loss-plots.

    # Define the styles
    stylesheet = getSampleStyleSheet()

    # Get the keys in the correct order from the DataFrame
    keys_in_order = sorted_dataframe['model_name'].tolist()
    local_sorted_local_examples_model_predictions = {key: local_examples_model_predictions[key] for key in keys_in_order if key in local_examples_model_predictions}

    keys = list(local_sorted_local_examples_model_predictions.keys())

    # Number of keys to print beside each other in a row
    keys_per_row = 4

    # Initialize an index to keep track of the keys
    key_index = 0

    while key_index < len(keys):
        # Create a list to store key paragraphs for this row
        key_paragraphs = []

        # Create a list to store array paragraphs for this row
        array_paragraphs = []

        # Iterate through keys for the current row
        for local_i in range(key_index, min(key_index + keys_per_row, len(keys))):
            key = keys[local_i]

            # Add a line for the key
            key_line = Paragraph(f"<b><font size=12>{key}:</font></b><br/><br/>", stylesheet['Normal'])
            key_paragraphs.append(key_line)

            # Add a line for the value
            array_to_display = local_examples_model_predictions[key]
            array_paragraph = Paragraph("<br/>".join(array_to_display), stylesheet['Normal'])
            array_paragraphs.append(array_paragraph)

        # Create a table with key-value pairs for the current row
        kv_table = Table([key_paragraphs, array_paragraphs])

        # Optionally, add some styling to the kv table
        kv_table.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP')]))

        # Add the kv table to elements
        elements.append(kv_table)

        # Increment the key index for the next row
        key_index += keys_per_row

        # Add a spacer to separate rows
        elements.append(Spacer(1, 0.2 * inch))  # Adjust the spacing as needed

    # Assuming figure_filenames is a list of image file names
    # Process images in pairs
    for local_i in range(0, len(figure_filenames), 2):
        # Create an empty row
        row = []

        for local_j in range(2):
            index = local_i + local_j
            if index < len(figure_filenames):
                # Load and resize image
                img = Image(figure_filenames[index])
                img.drawHeight = img.drawHeight * 0.45
                img.drawWidth = img.drawWidth * 0.45
                row.append(img)
            else:
                # Add an empty cell if no image left
                row.append('')

        # Corrected line to handle string elements
        col_widths = [img.drawWidth if isinstance(img, Image) else 0 for img in row]

        # Create a table for each row
        table = Table([row], colWidths=col_widths)

        # Add the table to elements
        elements.append(table)
        elements.append(Spacer(1, 0.5 * inch))  # Space after each row

    # Build the PDF
    pdf.build(elements)

    # Optionally, delete the temporary image files
    for local_img_filename in figure_filenames:
        os.remove(local_img_filename)


def optuna_output_to_pdf(study, sublocal_mean_percent_error, sublocal_timestamp, local_cost_function):

    file_path = "../output/nn_hyperpara_screener__output/optuna_results/optuna_report__{}.pdf".format(sublocal_timestamp)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    subprocess.run(["xdotool", "key", "F5"])                    # This is needed, otherwise you risk a dir which just says its loading the files (which is permanent, it will never load)

    # Initialize PDF in A4 portrait orientation
    c = canvas.Canvas(file_path, pagesize=A4)
    width, height = A4

    # Define margin and position for the header
    left_margin = 50
    top_position = height - 50

    # Add Header on the first page
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top_position, f"NN Optuna Optimization Report  {sublocal_timestamp}")

    c.setFont("Helvetica", 12)
    c.drawString(left_margin, top_position - 30, f"Number of finished trials: {len(study.trials)}")

    c.setFont("Helvetica", 12)
    sublocal_mean_percent_error = round_to_three_custom(sublocal_mean_percent_error)
    c.drawString(left_margin, top_position - 60, f"Mean % error on the dev-set for the best hyperparameters found by the Optuna study: {sublocal_mean_percent_error}")

    # Round best trial parameters if they are floats
    rounded_params = {k: round_to_three_custom(v) for k, v in study.best_trial.params.items()}

    # Set font for the header of the best hyperparameters section
    c.setFont("Helvetica-Bold", 12)
    header_position = top_position - 100  # Adjust position for the header

    # Add a line for the header of the best hyperparameters
    c.drawString(left_margin, header_position, "Best hyperparameters found by the Optuna study:")

    # Write each parameter on a separate line
    c.setFont("Helvetica", 12)
    line_height = 20
    param_start_position = top_position - 120  # Starting position for parameters

    for local_i, (k, v) in enumerate(rounded_params.items()):
        c.drawString(left_margin, param_start_position - (local_i * line_height), f"{k} : {v}")

    param_lines = len(rounded_params)

    # Calculate position for 'Copy and paste:' line
    copy_paste_position = param_start_position - (param_lines * line_height) - line_height

    # Add a line for 'Copy and paste:' in bold
    c.setFont("Helvetica-Bold", 12)  # Set font to bold for 'Copy and paste:'
    c.drawString(left_margin, copy_paste_position, "Copy and paste:")

    # Convert to list of strings (only values)
    param_values = [str(v) for v in rounded_params.values()]

    # Remove the first element from the list
    local_num_hidden_layers = int(param_values[0])
    param_values = param_values[1:]  # Keeps all elements of the list except the first one
    param_values.insert(0, 'Optuna_model')
    param_values.append(local_cost_function)

    local_iterations = 8 + local_num_hidden_layers     # The Optimizer is at that pos, and " also need to be added there, because the params of the optimizer should be in one field in the .csv file

    for local_i in range(local_iterations):
        if local_i == 1:
            param_values[local_i] = '"' + param_values[local_i]         # Add an " before the first neuron number.
        elif local_i == local_num_hidden_layers + 1:
            param_values[local_i] = param_values[local_i] + '"'         # Add an " after the last neuron number
        elif local_i == local_iterations - 1:
            param_values[local_i] = '"' + param_values[local_i]

    param_values[-6] = param_values[-6] + '"'       # Add an " after the last parameter of the optimizer (or after the optimizer itself).

    # Join the strings without spaces after commas
    values_str = ','.join(param_values)

    # Set font and draw the string
    c.setFont("Helvetica", 6)  # Set back to regular font for the values
    c.drawString(left_margin, copy_paste_position - line_height, values_str)

    # Start figures from the second page
    c.showPage()

    # List of plot filenames
    plot_files = [
        "optimization_history.png",
        "param_importances.png",
        "edf.png"
    ]

    # Image dimensions
    img_width = 500
    img_height = 250  # Adjusted height to fit three images per page

    # Add plots to PDF, three on each page
    for local_i, plot_file in enumerate(plot_files):
        if local_i % 3 == 0 and local_i > 0:  # Create a new page after every three plots
            c.showPage()

        full_path = os.path.join(os.getcwd(), plot_file)
        if os.path.exists(full_path):
            # Calculate image Y position (top, middle, bottom image)
            img_y_position = top_position - 220 - (local_i % 3) * (img_height + 5)

            c.drawImage(full_path, left_margin, img_y_position, width=img_width, height=img_height, preserveAspectRatio=True)

            os.remove(full_path)    # Delete plot file
        else:
            print(f"Error: Plot file not found: {full_path}")
            sys.exit()

    # Save the PDF
    c.save()


def parse_optuna_hyperparameter_ranges(local_csv_file):
    """
    Parses a CSV file to extract hyperparameter ranges for Optuna optimization.

    The function reads a CSV file where each column represents a different hyperparameter.
    It processes each column to determine the type of hyperparameter (e.g., list of categorical options,
    numerical range, single value) and formats it accordingly for use in hyperparameter optimization.

    Args:
    local_csv_file: A string representing the path to the CSV file containing hyperparameter configurations.

    Returns:
    local_hyperparameters: A dictionary where keys are hyperparameter names and values are the respective
                           hyperparameter configurations (ranges, lists of options, or single values).

    The function supports:
    - Categorical hyperparameters specified as a comma-separated string.
    - Numerical ranges specified as a min-max pair, separated by a comma.
    - Single numerical or categorical values.
    - Special handling for 'batch_size' to interpret it as a single value or a list of values.
    """

    local_df = pd.read_csv(local_csv_file)
    local_hyperparameters = {}
    for local_column in local_df.columns:
        local_values = str(local_df[local_column].values[0])

        # Special handling for 'batch_size'
        if local_column == 'batch_size':
            if ',' in local_values:  # It's a list of batch sizes
                local_hyperparameters[local_column] = [int(val.strip()) for val in local_values.split(',')]
            else:  # It's a single batch size
                local_hyperparameters[local_column] = int(local_values)
        elif local_values[0].isalpha():  # It's a list of categorical options
            if ',' in local_values:
                local_hyperparameters[local_column] = local_values.split(',')
            else:
                local_hyperparameters[local_column] = local_values
        elif local_values[0].isdigit():  # It's a numerical range
            if ',' in local_values:
                local_min_val, local_max_val = map(float, local_values.split(','))
                local_hyperparameters[local_column] = (local_min_val, local_max_val)
            else:
                local_hyperparameters[local_column] = int(local_values)
    return local_hyperparameters


def run_optuna_study(local_train_dev_dataframes, local_timestamp, local_n_trials, local_input_size):

    def objective(trial, sublocal_train_dev_dataframes, sublocal_input_size):

        sublocal_hyperparameter_ranges = parse_optuna_hyperparameter_ranges('../input/nn_hyperpara_screener_optuna_ranges.csv')

        # Hyperparameters to be tuned by Optuna
        # Hyperparameter range for the number of layers
        num_layers = trial.suggest_int('num_hidden_layers', *sublocal_hyperparameter_ranges['num_hidden_layers'])

        # Dynamically creating hyperparameters for each layer's neuron count
        local_nr_neurons = []
        for local_i in range(1, num_layers + 1):         # So that the first hidden layer has the number 1 (nr. 0 could be confused with input layer)
            local_num_neurons = trial.suggest_int(f'neurons_hidden_layer_{local_i}', *sublocal_hyperparameter_ranges['nr_neurons_hidden_layers'])
        local_nr_neurons.append(local_num_neurons)
        local_nr_output_neurons = trial.suggest_int('num_neurons_output_layer', *sublocal_hyperparameter_ranges['num_neurons_output_layer'])

        local_acti_fun_type = trial.suggest_categorical('acti_fun', sublocal_hyperparameter_ranges['acti_fun_hidden_layers'])
        local_acti_fun_out_type = trial.suggest_categorical('acti_fun_out', sublocal_hyperparameter_ranges['acti_fun_output_layer'])
        local_nr_epochs = trial.suggest_int('nr_epochs', *sublocal_hyperparameter_ranges['nr_epochs'])
        local_batch_size = trial.suggest_categorical('batch_size', sublocal_hyperparameter_ranges['batch_size'])
        local_noise = trial.suggest_float('noise', *sublocal_hyperparameter_ranges['gaussian_noise'])

        # local_optimizer_column_list = hyperparameter_ranges['optimizer'].split(',')
        local_optimizer_type = trial.suggest_categorical('optimizer', sublocal_hyperparameter_ranges['optimizer'])

        # Conditional hyperparameters for the optimizer
        if local_optimizer_type == 'Adam':
            local_beta1 = trial.suggest_float('beta1', 0.5, 0.9)
            local_beta2 = trial.suggest_float('beta2', 0.999, 0.9999)
            local_optim_add_params = [local_beta1, local_beta2]
        elif local_optimizer_type == 'SGD':
            local_momentum = trial.suggest_float('momentum', 0.5, 0.9)
        else:
            print(f"Error: Optimizer name {local_optimizer_type} not valid!")
            sys.exit()

        local_learning_rate = trial.suggest_float('alpha', *sublocal_hyperparameter_ranges['alpha'], log=True)
        local_lamda = trial.suggest_float('lamda', *sublocal_hyperparameter_ranges['lamda'], log=True)
        local_dropout = trial.suggest_float('dropout', *sublocal_hyperparameter_ranges['dropout'])
        sublocal_cost_function = sublocal_hyperparameter_ranges['cost_function']
        local_psi = trial.suggest_float('psi', *sublocal_hyperparameter_ranges['psi'])

        local_hyperparams = {
            'nr_neurons_hidden_layers': local_nr_neurons,
            'nr_neurons_output_layer': local_nr_output_neurons,
            'activation_function_type': local_acti_fun_type,
            'acti_fun_out': local_acti_fun_out_type,
            'batch_size': local_batch_size,
            'optimizer_type': local_optimizer_type,
            'optim_add_params': local_optim_add_params if local_optimizer_type == 'Adam' else [local_momentum],
            'learning_rate': local_learning_rate,
            'lamda': local_lamda,
            'dropout': local_dropout,
            'psi_value': local_psi,
            'cost_function': sublocal_cost_function,
        }

        local_model, local_optimizer, local_criterion, local_train_loader, local_dev_loader, local_x_dev_tensor, local_y_dev_tensor = prepare_model_training(local_hyperparams, sublocal_train_dev_dataframes, sublocal_input_size)

        local_loss_dev = train_and_optionally_plot(local_model, local_train_loader, local_nr_epochs, local_optimizer, local_criterion, local_x_dev_tensor, local_y_dev_tensor, local_noise)

        return local_loss_dev

    global study_mean_percentage_error

    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, local_train_dev_dataframes, local_input_size), local_n_trials)

    #################################################################################
    # Generate a range of plots that Optuna has to offer
    local_fig = optuna.visualization.plot_optimization_history(study)
    local_fig.write_image("optimization_history.png")

    local_fig = optuna.visualization.plot_param_importances(study)
    local_fig.write_image("param_importances.png")

    local_fig = optuna.visualization.plot_edf(study)
    local_fig.write_image("edf.png")
    #################################################################################

    local_hyperparameter_ranges = parse_optuna_hyperparameter_ranges('../input/nn_hyperpara_screener_optuna_ranges.csv')
    local_cost_function = local_hyperparameter_ranges['cost_function']
    optuna_output_to_pdf(study, study_mean_percentage_error, local_timestamp, local_cost_function)       # Generate a report for the Optuna study.

    return study.best_trial.params


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
    study_mean_percentage_error = None
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

    # The print statement below is to separate the different terminal sections for the models.
    print("\n\n############################################################################################\n")
    print(f"Model: {model_name}\n")         # To assign potential terminal messages to a model.

    model, optimizer, criterion, train_loader, dev_loader, x_dev_tensor, y_dev_tensor = prepare_model_training(hyperparams, train_dev_dataframes, input_size)

    inside_optuna = False
    fig = train_and_optionally_plot(model, train_loader, nr_epochs, optimizer, criterion, x_dev_tensor, y_dev_tensor, noise_stddev, inside_optuna, model_name, timestamp, show_plots)
    loss_vs_epoch_figures.append(fig)

    ten_examples_model_predictions = evaluate_model(model, x_dev, y_dev, model_nr, config)
    examples_model_predictions[model_name] = ten_examples_model_predictions                             # Store the predictions in a hash for the pdf.


##################################################

figure_filenames_list = []  # To store filenames of saved figures

for i, fig in enumerate(loss_vs_epoch_figures):
    model_name = config.iloc[i]['model_name']
    img_filename = f"{model_name}.png"
    fig.savefig(img_filename, bbox_inches='tight')
    figure_filenames_list.append(img_filename)

pandas_df_to_pdf(config, timestamp, figure_filenames_list, train_set_name, input_size, amount_of_rows, examples_model_predictions)

os.system('play -nq -t alsa synth 1 sine 600')                                      # Give a notification sound when done.
