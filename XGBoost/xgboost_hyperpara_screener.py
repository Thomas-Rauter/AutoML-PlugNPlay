#!/usr/bin/env python
# coding: utf-8

# Import necessary libraries
import pandas as pd
import xgboost as xgb
from xgboost.callback import TrainingCallback
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
import pygame
import os
import numpy as np
import glob
from sklearn.metrics import mean_squared_error
from copy import deepcopy
import optuna
import subprocess


#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Section, where all the functions and classes are stored. All the function calls and class instantiations are below this section.


def round_to_three_custom(num):
    """
    Modify a floating-point number to have at most three non-zero digits after the decimal point.

    Parameters:
    num (float): The number to be modified.

    Returns:
    float: The modified number with at most three non-zero digits after the decimal point.
    """

    if isinstance(num, (float, np.float64, np.float32)):                                  # To avoid errors when this is applied to a mixed list.
        num_str = str(num)
        if '.' in num_str:
            whole, decimal = num_str.split('.')                 # Splitting the number into whole and decimal parts
            non_zero_count = 0
            found_non_zero_digit_before_dot = False

            for local_i, digit in enumerate(whole):
                if digit != '0':
                    found_non_zero_digit_before_dot = True

            for local_i, digit in enumerate(decimal):           # Loop over the decimal digits (starting from the . on leftwards)
                if digit != '0':
                    non_zero_count += 1

                if non_zero_count == 3:
                    # Keeping up to the third non-zero digit and truncating the rest
                    new_decimal = decimal[:local_i + 1]
                    return float(whole + '.' + new_decimal)

                if local_i == 2 and found_non_zero_digit_before_dot:
                    new_decimal = decimal[:local_i]
                    return float(whole + '.' + new_decimal)

            return float(num_str)  # Return the original number if less than 3 non-zero digits
        else:
            return int(num_str)  # Return the original number if no decimal part
    else:
        return num


def create_config():
    filename = 'xgboost_hyperpara_screener_config.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            headers = [
                "model_name", "booster", "ntrees", "eta", "mcw",
                "mds" ,"md", "gam", "ss", "cs_tree", "cs_lev",
                "cs_node", "lam", "al", "spw", "ml", "objective",
                "eval", "seed", "nthread"
            ]
            file.write(','.join(headers))
            starter_model = ["Model_1", "gbtree", 60, 0.1, 1, 0, 6, 0, 1, 1, 1, 1, 1, 0, 1, 0, "reg:squarederror", "rmse", 0, 4]
            file.write(','.join([str(item) for item in starter_model]))


def create_optuna_ranges_file():
    filename = 'xgboost_hyperpara_screener_optuna_ranges.csv'
    if not os.path.exists(filename):
        with open(filename, 'w') as file:
            headers = [
                "booster", "n_estimators", "eta" "min_child_weight", "max_delta_step", "max_depth", "gamma",
                "subsample", "colsample_bytree", "colsample_bylevel", "colsample_bynode",
                "lambda", "alpha", "scale_pos_weight", "max_leaves", "objective", "eval_metric", "seed", "nthread"
            ]
            file.write(','.join(headers) + '\n')

            values = [
                '"gbtree,gblinear,dart"', '"1, 200"', '"0.01, 0.4"', '"1, 30"', '"0, 20"', '"1, 50"', '"0, 20"',
                '"0.01, 1"', '"0.01, 1"', '"0.01, 1"', '"0.01, 1"', '"0.01, 20"', '"0, 10"',
                '"1, 100"', '"0, 256"', 'reg:squarederror', 'rmse', '999', '16'
            ]
            file.write(','.join(values))


def parse_command_line_args(arguments):
    """
    Parse command line arguments for train_set_name, and dev_set_name, dev_set_size.

    Args:
        arguments (list): A list of command line arguments.

    Returns:
        float: The local_dev_set_size as a fraction.
        str: The local_train_set_name.
        str: The local_dev_set_name.
    """

    if len(arguments) == 1 and arguments[0] == 'info':

        print("Information about the script and its usage and supported options:")
        print("This script takes as command line arguments the filename of the train set and the filename of the dev set located in current working dir (cwd).")
        print("Additionally, it requires the file xgboost_hyperpara_screener_config.csv to be present in the cwd.")
        print("It outputs 1 pdf-file, called xgboost_hyperpara_screener_output__ followed by the current date-time.")
        print("This pdf contains the hyperparameters of the different models and their performance on the dev set, together with loss-plots.")
        print("The purpose of this script is to simplify and streamline the process of 'screening' for the best set of hyperparameters for XGBoost, on a given dataset.")

        supported_file_extensions = {
            ".csv": "Comma separated value file.",
            ".json": "Add description.",
            ".xlsx": "Add description.",
            ".parquet": "Add description.",
            ".hdf": "Add description.",
            ".h5": "Add description."
        }

        print("\nSupported file extensions for the datafiles:")
        for local_file_extension, description in supported_file_extensions.items():
            print(f"- {local_file_extension}: {description}")

        sys.exit()
    elif len(arguments) == 1 and arguments[0] == 'config':
        create_config()
        sys.exit()
    elif len(arguments) == 1 and arguments[0] == 'optuna-ranges':
        create_optuna_ranges_file()
        sys.exit()

    local_parser = argparse.ArgumentParser(description='Process the development set size and file names.')
    local_parser.add_argument('train_set_name', type=str, help='Name of the train set file')
    local_parser.add_argument('dev_set_name', type=str, help='Name of the dev set file')

    # Optional arguments
    local_parser.add_argument('--plots', action='store_true', help='Optional argument to show plots')

    local_parser.add_argument('--optuna', nargs='?', const=50, type=int, default=False, help='Optional argument to run Optuna with a specified number')

    local_args = local_parser.parse_args(arguments)

    local_train_set_name = local_args.train_set_name
    local_dev_set_name = local_args.dev_set_name
    local_show_plots = local_args.plots
    local_run_optuna = local_args.optuna is not False
    local_trials = local_args.optuna if local_args.optuna is not False else None

    return local_train_set_name, local_dev_set_name, local_show_plots, local_run_optuna, local_trials


def create_timestamp():
    now = datetime.now()                                    # Get the current time
    precise_time = now.strftime("%d-%m-%Y_%H-%M-%S")        # Format the time as "day-month-year_hour-minute-second"
    return precise_time


def read_and_process_config():
    """
    Reads a CSV file into a pandas DataFrame, replaces NaN values with
    specified default values, and enforces specific data types for each column.

    Parameters:
    file_path (str): Path to the CSV file to be read.
    default_values (dict): Dictionary where keys are column names and values are default values.
    data_types (dict): Dictionary specifying the desired data type for each column.

    Returns:
    pandas.DataFrame: The processed DataFrame with default values and specified data types.
    """

    default_values = {
        'booster': 'gbtree',  # Default booster is gradient boosted trees
        'ntrees': 60,
        'eta': 0.1,  # Default learning rate
        'mcw': 1,  # Minimum sum of instance weight (hessian) needed in a child
        'mds': 1,  # Max delta step
        'md': 6,  # Maximum depth of a tree
        'gam': 0,  # Minimum loss reduction required to make a further partition on a leaf node of the tree
        'ss': 1,  # Subsample ratio of the training instance (1 means no subsample)
        'cs_tree': 1,  # Subsample ratio of columns when constructing each tree (1 means no subsample)
        'cs_lev': 1,  # Subsample ratio of columns for each split, in each level
        'cs_node': 1,  # Subsample ratio of columns for each split
        'lam': 1,  # L2 regularization term on weights
        'al': 0,  # L1 regularization term on weights
        'spw': 1,  # Balancing of positive and negative weights
        'ml': 0,  # Max leaves
        'objective': 'reg:squarederror',  # Default objective for regression tasks
        'eval': 'rmse',  # Default evaluation metric for regression tasks
        'seed': 0,  # Random seed
        'nthread': 4,  # Number of parallel threads (set to number of cores in your machine for maximum performance)
    }

    data_types = {
        'model_name': str,
        'booster': str,
        'ntrees': int,
        'eta': float,
        'mcw': int,
        'mds': int,
        'md': int,
        'gam': float,
        'ss': float,
        'cs_tree': float,
        'cs_lev': float,
        'cs_node': float,
        'lam': float,
        'al': float,
        'spw': float,
        'ml': int,
        'objective': str,
        'eval': str,
        'seed': int,
        'nthread': int
    }

    # Read the CSV file
    local_config = pd.read_csv('xgboost_hyperpara_screener_config.csv')

    # Replace NaN values with defaults
    for column, default in default_values.items():
        if column in local_config.columns:
            local_config[column].fillna(default, inplace=True)

    # Enforce data types
    for column, dtype in data_types.items():
        if column in local_config.columns:
            local_config[column] = local_config[column].astype(dtype)

    return local_config


def check_config_columns(local_df, local_required_columns):
    def is_similar(column_a, column_b):
        # Simple function to check if two column names are similar
        return column_a.lower() == column_b.lower() or column_a in column_b or column_b in column_a

    # Identify exact missing and extra columns
    exact_missing_columns = set(local_required_columns.keys()) - set(local_df.columns)
    exact_extra_columns = set(local_df.columns) - set(local_required_columns.keys())

    # Initialize lists for similar but potentially misspelled columns
    similar_missing_columns = []
    similar_extra_columns = []

    # Check for similar columns
    for column in list(exact_missing_columns):
        for df_column in local_df.columns:
            if is_similar(column, df_column):
                similar_missing_columns.append((df_column, column))  # Note the order change here
                exact_missing_columns.remove(column)
                exact_extra_columns.discard(df_column)
                break

    for column in list(exact_extra_columns):
        for required_column in local_required_columns.keys():
            if is_similar(column, required_column):
                similar_extra_columns.append((column, required_column))  # Keeping this as it is
                exact_extra_columns.remove(column)
                exact_missing_columns.discard(required_column)
                break

    # Print the results
    if exact_missing_columns or exact_extra_columns or similar_missing_columns or similar_extra_columns:
        if exact_missing_columns:
            print("Missing columns:", ', '.join(exact_missing_columns))
        if exact_extra_columns:
            print("Extra columns:", ', '.join(exact_extra_columns))
        if similar_missing_columns:
            print("Mispelled or similar missing columns:", ', '.join([f"{x[0]} (similar to {x[1]})" for x in similar_missing_columns]))
        if similar_extra_columns:
            print("Mispelled or similar extra columns:", ', '.join([f"{x[0]} (similar to {x[1]})" for x in similar_extra_columns]))

        print("Error: DataFrame does not have the exact required columns.")
        sys.exit(1)


def check_dataframe_columns(local_df):
    """
    Checks if the given DataFrame contains a specific set of columns
    and no others. It also checks if the data in specific columns meets
    certain criteria:
    - 'nr_neurons': Must contain only ints separated by commas.
    - 'eta': Must contain only legal ints or floats.
    - 'nr_epochs' and 'batch_size': Must contain only ints.
    - 'lamda': Must contain only legal floats.
    - 'acti_fun': Must contain only letters.

    If any condition is not met, the function prints an error message
    along with the contents of the offending cell and terminates the script.

    Parameters:
    local_df (DataFrame): The DataFrame to be checked.
    """

    # Required columns and their specific checks
    required_columns_xgboost = {
        'model_name': lambda local_column: all(isinstance(local_element, str) for local_element in local_column) and len(set(local_column)) == len(local_column),
        'ntrees': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0),
        'booster': lambda local_x: pd.isna(local_x) or isinstance(local_x, str),
        'eta': lambda local_x: pd.isna(local_x) or isinstance(local_x, float),
        'mcw': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0),
        'mds': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0),
        'md': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x > 0),
        'gam': lambda local_x: pd.isna(local_x) or isinstance(local_x, float),
        'ss': lambda local_x: pd.isna(local_x) or (isinstance(local_x, float) and 0 <= local_x <= 1),
        'cs_tree': lambda local_x: pd.isna(local_x) or (isinstance(local_x, float) and 0 <= local_x <= 1),
        'cs_lev': lambda local_x: pd.isna(local_x) or (isinstance(local_x, float) and 0 <= local_x <= 1),
        'cs_node': lambda local_x: pd.isna(local_x) or (isinstance(local_x, float) and 0 <= local_x <= 1),
        'lam': lambda local_x: pd.isna(local_x) or isinstance(local_x, float),
        'al': lambda local_x: pd.isna(local_x) or isinstance(local_x, float),
        'spw': lambda local_x: pd.isna(local_x) or isinstance(local_x, float),
        'ml': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0),
        'objective': lambda local_x: pd.isna(local_x) or isinstance(local_x, str),
        'eval': lambda local_x: pd.isna(local_x) or isinstance(local_x, str),
        'seed': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0),
        'nthread': lambda local_x: pd.isna(local_x) or (isinstance(local_x, int) and local_x >= 0)
    }

    check_config_columns(local_df, required_columns_xgboost)

    # Checks if the column model_name just contains unique values (no two rows should have the same name)
    model_name_column = local_df.columns[0]

    # Find indices of rows with duplicate values in the first column
    duplicate_rows = local_df[local_df[model_name_column].duplicated(keep=False)].index

    # Print the result
    if not duplicate_rows.empty:
        print(f"Rows with duplicate values in the column model_name: {list(duplicate_rows)}")
        sys.exit()

    # Check each column for its specific criteria
    for col, check in required_columns_xgboost.items():
        if check:
            for index, value in local_df[col].items():
                if not check(value):
                    print(f"Error: Data in column '{col}', row {index}, does not meet the specified criteria. Offending data: {value}")
                    sys.exit(1)

    print("All required columns are present and data meets all specified criteria.")


def load_datasets(local_train_set_name, local_dev_set_name):
    """
    Load datasets based on file extension.

    Parameters:
    train_set_name (str): File path for the training dataset.
    dev_set_name (str): File path for the development dataset.

    Returns:
    tuple: A tuple containing two pandas DataFrames, one for the training set and one for the development set.
    """

    def load_dataset(file_name):
        # Determine the file extension
        file_ext = os.path.splitext(file_name)[1].lower()

        # Load the dataset based on the file extension
        if file_ext == '.csv':
            return pd.read_csv(file_name)
        elif file_ext == '.json':
            return pd.read_json(file_name)
        elif file_ext == '.xlsx':
            return pd.read_excel(file_name)
        elif file_ext == '.parquet':
            return pd.read_parquet(file_name)
        elif file_ext == '.hdf' or file_ext == '.h5':
            return pd.read_hdf(file_name)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

    # Load the datasets
    local_train_set_data = load_dataset(local_train_set_name)
    local_dev_set_data = load_dataset(local_dev_set_name)

    return local_train_set_data, local_dev_set_data


def get_xgboost_params(local_model_nr):
    # Define a mapping of column names to XGBoost hyperparameter names
    column_to_param = {
        'booster': 'booster',
        'eta': 'learning_rate',
        'mcw': 'min_child_weight',
        'mds': 'max_delta_step',
        'md': 'max_depth',
        'gam': 'gamma',
        'ss': 'subsample',
        'cs_tree': 'colsample_bytree',
        'cs_lev': 'colsample_bylevel',
        'cs_node': 'colsample_bynode',
        'lam': 'lambda',
        'al': 'alpha',
        'spw': 'scale_pos_weight',
        'ml': 'max_leaves',
        'objective': 'objective',
        'eval': 'eval_metric',
        'seed': 'seed',
        'nthread': 'nthread'
    }

    local_row = config.iloc[local_model_nr]  # Select a row from the DataFrame

    # Initialize an empty dictionary for XGBoost hyperparameters
    xgboost_params = {}

    # Iterate through the columns in the row and map them to XGBoost hyperparameters
    for column, value in local_row.items():
        if column in column_to_param:
            xgboost_param_name = column_to_param[column]
            xgboost_params[xgboost_param_name] = value

    return xgboost_params


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


def evaluate_xgboost_model(local_model, local_x_dev, local_y_dev):
    """
    Evaluate an XGBoost model on the development set.

    Parameters:
    - model: The trained XGBoost model.
    - x_dev: The development set features.
    - y_dev: The development set labels.

    Returns:
    - mse: Mean Squared Error on the development set.
    """

    # Make predictions on the development set
    local_y_dev_pred = local_model.predict(local_x_dev)

    # Calculate Mean Squared Error
    local_mse = mean_squared_error(local_y_dev, local_y_dev_pred)

    return local_mse


class LivePlottingCallback(TrainingCallback):
    def __init__(self, local_ax, local_fig, local_total_trees, local_model_name):
        super().__init__()
        self.ax = local_ax
        if show_plots:
            self.fig = local_fig
        self.total_trees = local_total_trees
        self.model_name = local_model_name
        self.min_train_rmse = float('inf')
        self.min_eval_rmse = float('inf')
        self.start_time = datetime.now()  # Start time is set when the object is created

    def after_iteration(self, model, iteration, evals_log):
        if iteration % 10 == 0 or iteration == self.total_trees - 1:
            self.ax.clear()
            train_rmse = evals_log['train']['rmse'][-1]
            eval_rmse = evals_log['eval']['rmse'][-1]
            self.min_train_rmse = min(self.min_train_rmse, train_rmse)
            self.min_eval_rmse = min(self.min_eval_rmse, eval_rmse)

            self.ax.plot(evals_log['train']['rmse'], label='Train RMSE')
            self.ax.plot(evals_log['eval']['rmse'], label='Validation RMSE')
            self.ax.legend()
            self.ax.set_xlabel('Number of Trees', fontsize=16)
            self.ax.set_ylabel('RMSE', fontsize=16)
            # self.ax.set_title(f'{self.model_name}: XGBoost Learning Curve, Tree {iteration + 1}', fontsize=18)
            self.ax.set_title(r'${\bf ' + self.model_name.replace("_", r"\_") + r'}$: XGBoost Learning Curve, Tree ' + str(iteration + 1), fontsize=18)


            x_current, y_current = 0.80, 0.85
            textstr_current = f'Current RMSE\nTrain: {train_rmse:.2f}\nValidation: {eval_rmse:.2f}'
            self.ax.text(x_current, y_current, textstr_current, transform=self.ax.transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Position for minimum RMSE values (adjust x_min, y_min as needed)
            x_min, y_min = 0.80, 0.73
            textstr_min = f'Minimum RMSE\nTrain: {self.min_train_rmse:.2f}\nValidation: {self.min_eval_rmse:.2f}'
            self.ax.text(x_min, y_min, textstr_min, transform=self.ax.transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

            # Runtime calculation
            end_time = datetime.now()
            runtime = end_time - self.start_time
            hours, remainder = divmod(runtime.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            runtime_text = f"Runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s"

            # Runtime display
            x_runtime, y_runtime = 0.80, 0.61
            self.ax.text(x_runtime, y_runtime, runtime_text, transform=self.ax.transAxes, fontsize=9,
                         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

            if show_plots:
                plt.pause(0.1)
        if show_plots:
            if iteration == self.total_trees - 1:
                plt.close(self.fig)  # Close the plot after the last update

        return False


def calculate_mean_percentage_error(local_model, local_model_nr, local_y_dev, local_ddev, local_config):
    """
    Calculate the mean percentage error for a given model's predictions.

    Args:
    local_model_nr (int): The row number in the config DataFrame corresponding to the model.
    local_y_dev (array): The actual values of the development set.
    local_ddev (DMatrix): The development set in DMatrix format for XGBoost predictions.
    local_config (DataFrame): The DataFrame containing model configurations and results.

    Updates the local_config DataFrame with the mean percentage error of the model in the specified row.
    """

    # Predict on the development set using the model from the specified row in config
    local_y_pred = local_model.predict(local_ddev)

    # Calculate absolute percentage error and mean percentage error
    local_percentage_errors = np.abs((local_y_dev - local_y_pred) / local_y_dev) * 100
    local_mean_percentage_error = np.mean(local_percentage_errors)

    # Update config DataFrame
    if '%err' not in local_config.columns:
        local_config['%err'] = np.nan
    local_config.at[local_model_nr, '%err'] = round(local_mean_percentage_error, 2)


def pandas_df_to_pdf(dataframe, filename, figure_filenames, filename_dataset, nr_features, nr_examples, local_examples_model_predictions):
    # Sort the DataFrame in descending order based on '%err'
    sorted_dataframe = dataframe.sort_values(by='%err', ascending=True)

    # Create a list of expected filenames based on sorted_dataframe's 'model_name'
    expected_filenames = [f"{row['model_name']}.png" for index, row in sorted_dataframe.iterrows()]

    # Reorder figure_filenames to match the sorted order
    figure_filenames = [filename for filename in expected_filenames if filename in figure_filenames]

    # Define main directory and subdirectory
    main_directory = 'xgboost_hyperpara_screener__output'
    sub_directory = 'manual_results'

    # Create full path for the subdirectory
    sub_directory_path = os.path.join(main_directory, sub_directory)

    # Check if the subdirectory exists, if not, create it
    if not os.path.exists(sub_directory_path):
        os.makedirs(sub_directory_path)
        subprocess.run(["xdotool", "key", "F5"])

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

    # Add a line that explains the short forms of the hyperparameters
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=13, spaceBefore=15, spaceAfter=10)
    dataset_info = f"Explanation of the short forms of the column names:"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that explains the short forms of the hyperparameters
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=11, spaceBefore=10, spaceAfter=10)
    dataset_info = f"mcw = min_child_weight, md = max_depth, gam = gamma, ss = subsample, cs_tree = colsample_bytree, cs_lev = colsample_bylevel,"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a line that explains the short forms of the hyperparameters
    dataset_paragraph_style = ParagraphStyle('DatasetInfo', fontSize=11, spaceBefore=10, spaceAfter=10)
    dataset_info = f"cs_node = colsample_bynode, lam = lamda, spw = scale_pos_weight, eval = eval_metric, %err = mean % error on dev set;"
    dataset_paragraph = Paragraph(dataset_info, dataset_paragraph_style)
    elements.append(dataset_paragraph)

    # Add a spacer
    elements.append(Spacer(1, 0.2 * inch))  # Add more space between title and table

    # Prepare data for the table (including column headers)
    local_data = [sorted_dataframe.columns.tolist()] + sorted_dataframe.values.tolist()

    additional_row = ['', 'Hyperparameters', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '', '']  # Fewer cells than total columns
    local_data.insert(0, additional_row)

    # Create a table with the data
    table = Table(local_data)

    # Find the index of '%err' column
    local_mape_index = sorted_dataframe.columns.tolist().index('%err')

    font_size = 8

    # Define the style for the table (but not the size)
    style = TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), font_size),          # Fontsize for all rows of the table.

        ('SPAN', (1, 0), (17, 0)),  # Span the large cell across columns 1 to 11
        ('SPAN', (18, 0), (18, 0)),  # Span for the last cell in the first row

        # Center align content in the large cell and the small cell afterward
        ('ALIGN', (1, 0), (1, 0), 'CENTER'),  # Center alignment for the large cell
        ('ALIGN', (18, 0), (18, 0), 'CENTER'),  # Center alignment for the small cell afterward

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
        ('BACKGROUND', (local_mape_index, 2), (local_mape_index, -1), colors.lightcoral),  # Column cells background color
        ('BACKGROUND', (local_mape_index, 1), (local_mape_index, 1), colors.darkred)  # Header cell background color
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

    # Process images in pairs
    for local_i in range(0, len(figure_filenames), 2):
        # Create an empty row
        row = []

        for local_j in range(2):
            index = local_i + local_j
            if index < len(figure_filenames):
                # Load and resize image
                img = Image(figure_filenames[index])
                img.drawHeight = img.drawHeight * 0.40
                img.drawWidth = img.drawWidth * 0.40
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
        elements.append(Spacer(1, 0.1 * inch))  # Space after each row

    # Build the PDF
    pdf.build(elements)

    # Optionally, delete the temporary image files
    for local_img_filename in figure_filenames:
        os.remove(local_img_filename)


def optuna_output_to_pdf(study, sublocal_mean_percent_error, sublocal_timestamp):

    file_path = "./xgboost_hyperpara_screener__output/optuna_results/optuna_report__{}.pdf".format(sublocal_timestamp)
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
    c.drawString(left_margin, top_position, f"Optuna Optimization Report  {sublocal_timestamp}")

    c.setFont("Helvetica", 12)
    c.drawString(left_margin, top_position - 30, f"Number of finished trials: {len(study.trials)}")

    c.setFont("Helvetica", 12)
    sublocal_mean_percent_error = round_to_three_custom(sublocal_mean_percent_error)
    c.drawString(left_margin, top_position - 60, f"Mean % error for the best hyperparameters found by the Optuna study: {sublocal_mean_percent_error}")

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

    # Write all parameter values in a single line for easy copy-pasting
    values_str = ', '.join([str(v) for v in rounded_params.values()])
    c.setFont("Helvetica", 12)  # Set back to regular font for the values
    c.drawString(left_margin, copy_paste_position - line_height, values_str)

    # Start figures from the second page
    c.showPage()

    # List of plot filenames
    plot_files = [
        "optimization_history.png",
        "param_importances.png",
        "contour.png",
        "parallel_coordinate.png",
        "slice_plot.png",
        "edf.png"
    ]

    # Image dimensions
    img_width = 500
    img_height = 250  # Adjusted height to fit three images per page

    # Add plots to PDF, three on each page
    for i, plot_file in enumerate(plot_files):
        if i % 3 == 0 and i > 0:  # Create a new page after every three plots
            c.showPage()

        full_path = os.path.join(os.getcwd(), plot_file)
        if os.path.exists(full_path):
            # Calculate image Y position (top, middle, bottom image)
            img_y_position = top_position - 220 - (i % 3) * (img_height + 5)

            c.drawImage(full_path, left_margin, img_y_position, width=img_width, height=img_height, preserveAspectRatio=True)
            # Delete plot file
            os.remove(full_path)
        else:
            print(f"Plot file not found: {full_path}")

    # Save the PDF
    c.save()


def parse_hyperparameters(csv_file):
    df = pd.read_csv(csv_file)
    hyperparameters = {}
    for column in df.columns:
        values = str(df[column].values[0])
        if values[0].isalpha():                                     # It's a list of categorical options
            if ',' in values:
                hyperparameters[column] = values.split(',')
            else:
                hyperparameters[column] = values
        elif values[0].isdigit():                                   # It's a numerical range
            if ',' in values:
                min_val, max_val = map(float, values.split(','))
                hyperparameters[column] = (min_val, max_val)
            else:
                hyperparameters[column] = int(values)
    return hyperparameters


def run_optuna_study(local_x_train, local_y_train, local_x_dev, local_y_dev, local_timestamp, local_n_trials):
    def objective(trial):
        hyperparameter_ranges = parse_hyperparameters('xgboost_hyperpara_screener_optuna_ranges.csv')

        # Hyperparameters to be tuned by Optuna
        param = {
            'booster': trial.suggest_categorical('booster', hyperparameter_ranges['booster']),
            'n_estimators': trial.suggest_int('n_estimators', *hyperparameter_ranges['n_estimators']),
            'eta': trial.suggest_float('eta', *hyperparameter_ranges['eta']),
            'min_child_weight': trial.suggest_int('min_child_weight', *hyperparameter_ranges['min_child_weight']),
            'max_delta_step': trial.suggest_int('max_delta_step', *hyperparameter_ranges['max_delta_step']),
            'max_depth': trial.suggest_int('max_depth', *hyperparameter_ranges['max_depth']),
            'gamma': trial.suggest_float('gamma', *hyperparameter_ranges['gamma']),
            'subsample': trial.suggest_float('subsample', *hyperparameter_ranges['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *hyperparameter_ranges['colsample_bytree']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', *hyperparameter_ranges['colsample_bylevel']),
            'colsample_bynode': trial.suggest_float('colsample_bynode', *hyperparameter_ranges['colsample_bynode']),
            'lambda': trial.suggest_float('lambda', *hyperparameter_ranges['lambda']),
            'alpha': trial.suggest_float('alpha', *hyperparameter_ranges['alpha']),
            'scale_pos_weight': trial.suggest_float('scale_pos_weight', *hyperparameter_ranges['scale_pos_weight']),
            'max_leaves': trial.suggest_int('max_leaves', *hyperparameter_ranges['max_leaves']),
            'objective': hyperparameter_ranges['objective'],
            'eval_metric': hyperparameter_ranges['eval_metric'],
            'seed': hyperparameter_ranges['seed'],
            'nthread': hyperparameter_ranges['nthread']
        }

        # Model training and evaluation
        sublocal_model = xgb.XGBRegressor(**param)
        sublocal_model.fit(local_x_train, local_y_train, eval_set=[(local_x_dev, local_y_dev)], early_stopping_rounds=10, verbose=False)
        sublocal_preds = sublocal_model.predict(local_x_dev)
        sublocal_rmse = mean_squared_error(local_y_dev, sublocal_preds, squared=False)

        return sublocal_rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, local_n_trials)

    # Output the best hyperparameters
    local_fig = optuna.visualization.plot_optimization_history(study)
    local_fig.write_image("optimization_history.png")

    local_fig = optuna.visualization.plot_param_importances(study)
    local_fig.write_image("param_importances.png")

    local_fig = optuna.visualization.plot_contour(study)
    local_fig.write_image("contour.png")

    local_fig = optuna.visualization.plot_parallel_coordinate(study)
    local_fig.write_image("parallel_coordinate.png")

    local_fig = optuna.visualization.plot_slice(study)
    local_fig.write_image("slice_plot.png")

    local_fig = optuna.visualization.plot_edf(study)
    local_fig.write_image("edf.png")

    # Retrieve the best parameters from the Optuna study
    best_params = study.best_trial.params

    # Initialize the XGBoost model with the best parameters
    sublocal_model = xgb.XGBRegressor(**best_params)

    # Fit the model on the training data
    sublocal_model.fit(local_x_train, local_y_train, eval_set=[(local_x_dev, local_y_dev)], early_stopping_rounds=10, verbose=False)

    # Make predictions on the development set
    sublocal_preds = sublocal_model.predict(local_x_dev)

    # Calculate the mean percentage error
    local_mean_percent_error = np.mean(np.abs((local_y_dev - sublocal_preds) / local_y_dev)) * 100

    optuna_output_to_pdf(study, local_mean_percent_error, local_timestamp)

    return study.best_trial.params


# End of the section where all the functions and classes are stored.
#####################################################################################################################################
#####################################################################################################################################
#####################################################################################################################################
# Section with the function calls and class instantiations and the rest of the code.

# Get the command line arguments (excluding the script name)
cmd_args = sys.argv[1:]

# Call the parse_command_line_args function with the command line arguments
train_set_name, dev_set_name, show_plots, run_optuna, nr_trials = parse_command_line_args(cmd_args)

train_set_data, dev_set_data = load_datasets(train_set_name, dev_set_name)          # Load the dataset and assign it to variables

# Assuming the last column is the output vector 'y' and the rest are features 'x'
x_train = train_set_data.iloc[:, :-1].values
y_train = train_set_data.iloc[:, -1].values

x_dev = dev_set_data.iloc[:, :-1].values
y_dev = dev_set_data.iloc[:, -1].values

timestamp = create_timestamp()

if run_optuna:
    best_trial_parameter = run_optuna_study(x_train, y_train, x_dev, y_dev, timestamp, nr_trials)
    os.system('play -nq -t alsa synth 1 sine 600')  # Give a notification sound when done.
    sys.exit()

input_size = x_train.shape[1]  # Automatically set input_size based on the number of features (number of columns)
amount_of_rows = x_train.shape[0]

config = read_and_process_config()

check_dataframe_columns(config)

nr_of_models = config.shape[0]                              # Get the number of rows = number of models to test

loss_vs_tree_figures = []                                  # In this list, all the figures of the model trainings are stored.

model_directory = 'xgboost_hyperpara_screener__output/model_files'

# Create the directory if it doesn't exist
if not os.path.exists(model_directory):
    os.makedirs(model_directory)
    subprocess.run(["xdotool", "key", "F5"])

# Remove any existing .model files in the directory
for model_file in glob.glob(os.path.join(model_directory, '*.json')):
    os.remove(model_file)

examples_model_predictions = {}                             # This hash stored all the arrays of the 10 examples of the model predictions.

##################################################
# Section where it loops over the models and trains them.
for model_nr in range(nr_of_models):
    print("\n\n############################################################################################\n")
    model_name = config.iloc[model_nr]['model_name']
    print(f"Model: {model_name}\n")

    params = get_xgboost_params(model_nr)  # Get the XGBoost hyperparameters

    dtrain = xgb.DMatrix(x_train, label=y_train)
    ddev = xgb.DMatrix(x_dev, label=y_dev)

    evals = [(dtrain, 'train'), (ddev, 'eval')]
    evals_result = {}

    # Initialize the plot
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    fig.set_size_inches(10, 6)

    # Train model with the custom callback
    ntrees = config.iloc[model_nr]['ntrees']
    start_time = create_timestamp()
    callbacks_list = []
    live_plot_callback = LivePlottingCallback(ax, fig, ntrees, model_name)
    callbacks_list.append(live_plot_callback)
    model = xgb.train(params, dtrain, num_boost_round=ntrees, evals=evals, early_stopping_rounds=10, verbose_eval=False, callbacks=callbacks_list)

    model_filename = f"{model_name}__{timestamp}.json"
    model_path = os.path.join(model_directory, model_filename)
    model.save_model(model_path)

    plt.ioff()                                                  # Turn off interactive mode

    loss_vs_tree_figures.append(deepcopy(fig))

    calculate_mean_percentage_error(model, model_nr, y_dev, ddev, config)

    predictions = model.predict(ddev)
    comparisons = []

    # Generate 10 comparisons of predictions.
    for i in range(10):
        rounded_prediction = round_to_three_custom(predictions[i])
        rounded_actual = round_to_three_custom(y_dev[i])
        comparison = f"Prediction: {rounded_prediction}, Actual: {rounded_actual}"
        comparisons.append(comparison)

    examples_model_predictions[model_name] = comparisons

figure_filenames_list = []                                      # To store filenames of saved figures

for i, fig in enumerate(loss_vs_tree_figures):
    model_name = config.iloc[i]['model_name']
    img_filename = f"{model_name}.png"
    fig.savefig(img_filename, bbox_inches='tight')
    figure_filenames_list.append(img_filename)

output_pdf_filename = 'xgboost_hyperpara_screener_output__' + timestamp + '.pdf'

pandas_df_to_pdf(config, output_pdf_filename, figure_filenames_list, train_set_name, input_size, amount_of_rows, examples_model_predictions)

os.system('play -nq -t alsa synth 1 sine 600')                                      # Give a notification sound when done.
