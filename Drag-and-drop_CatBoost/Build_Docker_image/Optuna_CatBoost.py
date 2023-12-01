import sys
from catboost import CatBoostClassifier, Pool
import pandas as pd
import optuna
import matplotlib.pyplot as plt
import seaborn as sns
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from sklearn.model_selection import train_test_split


def find_best_model(local_filepath):
    # Read the data from the .csv file into a pandas DataFrame
    data = pd.read_csv(local_filepath)

    # Split the data into train, dev, and test sets with a 60-20-20 ratio
    train_data, dev_and_test_data = train_test_split(data, test_size=0.4, random_state=42)
    dev_data, test_data = train_test_split(dev_and_test_data, test_size=0.5, random_state=42)

    # Define the ranges of hyperparameters for CatBoost (you can adjust these)
    hyperparameter_ranges = {
        'depth': (4, 10),  # Depth of the tree
        'learning_rate': (0.01, 0.3),  # Learning rate
        'iterations': (20, 40),  # Number of boosting iterations
        'l2_leaf_reg': (1, 10),  # L2 regularization coefficient
        'border_count': (32, 255),  # Number of splits for numerical features
        'min_data_in_leaf': (1, 10),  # Minimum number of samples in a leaf
        'bagging_temperature': (0, 1),  # Bayesian bootstrap parameter
        'random_strength': (1, 10),  # Randomness for scoring splits
        'max_ctr_complexity': (1, 4),  # Max complexity for categorical feature combinations
        # Additional hyperparameters can be added here
    }

    # Define the Optuna objective function
    def objective(trial):
        # Sample hyperparameters from the defined ranges
        depth = trial.suggest_int('depth', *hyperparameter_ranges['depth'])
        learning_rate = trial.suggest_float('learning_rate', *hyperparameter_ranges['learning_rate'])
        iterations = trial.suggest_int('iterations', *hyperparameter_ranges['iterations'])
        l2_leaf_reg = trial.suggest_float('l2_leaf_reg', *hyperparameter_ranges['l2_leaf_reg'])
        border_count = trial.suggest_int('border_count', *hyperparameter_ranges['border_count'])
        min_data_in_leaf = trial.suggest_int('min_data_in_leaf', *hyperparameter_ranges['min_data_in_leaf'])
        bagging_temperature = trial.suggest_float('bagging_temperature', *hyperparameter_ranges['bagging_temperature'])
        random_strength = trial.suggest_float('random_strength', *hyperparameter_ranges['random_strength'])
        max_ctr_complexity = trial.suggest_int('max_ctr_complexity', *hyperparameter_ranges['max_ctr_complexity'])

        # Create and train a CatBoostClassifier with the sampled hyperparameters
        model = CatBoostClassifier(
            depth=depth,
            learning_rate=learning_rate,
            iterations=iterations,
            l2_leaf_reg=l2_leaf_reg,
            border_count=border_count,
            min_data_in_leaf=min_data_in_leaf,
            bagging_temperature=bagging_temperature,
            random_strength=random_strength,
            max_ctr_complexity=max_ctr_complexity,
            task_type='CPU',
            verbose=1  # Set verbosity to 0 to avoid console output
        )

        # Using all columns except the last as features
        features = train_data.iloc[:, :-1]

        # Using the last column as the target
        target = train_data.iloc[:, -1]

        # Fitting the model
        model.fit(features, target)

        # Using all columns except the last as features for the dev_data
        dev_features = dev_data.iloc[:, :-1]

        # Using the last column as the target for the dev_data
        dev_target = dev_data.iloc[:, -1]

        # Calculating the accuracy (higher is always better).
        local_accuracy = model.score(dev_features, dev_target)

        return local_accuracy

    # Create and run the Optuna study (100 trials)
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    # Get the best trial and its hyperparameters
    best_trial = study.best_trial
    best_hyperparameters = best_trial.params

    # Create the final model
    final_model = CatBoostClassifier(
        **best_hyperparameters,
        task_type='CPU',
        verbose=0
    )

    train_features = train_data.iloc[:, :-1]

    # Using the last column as the target
    train_target = train_data.iloc[:, -1]

    final_model.fit(train_features, train_target)

    # Save the final_model file in the output dir.
    model_file_path = '/app/output/best_catboost_model.cbm'
    final_model.save_model(model_file_path)

    test_features = test_data.iloc[:, :-1]
    test_target = test_data.iloc[:, -1]

    final_accuracy = final_model.score(test_features, test_target)
    final_accuracy_rounded = round(final_accuracy, 2)

    # Get feature importances from the final model
    feature_importances = final_model.get_feature_importance()

    # Extract feature names
    feature_names = train_features.columns

    # Create a DataFrame for plotting
    importances_df = pd.DataFrame({'Features': feature_names, 'Importance': feature_importances})

    # Sort by importance
    importances_df = importances_df.sort_values(by='Importance', ascending=False)

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.barplot(x="Importance", y="Features", data=importances_df)
    plt.title('Feature Importances of Best Model')
    plt.tight_layout()

    # Save the plot
    plt.savefig('/app/output/feature_importances_best_model.png')

    # Save the .cbm file of the best model to the current directory
    best_model_file = '/app/output/best_catboost_model.cbm'
    final_model.save_model(best_model_file)

    # Function to get image size
    def get_image_size(path):
        img = ImageReader(path)
        iw, ih = img.getSize()
        aspect = ih / float(iw)
        return aspect

    # File paths
    best_model_report = '/app/output/best_catboost_model_report.pdf'
    feature_importance_image = '/app/output/feature_importances_best_model.png'

    # Create a PDF with ReportLab
    c = canvas.Canvas(best_model_report, pagesize=letter)

    # Add a title in the content of the PDF, positioned a bit lower
    c.setFont("Helvetica-Bold", 16)
    c.drawCentredString(300, 760, "Best CatBoost Model Report")

    # Add accuracy text
    c.setFont("Helvetica", 12)
    accuracy_text = f"Accuracy of the best CatBoost model: {final_accuracy_rounded}"
    c.drawString(72, 720, accuracy_text)  # Position text on the page

    # Add explanatory text for classification
    classification_text = "For classification, an accuracy of 1 (100%) means all predictions are correct,"
    c.drawString(72, 700, classification_text)  # Position the classification explanation on the page

    classification_text = "while 0 (0%) means none are correct."
    c.drawString(72, 680, classification_text)  # Position the classification explanation on the page

    # Add explanatory text for regression
    regression_text = "For regression, the accuracy represents the R² coefficient, where 1 indicates perfect fit,"
    c.drawString(72, 660, regression_text)  # Position the regression explanation on the page

    regression_text = "and values can sometimes be negative for poor fits. A value of 0.8 means it explains"
    c.drawString(72, 640, regression_text)  # Position the regression explanation on the page

    regression_text = "80% of the variation in the data."
    c.drawString(72, 620, regression_text)  # Position the regression explanation on the page

    # Add feature importance image
    aspect = get_image_size(feature_importance_image)
    image_width = 400
    image_height = 400 * aspect
    c.drawImage(feature_importance_image, 72, 580 - image_height, width=image_width, height=image_height)

    # Save the PDF
    c.save()


def generate_predictions(local_model_file, local_data_file):
    # Load the model from the .cbm file
    local_model = CatBoostClassifier()
    local_model.load_model(local_model_file)

    # Read the data from the .csv file
    local_data = pd.read_csv(local_data_file)

    # Make predictions
    local_predictions = local_model.predict(local_data)

    # Store predictions in a DataFrame
    local_prediction_df = pd.DataFrame(local_predictions, columns=['predictions'])

    return local_prediction_df


def append_predictions_and_save(local_csv_file, local_predictions_df):
    # Load the original data from the .csv file
    local_original_data = pd.read_csv(local_csv_file)

    # Append the predictions DataFrame to the original data
    local_updated_data = pd.concat([local_original_data, local_predictions_df], axis=1)

    # Write the DataFrame to a CSV file
    output_file_path = "/app/output/predictions_unseen_data.csv"
    local_updated_data.to_csv(output_file_path, index=False)


def main():
    mode = sys.argv[1]

    if mode == "tune":
        find_best_model('/app/input/train_data.csv')
    elif mode == "predict":
        prediction_df = generate_predictions('/app/input/catboost_model.cbm', '/app/input/unseen_data.csv')
        append_predictions_and_save('/app/input/unseen_data.csv', prediction_df)
    else:
        raise ValueError("Invalid arguments. Use --train or --predict with appropriate file paths.")


if __name__ == "__main__":
    main()
