This dir contains everything used to build and employ a drag-and-drop implementation of the CatBoost algorithm, when the data is present in .csv file format. 
Simply place your .csv file with your data into the same dir, where the Bash script AutoCatBoost.sh is located, open a terminal, and run the script (bash AutoCatBoost.sh). The Bash script runs a Docker container, that uses the hyperparameter optimizer Optuna to find the best possible hyperparameters for a CatBoost model in 100 trials. The data in the .csv file is randomly split in 60% train, 20% dev, and 20% test set. The Optuna optimisation trains the model with the train set and reports the error on the dev set to Optuna, which updates the hyperparameters. At the end, the best model is evaluated on the test set. After it is done, it just outputs the .cbm file of the best model in the same dir, along with a .pdf file, which reports the accuracy of the best model and a plot with the parameter importances. To use this model for prediction, just remove the .csv file with the data from the dir, and place in your .csv file with the data where you want to run your prediction on, and run the Bash script AutoCatBoost.sh. The predictions generated by the best model found by Optuna before are appended as last column into your .csv file

The .csv files should be structured in the following way:
The .csv file with the data used for generating and evaluating the best model must only contain the feature columns followed by the column of the target labels. The first row must contain the headers for each column. Do not include a column that specifies the sample number, since CatBoost will use that as a feature (which however contains no valuable info).
The .csv file where you want to have the predictions on must only contain the feature columns, with the same structure as the .csv file with the training data but containing no target label column.


The only thing required for the Bash script to run on Linux and Max and for the .exe file to run on Windows is the availability of the Docker engine. A simple way of getting the Docker engine is through Docker desktop. To get Docker desktop, follow the guidelines on the official site:

Ubuntu:  https://docs.docker.com/desktop/install/ubuntu/                                                                                                               

Mac:     https://docs.docker.com/desktop/install/mac-install/                                                                                                                  
Windows: https://docs.docker.com/desktop/install/windows-install/

