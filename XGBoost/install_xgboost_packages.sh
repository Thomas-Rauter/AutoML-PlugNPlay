#!/bin/bash

# Python packages to install
packages=(
    pandas
    xgboost
    matplotlib
    reportlab
    pygame
    numpy
    scikit-learn
    optuna
)

# Install each package using pip
for package in "${packages[@]}"
do
    echo "Installing $package..."
    pip install "$package"
done

echo "All packages installed."
