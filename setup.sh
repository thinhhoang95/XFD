#!/bin/bash

# Variables
MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
MINICONDA_INSTALLER="Miniconda3-latest-Linux-x86_64.sh"
ENV_NAME="myenv"
REQUIREMENTS_FILE="requirements.txt"

# Download Miniconda installer
echo "Downloading Miniconda installer..."
wget $MINICONDA_URL -O $MINICONDA_INSTALLER

# Install Miniconda
echo "Installing Miniconda..."
bash $MINICONDA_INSTALLER -b -p $HOME/miniconda

# Initialize conda
echo "Initializing conda..."
source $HOME/miniconda/etc/profile.d/conda.sh

# Add conda initialization to .bashrc
echo "Adding conda initialization to .bashrc..."
echo -e "\n# >>> conda initialize >>>" >> ~/.bashrc
echo "source $HOME/miniconda/etc/profile.d/conda.sh" >> ~/.bashrc
echo "# <<< conda initialize <<<" >> ~/.bashrc

# Add conda-forge channel
echo "Adding conda-forge channel..."
conda config --add channels conda-forge

# Update conda
echo "Updating conda..."
conda update -y conda

# Create new environment
echo "Creating new environment: $ENV_NAME"
conda create -y -n $ENV_NAME

# Activate the new environment
echo "Activating the environment: $ENV_NAME"
conda activate $ENV_NAME

# Install packages from requirements.txt
if [ -f $REQUIREMENTS_FILE ]; then
    echo "Installing packages from $REQUIREMENTS_FILE"
    conda install --yes --file $REQUIREMENTS_FILE
else
    echo "Error: $REQUIREMENTS_FILE not found!"
    exit 1
fi

# Install additional packages with pip
echo "Installing additional packages with pip..."
pip install changepy zarr multiprocess

# Cleanup
echo "Cleaning up..."
rm $MINICONDA_INSTALLER

# Run git config user.name and user.email
echo "Configuring git..."
git config --global user.name "Thinh Hoang"
git config --global user.email "thinhhoangdinh95@hotmail.com"

echo "Setup complete. To activate the environment, use: conda activate $ENV_NAME"

