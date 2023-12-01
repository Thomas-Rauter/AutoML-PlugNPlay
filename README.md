# AutoML-PlugNPlay
The core idea of this repo is to make the generation of machine learning models on a given dataset as easy as drag-and-drop the file of the data into a dir and run a single Bash script with no command line arguments in that dir, or as simple as modifying a config file with the hyperparameters of a model. 

It is intendet that the only prerequesite of running the applications in this repo is the availability of the Docker engine (for example in the form of Docker desktop) on the local machine.

All applications are packed into Docker containers that are available over Docker-Hub, so no setting up of the environment is needed. For all Docker containers, Bash scripts (for Linux and Mac) or Powershell scripts (for Windows) are available that simplify the exectution.

Currently, this is work in progess, which means the documentation is poor and the scripts can contain errors. However, I intend to provide a very good documentation and to iteratively refine the scripts and find possibly bugs.
