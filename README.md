# Structuring-ML-Project
This repository provides a template for structuring an ML project. The task of image classification is used here. It includes code for downloading the data to local machine, preprocessing of data, training and evaluation pipeline. The model can be trained via command line as well as with the use of jupyter notebook. Here we are using pytorch and pytorch lightning. VS code is used as the code editor. For reusing the repository necessary dependencies needs to be installed. Below steps can be followed irrespective of the project you do.

1. Install conda - https://conda.io/projects/conda/en/latest/user-guide/install/windows.html
2. Open anaconda prompt and create a new virtual environment: conda env create -f environment.yml.Sample yaml is provided along with the repository.
   Note: dependencies can also be provided in yaml file
3. Activate the newly created environment:
             conda activate env_name
4. Intallation of other dependencies using either one of the method listed below. Required dependencies for this projects are: python, pytorch, pytorch_lightning, 
   tensorboard, torchvision, cudatoolkit(if there is GPU in the system) 
   1. Use conda
             conda install pytorch
   2. Use pip
             pip install pytorch
   3. Run 1. pip-compile dev.in
          2. pip-sync dev.txt.
       Sample file is provided in the repository. The .txt file is generated by the first command
 
 For reusing the code: if you are using VS code, 
 1. Open the downloaded project folder in VS code
 2. Set the intrepreter as the one we created above
 3. Locate and run: python run_experiment.py (you can provide the arguments like --max_epochs=5)
 If you prefer using notebook, then use the exp.ipynb file under the folder notebooks
