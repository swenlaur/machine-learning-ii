# Machine Learning II

This is a repository for [Machine Learning II course](https://courses.cs.ut.ee/2019/ml-ii/spring/Main/HomePage) held in [University of Tartu](https://www.cs.ut.ee/et) 

You are welcome to use materials elsewhere provided that you:

* Do not sue me
* Do not get money out of it
* Refer to the original materials


## Setting up the environment

As the course will be held in Pyhton you have two options:

* Manage your Pyhton installation by yourself
* [Use binary package manager Anaconda for it](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

The following provides the shortest list of shell commands to set up necessary libraries. The list may change during the course.

First you have to set up a virtual environment for the course so that you do not destroy your current Pyhton distribution and later on changes in your current Python distribution do not create random incompatibilities with libraries needed for machine learning course:

```
conda create --name machine-learning
```

Next you have to install a lot of libraries to get a lift-off:

```
conda install -n machine-learning jupyter
conda install -n machine-learning pandas
conda install -n machine-learning matplotlib
conda install -n machine-learning seaborn
conda install -n machine-learning -c conda-forge plotnine
conda install -n machine-learning scipy

conda install -n machine-learning psycopg2
conda install -n machine-learning pandas-datareader

conda install -n machine-learning scikit-learn
conda install -n machine-learning tensorflow
conda install -n machine-learning keras

conda install tqdm -n machine-learning
```

First four packages set up a minimal environment for data analysis and visualisation.
The second block sets up packages needed for more complex input-output pipelines: databases and websources.
The third block contains libraries for various machine learning methods.
The fourth block contains various utility packages.

If you feel advantageous you can also install TensorFlow GPU library. In my environment I had to do it manually as Anaconda did not provide a binary package for it:

```
source activate machine-learning
pip install tensorflow-gpu
```

## Using the environment

The simples way to use it is through a command line

```
source activate machine-learning
jupyter notebook
conda deactivate
``` 
 
The first command activates `machine-learning` environment. 
The second command launches [Jupyter](https://jupyter.org) shell in your web browser window and the last command deactivates the environment after you have finished with Jupyter. If you do not want to think about it you can package it as a bash script.

## How to use Jupyter

Jupyter is a cell based computational environment for data scripting and explorative programming. It helps if you know:

* [The cheat sheet](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/) 
* [Magic commands for code completion](https://forums.fast.ai/t/jupyter-notebook-how-to-enable-intellisense/8636)  

