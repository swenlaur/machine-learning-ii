# Machine Learning II

This is a repository for [Machine Learning II course](https://courses.cs.ut.ee/2019/ml-ii/spring/Main/HomePage) held in [University of Tartu](https://www.cs.ut.ee/et)

You are welcome to use materials elsewhere provided that you:

* Do not sue me
* Do not get money out of it
* Refer to the original materials


## Setting up the environment

As the course will be held in Python you have two options:

* Manage your Python installation by yourself
* [Use binary package manager Anaconda for it](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands)

The following provides the shortest list of shell commands to set up necessary libraries.
The list may change during the course.

First, you have to set up a virtual environment for the course so that you do not destroy your current Python distribution and later on changes in your current Python distribution do not create random incompatibilities with libraries needed for machine learning course:

```
conda create --name machine-learning python=3.8
```

Next, you have to install a lot of libraries to get a lift-off:

```
# Minimal environment for data analysis
conda install -n machine-learning jupyter
conda install -n machine-learning tqdm
conda install -n machine-learning pandas
conda install -n machine-learning scipy
conda install -n machine-learning scikit-learn

# Various visualisation packages
conda install -n machine-learning matplotlib
conda install -n machine-learning -c conda-forge plotnine
conda install -n machine-learning -c conda-forge scikit-misc
conda install -n machine-learning mizani
conda install -n machine-learning seaborn
conda install -n machine-learning -c conda-forge ipyvolume

# Interfaces for databases and web resources
conda install -n machine-learning psycopg2
conda install -n machine-learning pandas-datareader

# Task specific packages
conda install -n machine-learning sympy
conda install -n machine-learning -c conda-forge editdistance
conda install -n machine-learning -c etetoolkit ete3
```

As Apple Silicon is relatively new you might have to play with `pip install` to
install some packages from other channels.
For that you must first activate the environment and only the proceed with installation.
See the discussion in the next section for further details.

conda install -n machine-learning -c conda-forge scikit-misc
fails for some unknown reason

conda install jupyter tornado==6.1

conda activate machine-learning
conda install -c conda-forge gfortran
pip install scikit-misc


pip install tornado==6.1 // tornado does not work

### Optional neural networks packages

Neural networks are not essential in this course but the following allows you to set up most relevant packages:

```
conda install -n machine-learning tensorflow
conda install -n machine-learning tensorflow-addons
conda install -n machine-learning -c conda-forge tensorflow-probability
conda install -n machine-learning keras
```

If tensorflow does not work out of the box do the following steps:

```
conda activate machine-learning
pip install --ignore-installed --upgrade tensorflow
pip install --ignore-installed --upgrade tensorflow-probability
```

If you feel advantageous you can also install TensorFlow GPU library.
In my environment I had to do it manually as Anaconda did not provide a binary package for it:

```
conda activate machine-learning
pip install tensorflow-gpu
```


If you work on Apple silicon then the pip setup is slightly different, see [MakeOptim's tutorial](https://makeoptim.com/en/deep-learning/tensorflow-metal).
For some odd reason the necessary package `tensorflow-deps` is not available in `miniconda` and you have to use a [`miniforge` fork](https://github.com/conda-forge/miniforge) instead.
This alters your `.zshrc` file and renders `miniconda` distribution unreachable by redefining `PATH` variable.
By setting path differently you can switch between distributions if you really need it.

After that you have to recreate the environment and then add tensorflow.

```
conda install -n machine-learning -c apple tensorflow-deps

conda activate machine-learning
python -m pip install tensorflow-macos
python -m pip install tensorflow-metal
conda install -n machine-learning -c conda-forge tensorflow-probability

python -m pip install tensorflow-addons
```

If `tensorflow_addons` does not install then as a last resource try installing from sources directly

```
pip install compilers
pip install -e git+https://github.com/tensorflow/addons#egg=tensorflow_addons
```


Note that setup automatically sets up GPU for evaluating and training of neural networks.
Further details can be found    




### Optional packages for Markov-Chain-Monte-Carlo methods

To be updated, see https://github.com/Gabriel-p/pythonMCMC for large list of packages


## Using the environment

The simplest way to use it is through a command line

```
conda activate machine-learning
jupyter notebook
conda deactivate
```

The first command activates `machine-learning` environment.
The second command launches [Jupyter](https://jupyter.org) shell in your web browser window and the last command deactivates the environment after you have finished with Jupyter. If you do not want to think about it you can package it as a bash script.

## How to use Jupyter

Jupyter is a cell-based computational environment for data scripting and explorative programming. It helps if you know:

* [The cheat sheet](https://www.dataquest.io/blog/jupyter-notebook-tips-tricks-shortcuts/)
* [Magic commands for code completion](https://forums.fast.ai/t/jupyter-notebook-how-to-enable-intellisense/8636)  

## How not to delete your local changes!

As this is an evolving repository, we might happen to update files modified by you during exercise sessions.
This introduces conflicts! If you work with GIT without thinking ***you might lose all your work!***
The simplest way to handle this is to rename files you have locally modified or move them out of the repository.
If you want to play GIT hero you can create your local branch and pull changes from the master.
May the force be with you!
