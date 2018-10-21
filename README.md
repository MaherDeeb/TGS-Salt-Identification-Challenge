# TGS-Salt-Identification-Challenge
Kaggle Munich Meetup team work
## Requirements:
The pipeline uses Keras framework on top of TensorFlow. Please install CUDA 9.0 and cuDNN SDK >=7.2 to use GPU (only NVIDIA® GPUs) support.

NOTE: Check if your GPU support these libraries. Besides that install the latest version of the GPU drivers by following link:

https://www.nvidia.com/Download/index.aspx?lang=en-us

To install CUDA 9.0 please follow the link:

https://developer.nvidia.com/cuda-zone

Install cuDNN SDK >=7.2 by following the link

https://developer.nvidia.com/cudnn

To install Tensorflow:

1. GPU support:

pip install tensorflow-gpu

2. without GPU support:

pip install tensorflow

If you use python with anaconda, you can install the libraries directly using anaconda environment UI or:

conda install tensorflow-gpu

Other libraries:
Numpy, matplotlib, seaborn, ipywidgets (for interactive jupyter notebook)

The train and test data has to be saved inside the following directories (folders)
path_train = './data/images/' \n
path_train_mask = './data/masks/' \n
path_test = './data/images_test/' \n
path_depth = './data/depths.csv' \n

## The pipeline components:
1. Full_Pipeline.py: this is the main file that should be used to train the model and prepare the submission file. It calls functions saved in other scripts. Keras and and tensorFlow should be installed to the run the script.

Run the script:

python Full_Pipeline.py

2. ETL.py: This script loads train and test data and prepare it by applying padding and transforming it to numpy array.

## References:
https://www.anaconda.com/download/

https://www.tensorflow.org/install/

https://www.tensorflow.org/install/gpu

https://keras.io/

https://ipywidgets.readthedocs.io/en/stable/

