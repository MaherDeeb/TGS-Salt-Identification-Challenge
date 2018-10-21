# TGS-Salt-Identification-Challenge
Kaggle Munich Meetup team work
## Requirements:
The pipeline uses Keras framework on top of TensorFlow. Please install CUDA 9.0 and cuDNN SDK >=7.2 to use GPU (only NVIDIA® GPUs) support.
<br />
NOTE: Check if your GPU support these libraries. Besides that install the latest version of the GPU drivers by following link:
<br />
https://www.nvidia.com/Download/index.aspx?lang=en-us
<br />
To install CUDA 9.0 please follow the link:
<br />
https://developer.nvidia.com/cuda-zone
<br />
Install cuDNN SDK >=7.2 by following the link
<br />
https://developer.nvidia.com/cudnn
<br />
To install Tensorflow:
<br />
1. GPU support:
<br />
pip install tensorflow-gpu
<br />
2. without GPU support:
<br />
pip install tensorflow
<br />
If you use python with anaconda, you can install the libraries directly using anaconda environment UI or:
<br />
conda install tensorflow-gpu

Other libraries:
Numpy, matplotlib, seaborn, ipywidgets (for interactive jupyter notebook)
<br />
The train and test data has to be saved inside the following directories (folders) <br />
train imges: path_train = './data/images/' <br />
train masks: path_train_mask = './data/masks/' <br />
test images: path_test = './data/images_test/' <br />
train and test depth values: path_depth = './data/depths.csv' <br />

## The pipeline components:
1. Full_Pipeline.py: this is the main file that should be used to train the model and prepare the submission file. It calls functions saved in other scripts. Keras and and tensorFlow should be installed to the run the script.
<br />
Run the script:
<br />
python Full_Pipeline.py
<br />
2. ETL.py: This script loads train and test data and prepare it by applying padding and transforming it to numpy array.

## References:
https://www.anaconda.com/download/

https://www.tensorflow.org/install/

https://www.tensorflow.org/install/gpu

https://keras.io/

https://ipywidgets.readthedocs.io/en/stable/

