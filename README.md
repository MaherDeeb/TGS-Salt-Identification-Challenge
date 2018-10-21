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

`pip install tensorflow-gpu`

2. without GPU support:

`pip install tensorflow`

If you use python with anaconda, you can install the libraries directly using anaconda environment UI or:
<br />
`conda install tensorflow-gpu`

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

`python Full_Pipeline.py`

2. ETL.py: this script loads train and test data and prepare it by applying padding and transforming it to numpy array.

3. U_Net_layers.py, U_Net_layers_2.py, U_Net_res_layers.py: different versions of U-net models.

4. padding.py: this script applys different numpy padding so that the size of the image increases from 101 to 128

5. scoring.py: this script calculates the IoU score

6. label_data.py: this script labels the train data and filter the images based on salt contain. It only returns the images that have salt.

7. diff_data.py: it returns the first derivatives of each image. Training the model using diff_data or by adding the first derivatives to the original images increases the accuracy of the model in case of train set but test set.

8. create_new_image.py: it provides a procedure to create new images using random parts from the original images. It did not improve the model.

9. submission.py: it prepares the submission file with a date-based prefix

10. combine_models.py: this script combines the results of different models (provided in this repo *.model) together. It increases the score about 1%

## References:
https://www.anaconda.com/download/

https://www.tensorflow.org/install/

https://www.tensorflow.org/install/gpu

https://keras.io/

https://ipywidgets.readthedocs.io/en/stable/

