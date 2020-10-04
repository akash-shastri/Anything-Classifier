# Anything-Classifier
Have Images you want classified? This apps for you.

## About
This is a fastai based full stack GUI deep learning pipeline from dataset assembly all the way to classification. All you need to know is the classes of images you want to classify into and follow the very simple installation instructions. This app is built on 2 key libraries, Selenium to create the dataset, and Fastai to train and classify (inference). 

## Installation
You can find selenium installation at https://selenium-python.readthedocs.io/installation.html. With this you also need to install Google chrome, and chrome web driver at  https://chromedriver.chromium.org/downloads. This is relatively the easier of the 2 installations.
For fastai installation, you can find it at https://docs.fast.ai/ . However fastai is recommended to be used on linux, and has a few issues on windows. I found the easiest way to install fastai on windows is with anaconda. This is a three step process for fresh installations.
1. You need cudatoolkit. (link: https://anaconda.org/anaconda/cudatoolkit)
```conda install -c anaconda cudatoolkit```
2. You need Pytorch. (link: https://pytorch.org/). 
3. you need fastai. The conda install on the link doesn't work perfectly (at least for me), what worked for me was to clone github repo and pip install. 
```
git clone https://github.com/fastai/fastai
pip install -e "fastai[dev]"
```

## How it work 
We use selenium to create our dataset. If you already have a dataset, you can skip this step. However as of now only one format of input dataset is implemented, and its the from_folder version of fastai, where images are in subfolders each named the class name, and all in a parent folder named images.
Currently as a proof of concept, only resnet18 is implemented, but if more complex architechture is needed its very easy to change it in the source, I will work on adding other architechtures in the near future.

### Creating Data set
I use selenium to automate image downloads from google to create dataset. You can see https://github.com/akash-shastri/Chrome_image_scraper for more details.

### Training
The fastai dataloader is created from the dataset (dataset can be your own dataset, but only works in directory tree format), and transfer learning is used on a resnet18 that was originally trained on imagenet. This is enough for most simple classification tasks, and i have found great accuracy in my experiments. Other architechtures and longer training cycles can be implemented in source. 

### Classification
Classification is again done using fastai library. The model we trained during training phase is saved as a pickle file, that we read and use to load weights for inference. This makes inference very fast. Right now only single image inference is implemented, will again work on implementing multiple image inference.

## Instructions
### Creating Dataset
Follow instructions of https://github.com/akash-shastri/Chrome_image_scraper.
TLDR; 
1. Run the main.py script
2. Select one of the three options
![Any Clas main IMG](https://github.com/akash-shastri/Images/blob/main/any_clas_main_img.PNG)

2a. If you selected create dataset, a new window will appear with 4 options, 
![Any Clas create ds IMG](https://github.com/akash-shastri/Images/blob/main/any_clas_create_ds_img.PNG)
2b. Enter the path to chrome web driver in DRIVER_PATH
2c. Add the classes of images you want the dataset to be in search term (for example, dog cat cow). IMPORTANT, each class should be in a new line.
2d. Add the number of images you want in each class under number of images, I have found that 100 is a good number thats not too big and creates a model with very good accuracy.
2e. Add the target path of where you want the dataset to be created. By default, this is in "./images".

3. If you want to train, simply click the train button.

4. If you want to classify using a trained model, Click the classify button.
![Any Clas main IMG](https://github.com/akash-shastri/Images/blob/main/any_clas_clasification_img.PNG)
4a. Use the upload button to upload the image you want to classify.
4b. Use the classify button to classify.
4c. Use the clear button to clear outputs to redo the process with different image.

## To - do
1. Improve GUI
2. Add different dataset structures
3. Add different architechtures to train with and allow hyperparameter tuning.
4. Add other classification options, for example, full directory classification.

#### In the distant future,
5. Add maybe other applications like NLP and GANs etc,.

## Credits
1. https://towardsdatascience.com/image-scraping-with-python-a96feda8af2d for help with creating dataset functionality.
2. https://github.com/fastai for their amazing library and Jeremy's amazing courses on fast.ai (which are free and a must watch for anyone who wants to learn deep learning) 
