# Waveman

## Introduction

*Waveman* is pipeline to identify bat species for audio file

It is based on deep learning method and is written using Python3(64-bit), which is test in Windows 10, Linux, and Mac.
Not support Python2 and Python3 32-bit since some problems ocourred when install package librosa and tensorflow.

Here, we summarize how to setup this software package and run the scripts on a small dataset, which contend 10 audio files with format of wav.

## Citation

Xing Chen, Jun Zhao, Yan-hua Chen, Wei Zhou, Alice C. Hughes. Automatic standardized processing and identification of tropical bat calls using deep learning approaches. Submitted.

## Parts 

This repo has two components: Python scripts and a small size of data to run the algorithm. 
In addition, the trained models of BatNet provided.


## Dependencies

*Waveman* depends on 
+ [Python 64-bit]

Windows: https://www.python.org/downloads/windows/

Linux: https://www.python.org/downloads/source/

Mac: https://www.python.org/downloads/mac-osx/

Remind: Don't install version 3.7 since the tensorflow package not support this version currently(until 2018/09/10).

Tensorflow update: https://pypi.org/project/tensorflow/

Setting environment variables:

Please tick the "Add Python XX to PATH" when you open Python install execute.

If forget, please open Python.exe again, tick "Add Python to envrionment variables" in Advanced Options.

The following package need install in Python

(1) librosa

(2) tensorflow (tensorflow-gpu)

(3) matplotlib

(4) pandas

(5) Pillow

(6) numpy

(7) xlrd

For example if not install package 'librosa', the system will remind:

ModuleNotFoundError: No module named 'librosa'

Solution:

    $ pip install librosa

Anaconda users can install using conda:

    $ conda install -c conda-forge librosa
    $ conda install tensorflow

Mac users could use brew to install python3 and related packages.

## Species list
The model provided support following 36 species:

Aselliscus stoliczkanus

Hipposideros armiger

Hipposideros bicolor

Hipposideros cineraceus

Hipposideros diadema

Hipposideros larvatus

Hipposideros lekaguli

Hipposideros pomona

Hipposideros turpis

Rhinolophus affinis

Rhinolophus coelophyllus

Rhinolophus lepidus

Rhinolophus malayanus

Rhinolophus pearsonii

Rhinolophus pusillus

Rhinolophus rex

Rhinolophus robinsoni

Rhinolophus siamensis

Rhinolophus sinicus

Rhinolophus stheno

Rhinolophus yunanensis

Hypsugo pulveratus

Ia io

Kerivoula hardwickii

Miniopterus magnater

Murina cineracea

Murina cyclotis

Myotis laniger

Myotis muricola

Myotis siligorensis

Phoniscus jagorii

Scotomanes ornatus

Tylonycteris robustula

Tylonyctoris pachypus

Megaderma spasma

Cheiromeles torquatus

## Getting the source code

To obtain the source code from github, let us assume you want to clone this repo into a
directory named `Waveman`:

    git clone https://github.com/chenxing3/Waveman
    cd ./Waveman

You can also retrieve the code using wget by doing the following:

    wget --no-check-certificate https://github.com/chenxing3/Waveman

## Executing the code

# Direct Prediction:

The script you will need to execute is `Prediction.py`. To see command-line 
options that need to be passed to the script, you can do the following:

    $ python Prediction.py

Here is how you can use this script

=============================================================================

python Prediction.py

usage: Prediction.py [-h] [--AudioFile AUDIOFILE] [--Model MODEL] [--SpeciesLabel SPECIESLABEL] [--Output OUTPUT] [--ImageWidth IMAGEWIDTH] [--ImageHeight IMAGEHEIGHT] [--SegmentLength SEGMENTLENGTH] [--PredLength PREDICTLENGTH]  [--Probability PROBABILITY]
 [--Repeat REPEAT] [--SecondChk SECONDCHK] [--ChkRange CHKRANGE]

optional arguments:
  -h, --help            show this help message and exit
  --AudioFile AUDIOFILE
                        (Require) Please enter audio file for prediction
  --Model MODEL         Please enter model: BatNet, VggNet
  --SpeciesLabel SPECIESLABEL
                        Please enter species label file
  --Output OUTPUT       create folder to store result and other temp files
  --ImageWidth IMAGEWIDTH
                        (optimal) image width, default is 64
  --ImageHeight IMAGEHEIGHT
                        (optimal) image height, default is 64
  --SegmentLength SEGMENTLENGTH
                        (optimal) length of segment, default is 10368
  --PredLength PREDICTLENGTH
                        (optimal) predict number each time
  --Probability PROBABILITY
                        (optimal) The prediction lower than the probability
                        will be exculde
  --Repeat REPEAT       (optimal) The prediction lower than the repeat number
                        will be exculde
  --SecondChk SECONDCHK
                        (optimal) Choose whether use second check
  --ChkRange CHKRANGE   (optimal) flack range of episodes, must > 0

## Running on test data
The test data is an audio file with format of wav (only support wav). 
All the frozen Model in directory ./model.ckpt/BatNet and named frozen_model.pb
Species and corresponding to the labels are in ./list/Species_label.xlsx
All the result in default folder ./TMP, the Result_summary.xls is the final result. 
All log is written in the file log_predict.txt in ./logs folder.

    $ python Prediction.py --AudioFile=./audio/test.wav

If specify the output folder, please configure --Output, such as:

    $ python Prediction.py --AudioFile=./audio/test.wav --Output=test
	
## Make your own data and train to generate model
There are generally 5 steps to make database and train: 

Step 1. Convert audio files to images;

Step 2. Manually assign images to specific folders, which images from each audio file could 
		classify to three categories: strong, weak and no_signal;

Step 3. Sum up all the images and make a train and valid dataset. Then combine them in a file with tfrecord format;

Step 4. Train the tfrecord file and generate model;

Step 5. Freeze a specific model.
	

### Details
Step 1. Convert audio files to images.

(1) To convert a single file:

    $ python Convert_Image.py --AudioFile=./audio/test.wav
	
If not specify a ID, it will generate a random string with 8 characters. To specify ID, the command is:
	
    $ python Convert_Image.py --AudioFile=./audio/test.wav --ID=test
	
(2) To convert many audio files, please provide a excel list with three information: audio file, ID and species (Please refer a temple in ./list/Audio_list.xlsx).

If two or more species in audio file, please separate it to multiple audio files with only one species in each file.

    $ python Convert_Image_batch.py --AudioList=./list/Audio_list.xlsx


Step 2. Manually assign images to specific folders, which images classify to three categories.

This step requires manually selection. There are four folders generated in step 1 for you to assign the images, include strong, weak, nosignal and others.

Strong folder is provided to store bat signal with relatively intact call structure; 

Weak folder is to store echo, weak, ambiguous and other uncertain images;

nosignal folder is to store image with quiet signal.


Step 3. Sum up all the images and make a train and valid dataset. Then combine them in a file with tfrecord format

    $ python Make_dataset.py --ImageList=./list/Image_folder_list.xlsx --SpeciesList=./list/Species_label.xlsx
	
The above make dataset command have two steps: 1) assgin picture and split to train and valid datasets; 2) make tfrecord file.

You can choose single step by configure --Action. The default store directory of tf file is ./dataset/tf


Step 4. Train the tfrecord file and generate model.

    $ python Training.py
	
It require tfrecord file to train and valid. Please configure --TrainNum equal to the total number of train images. Currently only provide model BatNet (until 01/09/2018).

All the model are stored in the folder ./logs


Step 5. Freeze a specific model	
	
    $ python freeze.py
	
It will choose the latest generated model in the ./logs. The frozen model will save in the ./model.ckpt with pb format.
	
Almost the log file in the logs except step1. BatNet and VggNet networks in the MODEL.py in the folder utils.

## License
Code and audio data are available for research purposes only. For any other use of the software or data, please contact the us.
