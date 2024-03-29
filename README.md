# Waveman

## Introduction

*Waveman* is pipeline to infer bat species only based on acoustic data.

It is based on Deep Learning method and is written using Python3(64-bit). It was tested in Windows 7/10, Linux, and Mac OS.
Not support Python2 and Python3(32-bit) since some problems ocourred when we installed the package librosa.

Here, we summarize how to setup this software package and run the scripts on a small dataset with 10 recordings.

## Citation

Xing Chen, Jun Zhao, Yan-hua Chen, Wei Zhou, Alice C. Hughes. (2020) Automatic standardized processing and identification of tropical bat calls using deep learning approaches. Biological Conservation.

## Parts 

This repo has two components: Python scripts and a small size of data to run the algorithm. 
In addition, the trained models of BatNet are provided.


## Dependencies

*Waveman* depends on 
+ [Python 64-bit]

Windows: https://www.python.org/downloads/windows/

Linux: https://www.python.org/downloads/source/

Mac: https://www.python.org/downloads/mac-osx/

Tensorflow update: https://pypi.org/project/tensorflow/

### Setting environment variables:

Please tick the "Add Python XX to PATH" when you open Python install execute.

If forget, please open Python.exe again, tick "Add Python to envrionment variables" in Advanced Options.

The following packages require to install:

(1) librosa

(2) tensorflow (tensorflow-gpu <= 1.4, 28/09/2019)

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

Mac users could use brew to install python3 and use pip install related packages.

## BatNet architecture design
![Image text](https://github.com/chenxing3/Waveman/blob/master/model.ckpt/BatNet/Bat-Figure_1.png)


## Species list
The model provided support following 36 species:

Aselliscus stoliczkanus; 
Hipposideros armiger; 
Hipposideros bicolor; 
Hipposideros cineraceus; 
Hipposideros diadema; 
Hipposideros larvatus complex; 
Hipposideros lekaguli; 
Hipposideros pomona; 
Hipposideros turpis; 
Rhinolophus affinis; 
Rhinolophus coelophyllus; 
Rhinolophus lepidus; 
Rhinolophus malayanus complex; 
Rhinolophus pearsonii; 
Rhinolophus pusillus; 
Rhinolophus rex; 
Rhinolophus robinsoni; 
Rhinolophus siamensis; 
Rhinolophus sinicus; 
Rhinolophus stheno; 
Rhinolophus yunanensis; 
Hypsugo pulveratus; 
Ia io; 
Kerivoula hardwickii; 
Miniopterus magnater; 
Murina tubinaris; 
Murina cyclotis; 
Myotis laniger; 
Myotis muricola; 
Myotis siligorensis; 
Phoniscus jagorii; 
Scotomanes ornatus; 
Tylonycteris robustula; 
Tylonycteris pachypus; 
Megaderma spasma; 
Cheiromeles torquatus; 

## Getting the source code

To obtain the source code from github, let us assume you want to clone this repo into a
directory named `Waveman`:

    git clone https://github.com/chenxing3/Waveman
    cd ./Waveman

You can also retrieve the code using wget by doing the following:

    wget --no-check-certificate https://github.com/chenxing3/Waveman

# Executing the code

## Direct Prediction:

Please execute `Prediction.py`. To see command-line options that need to be passed to the script, you can do the following:

    $ python Prediction.py

Here is how you can use this script

=============================================================================

python Prediction.py

usage: Prediction.py [-h] [--AudioFile AUDIOFILE] [--Model MODEL] [--SpeciesLabel SPECIESLABEL] [--Output OUTPUT] [--ImageWidth IMAGEWIDTH] [--ImageHeight IMAGEHEIGHT] [--SegmentLength SEGMENTLENGTH] [--PredLength PREDICTLENGTH]  [--Probability PROBABILITY]
 [--Repeat REPEAT] [--SecondChk SECONDCHK] [--ChkRange CHKRANGE]

optional arguments:

  -h, --help
  
                        show help messages and exit
  
  --AudioFile AUDIOFILE
  
                        (Require) Please enter audio file for prediction
			
  --Model MODEL
  
                        Please enter model: BatNet, VggNet
  
  --SpeciesLabel SPECIESLABEL
  
                        Please enter species label file
			
  --Output OUTPUT
  
                        Create folder to store result and other temp files
  
  --ImageWidth IMAGEWIDTH
  
                        (optimal) image width, default is 64
			
  --ImageHeight IMAGEHEIGHT
  
                        (optimal) image height, default is 64
			
  --SegmentLength SEGMENTLENGTH
  
                        (optimal) length of segment, default is 10368
			
  --PredLength PREDICTLENGTH
  
                        (optimal) predict number each time
			
  --Probability PROBABILITY
  
                        (optimal) The prediction lower than the probability will be exculde
			
  --Repeat REPEAT       
  
                        (optimal) The prediction lower than the repeat number will be exculde
  
  --SecondChk SECONDCHK
  
                        (optimal) Choose whether use second check
			
  --ChkRange CHKRANGE   
  
                        (optimal) flack range of episodes, must > 0
  
## Running on test data
The test data is an audio file with format of wav (only support wav in this version). 

All the frozen Model in the file frozen_model.pb are placed in directory ./model.ckpt/BatNet

Species and corresponding to the labels are in ./list/Species_label.xlsx

The final result in the Result_summary.xls is stored in default directory ./TMP, 

All log is written in the file log_predict.txt in ./logs.

    $ python Prediction.py --AudioFile=./audio/test.wav

If specify the output folder, please configure --Output, such as:

    $ python Prediction.py --AudioFile=./audio/test.wav --Output=test
	
## Make your own reference database and train to generate model
There are generally 5 steps to make database and train: 

Step 1. Convert audio files to images;

Step 2. Manually assign images to specific folders, which images from each audio file could 
		classify to three categories: strong, weak and no_signal;

Step 3. Sum up all the images and make train and valid datasets with tfrecord format;

Step 4. Train the tfrecord files and generate model;

Step 5. Freeze a specific model.
	

### Details
#### Step 1. Convert audio files to images.

(1) To convert a single file:

    $ python Convert_Image.py --AudioFile=./audio/test.wav
	
If not specify a ID, it will generate a random string with 8 characters. To specify ID, the command is:
	
    $ python Convert_Image.py --AudioFile=./audio/test.wav --ID=test
	
(2) To convert many audio files, please provide a excel list with three information: audio file, ID and species (Please refer a temple in ./list/Audio_list.xlsx). We will use ID as prefix for all the images.

    $ python Convert_Image_batch.py --AudioList=./list/Audio_list.xlsx

If two or more species in audio file, please split it to multiple audio files with only one species in each audio file.
 
#### Step 2. Manually assign images to specific folders, which images classify to three categories.

This step requires manually selection. There are four folders generated in step 1 for you to assign the images, including strong, weak, no_signal and others.

Strong folder is provided to store bat signal with relatively intact call structure; 

Weak folder is to store echo, weak, ambiguous and other uncertain images;

no_signal folder is to store image with quiet signal.

 
#### Step 3. Sum up all the images and make train and valid datasets with tfrecord format.

    $ python Make_dataset.py --ImageList=./list/Image_folder_list.xlsx --SpeciesList=./list/Species_label.xlsx
	
The above making dataset command have two steps: 1) assgin picture and split to train and valid datasets; 2) generate tfrecord file.

You can choose single step by configure --Action. The default store directory of tf file is ./dataset/tf
 
 
#### Step 4. Train the tfrecord file and generate model.

    $ python Training.py
	
It requires tfrecord file to train and valid. Please configure --TrainNum equal to the total number of train images. Currently only provide model BatNet (until 18/12/2018).

All the model are stored in the folder ./logs
 
 
#### Step 5. Freeze a specific model	
	
    $ python freeze.py
	
It will choose the latest generated model in the directory ./logs. The frozen model will save in the ./model.ckpt with pb format.
	
Almost all the log files are placed in the ./logs directory except step 1. 

BatNet and VggNet networks are in the MODEL.py which placed in the folder utils.

## License
Code and audio data are available for research purposes only. For any other use of the software or data, please contact the us: chenxing3753@qq.com
