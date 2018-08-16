"""
This script provides some short functions to reduce code volume
"""

import os
import sys
import csv
import time
import librosa
import shutil
import glob
import pandas as pd
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

def ChkFile(file):
    '''
    check file exists or not
    '''
    if not os.path.isfile(file):
        print('Error! Cannot find audio file {}. \
            \nPlease inputa correct one!'.format(file))
        sys.exit(1)

def ChkDir(Dir):
    '''
    Create folder if not exists
    '''
    if not os.path.exists(Dir):
        os.mkdir(Dir)    

def GetEpisodes(array, length, episodes=None, repeat=0, Range=0):
    '''
    extract sub array according to the episodes number. 
    It equal to None means extract all. 
    '''
    matrix = []
    labels = []

    sub_length = 0
    img_number = 0
    while sub_length + length < len(array):
        sub_array = []
        for i in array[sub_length:(sub_length+length)]:
            sub_array.append(i)
        if episodes == None: # first round selection
            matrix.append(sub_array)
            labels.append(img_number)
        elif img_number in episodes: # second round check
            start = int(sub_length - Range*length + 1000)
            end = int(sub_length + (Range+1)*length + 1000)
            if start < 0:
                start = 0
            if end > len(array):
                end = len(array)
            stride = int((end - start)/(repeat+1))
            tmp_start = start
            for i in range(repeat):
                sub_array = []
                tmp_start = start + stride*i
                for i in array[tmp_start:(tmp_start+length)]:
                    sub_array.append(i)
                matrix.append(sub_array)
                labels.append(img_number)

        sub_length += length
        img_number += 1
    return matrix, labels


def decode_and_readcsv(path, List):
    '''
    create a list file with format of csv to store image and its path
    '''
    pic_path = []
    for cur_path, folders, pics in os.walk(path):
        for i in range(len(pics)):
            pics[i] = pics[i].split('.')
            pics[i][0] = int(pics[i][0])
        pics.sort()
        for i in range(len(pics)):
            pics[i][0] = str(pics[i][0])
            pics[i] = pics[i][0] + '.' + pics[i][1]
        for pic in pics:
            pic = os.path.join(cur_path, pic)
            pic_path.append(pic)

        csvFile = open(List, 'w', newline='')
        writer = csv.writer(csvFile)
        m = len(pic_path)
        for i in range(m):
            writer.writerow(np.expand_dims(pic_path[i],0))
        csvFile.close()


def CheckResult(PredLabel, Prob, number, TrueLabel, ProbThreshold):
    correct_number = 0
    for i in range(number):
        # print(PredLabel[i])
        if np.max(np.squeeze(Prob)[i]) >= ProbThreshold:
            if PredLabel[i] == TrueLabel:
                correct_number += 1

    if correct_number >= (number/2-1):
        return True
    else:
        return False

def TMPList(Dir, List):
    '''
    temp image list file for prediction
    '''
    if os.path.exists(List):
        os.remove(List)
    decode_and_readcsv(Dir, List)


def statistic(List):
    '''
    statistic frequency
    '''
    species_frequency = {}
    labels = []
    for record in List:
        content = record.split('\t')
        if int(content[1]) > 1:
            labels.append(int(content[1]))

    for i in labels:
        species_frequency[i] = species_frequency.get(i, 0) + 1
    return len(species_frequency), species_frequency


def Logs(log_time, output, type=logging.INFO):
    for handler in list(log_time.handlers):
        log_time.removeHandler(handler)

    if type == logging.INFO:
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s: %(message)s', 
            datefmt='%H:%M:%S')
    else:
        formatter = logging.Formatter('%(message)s')

    console = logging.StreamHandler(sys.stdout) # print log in the screen
    console.setFormatter(formatter)
    log_time.setLevel(type)
    loghandle = RotatingFileHandler(output, mode='a') # store log in the file
    loghandle.setLevel(type)
    loghandle.setFormatter(formatter)   

    log_time.addHandler(loghandle)
    log_time.addHandler(console)

    return log_time

def CheckAudio(audio_list):
    '''
    Check audio file and coresponding species and folder!
    '''

    audio_list = pd.read_excel(audio_list)
    IDs = []
    AudioFiles = []
    
    for index, row in audio_list.iterrows():
        # check audio file!
        if not os.path.isfile(row['Audiofile']):
            print('Error! Cannot find audio file: ', row['Audiofile'])
            sys.exit(1)            
        else:
            ## check if librosa could read audio file
            try:
                librosa.load(row['Audiofile'], sr=None)
                
            except:
                print('Error2! Cannot read audio file: ', row['Audiofile'])
                sys.exit(1)
        
            # check the ID is repeated!
            if row['ID'] in IDs:
                print('The ID {} already exists in the list! \
                      \nPlease change to a new one!'.format(row['ID']))
                sys.exit(1)
            else:
                IDs.append(row['ID'])
                AudioFiles.append((row['Audiofile'], row['ID']))
    return AudioFiles


def Species(species_list):
    '''
    extract species and its label
    '''
    species_dict = {}
    species_list = pd.read_excel(species_list)
    
    for index, row in species_list.iterrows():
        species_dict.update({row['Species']:row['Label']})
    return species_dict


def GenerateList(Dir, rate, ValidRate):
    '''
    Generate list for all the distributed images 
    '''
    image_files = []
    image_label_files = []
    image = Dir + '/*.jpg'
    image_files.extend(glob.glob(image))
    if float(rate) <= 1:
        select_number = int(float(rate)*len(image_files))
    else:
        select_number = rate
    if select_number == 0:
        print('Warning: No picture is selected in the folder ', Dir)
    else:
        np.random.shuffle(image_files)
        image_files = image_files[:select_number]

    if ValidRate > 0:
        TrainLength = int((1- ValidRate)*len(image_files))
        train_list = image_files[:TrainLength]
        valid_list = image_files[TrainLength:]

    return train_list, valid_list


def TransferImages(List, targetDir, Valid=False):
    '''
    copy images to another folder according to class label
    '''
    if Valid == True:
        targetDir = targetDir.replace('/train', '/valid')
    for file in List:
        target_file = os.path.join(targetDir, os.path.basename(file))
        shutil.copyfile(file, target_file)
