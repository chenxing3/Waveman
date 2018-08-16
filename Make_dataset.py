"""
This script is to assign pictures after manually select pictures and/or make tfrecord files
Please provide three fiels:
(1) list of fold of the assigned images; (Require species name for all folds)
(2) species and its label. (the label MUST start from 0)

"""

import os
import sys
import time
import logging
import argparse
import shutil
import pandas as pd
from utils import ops
import tensorflow as tf
from PIL import Image

def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('Make_dataset.py')

    parser.add_argument('--ImageList', dest='ImageList', type=str,
                        help='(require) Please enter audio list for prediction')
    parser.add_argument('--SpeciesList', dest='SpeciesList', type=str,
                        help='(require) species and label files')
    parser.add_argument('--OutputDir', dest='OutputDir', type=str,
                        default = path+'/dataset/train',
                        help='to store the assigned images to classes')
    parser.add_argument('--Action', dest='Action', type=str,
                        default='both', 
                        help='features: both, only_assign, only_tf')
    parser.add_argument('--StrongRate', dest='StrongRate', type=float, 
                        default=1, 
                        help='(optimal) the rate of strong signal image')
    parser.add_argument('--NoSignalRate', dest='NoSignalRate', type=float,
                        default= 1, 
                        help='(optimal) the rate of no signal image')
    parser.add_argument('--WeakRate', dest='WeakRate', type=float,
                        default= 1, 
                        help='(optimal) the rate of weak signal image')
    parser.add_argument('--ValidRate', dest='ValidRate', type=float,
                        default= 0.15, 
                        help='(optimal) the rate of weak signal image')
    parser.add_argument('--TFImageWidth', dest='TFImageWidth', type=int,
                        default= 64,
                        help='(optimal) image width, default is 64')
    parser.add_argument('--TFImageHeight', dest='TFImageHeight', type=int,
                        default= 64,
                        help='(optimal) image height, default is 64')
    args = parser.parse_args()
    
    # print help information
    if len(sys.argv) <= 2:
        parser.print_help()
        sys.exit(1)

    if args.Action not in ['both', 'only_assign', 'only_tf']:
        print('Invalid feature: ', args.Action)
        print("Please input a action: both, only_assign, or only_tf")

    if args.StrongRate > 1:
        print('\n-------------->')
        print('Error! strong image rate must be <= 1')
        sys.exit(1)

    if args.WeakRate > 1:
        print('\n-------------->')
        print('Error! weak image rate must be <= 1')
        sys.exit(1)

    if args.NoSignalRate > 1:
        print('\n-------------->')
        print('Error! no signal image rate must be <= 1')
        sys.exit(1)

    if args.ValidRate > 1:
        print('\n-------------->')
        print('Error! valid dataset rate must be <= 1')
        sys.exit(1)

    if args.TFImageWidth < 1:
        print('\n-------------->')
        print('Error! image width must be an integer that >= 1')
        sys.exit(1)

    if args.TFImageHeight < 1:
        print('\n-------------->')
        print('Error! image height must be an integer that >= 1')
        sys.exit(1)

    return args

def ChkSpecies(SpeciesDict, imagelist):
    '''
    check if all the image folder and correspond to species
    '''
    List = pd.read_excel(imagelist)
    for index, row in List.iterrows():
        if not os.path.exists(row['ImageFolder']): # check if image folders
            print('Error! Cannot find image folder: ', row['Audiofile'])
            sys.exit(1)            
        else:
            if row['Species'] not in SpeciesDict: # check species whether in list
                print('Error! Cannot find {} in species list file!'\
                      .format(row['Species']))
                print('Please add the species in the list or verify the spell!')
                sys.exit(1)

def Assign(SpeciesFolders, log, args):
    '''
    Assign strong, weak and nosignal images to the dataset folder
    '''
    List = pd.read_excel(args.ImageList)
    for index, row in List.iterrows():
        if row['Species'] in SpeciesFolders:
            # check folder 
            strong_folder = row['ImageFolder']+'/strong'
            weak_folder = row['ImageFolder']+'/weak'
            nosignal_folder = row['ImageFolder']+'/nosignal'
                
            # strong
            if os.path.exists(strong_folder):
                strong_train, strong_valid = ops.GenerateList(
                        strong_folder, args.StrongRate, args.ValidRate) # collect jpg files
                ops.TransferImages(strong_train, SpeciesFolders[row['Species']]) # copy to another folder
                if args.ValidRate > 0:
                    ops.TransferImages(strong_valid, SpeciesFolders[row['Species']], Valid=True)

            # weak
            if os.path.exists(weak_folder):
                weak_train, weak_valid = ops.GenerateList(weak_folder, args.WeakRate, args.ValidRate)
                ops.TransferImages(weak_train, SpeciesFolders['weak'])
                if args.ValidRate > 0:
                    ops.TransferImages(weak_valid, SpeciesFolders[row['Species']], Valid=True)

            # no signal
            if os.path.exists(nosignal_folder):
                nosignal_train, nosignal_valid = ops.GenerateList(nosignal_folder, 
                    args.NoSignalRate, args.ValidRate)
                ops.TransferImages(nosignal_train, SpeciesFolders['no_signal'])
                if args.ValidRate > 0:
                    ops.TransferImages(nosignal_valid, SpeciesFolders[row['Species']], Valid=True)

            log.info('Processed '+ row['ImageFolder'])

def make_dataset(validDir, args, log):
    # step 1. get the label name and species name 
    SpeciesLable = ops.Species(args.SpeciesList)

    # step 2. Check if all the audio species in the spceis list file
    log.info('Create class folders for train and valid images...')
    ChkSpecies(SpeciesLable, args.ImageList) # check 

    SpeciesDirs = {} # species and corresponding target dirs
    for species, label in SpeciesLable.items():
        target_dir = args.OutputDir+'/'+str(label)
        SpeciesDirs.update({species:target_dir})
        ops.ChkDir(target_dir) # make dir for all classes
        if args.ValidRate > 0: # if valid split require
            ops.ChkDir(validDir+'/'+str(label))

    # step 3. assigning the pictures from the manually assigned folders
    log.info('Assign images with a certain rate...')
    Assign(SpeciesDirs, log, args)

def make_tf(tranDir, validDir, tfDir, files, args, log):

    classes = []
    SpeciesLable = ops.Species(args.SpeciesList)
    for i in range(0, len(SpeciesLable)):  # 选择训练noweak时这里需要更改
        classes.append(str(i))    

    for file in files:

        writer = tf.python_io.TFRecordWriter(tfDir+'/'+file) # write

        if file == 'train.tfrecords':
            log.info('Start to make tfrecords for training...')
            pic_path = tranDir
        else:
            log.info('Start to make tfrecords for validing...')
            pic_path = validDir

        for index, name in enumerate(classes, 1):
            class_path = pic_path + '/' + name + '/'
            for img_name in os.listdir(class_path):
                img_path = class_path + img_name
                img = Image.open(img_path)
                img = img.resize((args.TFImageWidth, args.TFImageHeight))
                img_raw = img.tobytes()
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": tf.train.Feature(int64_list=tf.train.Int64List(value=[index])),
                    'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
                }))

                writer.write(example.SerializeToString())
            log.info('Processed {} class!'.format(index))
        writer.close()


def main():
    time_start = time.time()
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    args = parse_args(Path)

    # log file 
    LogFile = Path+'/logs/log_assign.txt'
    if os.path.isfile(LogFile):
        os.remove(LogFile)

    log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)
    log.info("\nThis pipeline is to assign image to class folders for train and valid.")
    log.info('Find updates in website and see README.md for more information.\n')
    log.info(' '.join(sys.argv) + '\n')

    log = ops.Logs(log, LogFile)

    ops.ChkDir(args.OutputDir)
    validDir = args.OutputDir.replace('train', 'valid')
    ops.ChkDir(validDir)

    if args.Action == 'both' or args.Action == 'only_tf':
        tfDir = args.OutputDir.replace('train', 'tf')
        if os.path.exists(tfDir):
            log.info('Create tf files in the folder: '+tfDir)
        ops.ChkDir(tfDir)
        # give tf file name for train and valid
        files = ['train.tfrecords', 'valid.tfrecords']

    # if need to assign images and make tfrecord file
    if args.Action == 'both':
        make_dataset(validDir, args, log)
        log.info('Assign done!\n')
        make_tf(args.OutputDir, validDir, tfDir, files, args, log)
        log.info('TF Done!')
    # if need to assign images only
    elif args.Action == 'only_assign':
        make_dataset(validDir, args, log)
        log.info('Assign done!\n')
    # if need make tfrecord file only
    elif args.Action == 'only_tf':
        make_tf(args.OutputDir, validDir, tfDir, files, args, log)
        log.info('TF Done!')


    log = ops.Logs(log, LogFile, logging.DEBUG)
    log.info("\nTotal cost " + str(time.time() - time_start))
    log.debug('\nPlease contract us if you find bugs! \nE-mail: chenxing@mail.xtbg.ac.cn ')
    logging.shutdown()

if __name__ == '__main__':
    main()

''' Copyright 2018 '''
