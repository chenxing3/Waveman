
'''
Covert audio file to image

'''

import os
import sys
import time
import shutil
import string
import logging
import random
import librosa
import argparse
from utils import img_utils, ops


def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('Convert_Image_batch.py')
    parser.add_argument('--AudioList', dest='AudioList', type=str,
                        help='Please enter audio file for prediction')
    parser.add_argument('--ImgDir', dest='ImgDir', type=str,
                        default = path+'/image',
                        help='image and other temp files')
    parser.add_argument('--ImageWidth', dest='ImageWidth', type=int,
                        default= 256,
                        help='(optimal) image width, default is 64')
    parser.add_argument('--ImageHeight', dest='ImageHeight', type=int,
                        default= 256,
                        help='(optimal) image height, default is 64')
    parser.add_argument('--SegmentLength', dest='SegmentLength', type=int,
                        default= 10368, 
                        help='(optimal) length of segment, default is 10368')
    args = parser.parse_args()
    
    # print help information
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return args

def main():
    time_start = time.time()
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    args = parse_args(Path.replace('\\','/'))

    # check audio file in list
    AudioFiles = ops.CheckAudio(args.AudioList)

    if args.ImageWidth < 1:
        print('\n-------------->')
        print('Error! image width must be an integer that >= 1')
        sys.exit(1)

    if args.ImageHeight < 1:
        print('\n-------------->')
        print('Error! image height must be an integer that >= 1')
        sys.exit(1)

    if args.SegmentLength < 1:
        print('\n-------------->')
        print('Error! Audio unit length must be an integer that >= 1')
        sys.exit(1)

    # log file 
    LogFile = Path+'/logs/log_Audio_to_IMG_batch.txt'
    if os.path.isfile(LogFile):
        os.remove(LogFile)

    log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)
    log.info('In total of {} files will to convert!'.format(1))

    for index_list, (AudioFile, ID) in enumerate(AudioFiles):

        signal, SampleRate = librosa.load(AudioFile, sr=None) # read audio file

        Dir = args.ImgDir+'/'+ID
			
        ops.ChkDir(Dir)

        log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)
        log.info('\n------------->')
        log.info('Convert No. {} file {} to images.\n'.format(index_list+1, AudioFile))
        log.info(' '.join(sys.argv) + '\n')
        log.info('The images store in the folder: {}\n'.format(Dir))

        log = ops.Logs(log, LogFile, logging.INFO)
        log.info('Creat folders for assign images!')
        # prepare folder for manual assigning images
        ops.ChkDir(Dir+'/nosignal') # store no signal episodes
        ops.ChkDir(Dir+'/weak') # store weak signal episodes
        ops.ChkDir(Dir+'/strong') # store weak strong episodes
        ops.ChkDir(Dir+'/other') # store weak echo or other episodes
        ops.ChkDir(Dir+'/raw') # backup audio files

        # Backup the audio file
        BackupFile = Dir+'/raw/'+os.path.basename(AudioFile)
        if not os.path.isfile(BackupFile):
            shutil.copyfile(AudioFile, BackupFile)

        # step 1. split array
        log.info('Split array!')
        signal_matrix, indexes = ops.GetEpisodes(signal, args.SegmentLength) 

        # step 2. convert signal array to images
        log.info('Convert signal array to {} images'.format(len(indexes)))
        image_files = []
        for i in range(len(indexes)):
            TargetImg = Dir+'/'+ID+'-'+str(i).zfill(5)+'.jpg' # conver array to image
            img_utils.ConvJPG(TargetImg, signal_matrix[i], SampleRate, 
                args.ImageWidth, args.ImageHeight, train=True) 
            image_files.append(TargetImg)

            if i % (int(len(indexes)/10)+1) == 0:
                log.info('Genereted {} images'.format(str(i)))
            elif i == (len(indexes)-1):
                log.info('Done!')

    log = ops.Logs(logging.getLogger(), LogFile, logging.DEBUG)
    log.info('\n============>')
    log.info("Total cost " + str(time.time() - time_start))
    log.debug('\nPlease contract us if you find bugs! \nE-mail: chenxing@mail.xtbg.ac.cn ')

if __name__ == '__main__':
        main()

''' Copyright 2018 '''
