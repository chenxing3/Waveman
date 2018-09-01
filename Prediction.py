'''
This pipeline is for predition. Please prepare 3 files: 
(1) audio file;
(2) frozen model;
(3) species list with label;
'''

import os
import sys
import time
import shutil
import librosa
import logging
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import img_utils, ops

def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('Prediction.py')
    parser.add_argument('--AudioFile', dest='AudioFile', type=str,
                        help='(Require) Please enter audio file for prediction')
    parser.add_argument('--Model', dest='Model', type=str,
                        default = 'BatNet',
                        help='Please enter model: BatNet, VggNet')
    parser.add_argument('--SpeciesLabel', dest='SpeciesLabel', type=str,
                        default = path+'/list/Species_label.xlsx',
                        help='Please enter species label file')
    parser.add_argument('--Output', dest='output', type=str,
                        default = path+'/TMP',
                        help='create folder to store result and other temp files')
    parser.add_argument('--ImageWidth', dest='ImageWidth', type=int,
                        default= 64,
                        help='(optimal) image width, default is 64')
    parser.add_argument('--ImageHeight', dest='ImageHeight', type=int,
                        default= 64,
                        help='(optimal) image height, default is 64')
    parser.add_argument('--SegmentLength', dest='SegmentLength', type=int,
                        default= 10368, 
                        help='(optimal) length of segment, default is 10368')
    parser.add_argument('--PredLength', dest='PredictLength', type=int,
                        default= 1000, 
                        help='(optimal) predict number each time')
    parser.add_argument('--Probability', dest='Probability', type=float,
                        default= 0.8, 
                        help='(optimal) The prediction lower than the \
                                probability will be exculde')
    parser.add_argument('--Repeat', dest='Repeat', type=int,
                        default= 7, 
                        help='(optimal) The prediction lower than the \
                                repeat number will be exculde')
    parser.add_argument('--SecondChk', dest='SecondChk', type=bool,
                        default= True, 
                        help='(optimal) Choose whether use second check')
    parser.add_argument('--ChkRange', dest='ChkRange', type=float,
                        default = 1.0, 
                        help='(optimal) flack range of episodes, must > 0')
    args = parser.parse_args()
    
    # print  help information    
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    if args.ImageWidth < 1:
        print('\n-------------->')
        print ('Error! image width must be an integer that >= 1')
        sys.exit(1)

    if args.ImageHeight < 1:
        print ('\n-------------->')
        print ('Error! image height must be an integer that >= 1')
        sys.exit(1)

    if args.SegmentLength < 1:
        print ('\n-------------->')
        print ('Error! Audio unit length must be an integer that >= 1')
        sys.exit(1)

    if args.Probability > 1:
        print ('\n-------------->')
        print ('Error! Probability must be <= 1')
        sys.exit(1)

    if args.Repeat < 1:
        print ('\n-------------->')
        print ('Error! Repeat number must be an integer that >= 1')
        sys.exit(1)

    if args.ChkRange <= 0:
        print ('\n-------------->')
        print ('Error! Flank range must be an integer that > 0')
        sys.exit(1)

    return args

def load_graph(frozen_graph_filename):
    '''
    read parameters from the PB file(model)
    '''
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            producer_op_list=None
        )
    return graph

def predict(model, SpeciesDict, ImageList, n):
    '''
    predict
    '''
    graph = load_graph(model)
    x = graph.get_tensor_by_name('prefix/Placeholder:0')
    y = graph.get_tensor_by_name('prefix/output_node:0')
    z = graph.get_tensor_by_name('prefix/training:0')    

    num, pics = img_utils.read_and_decode_single_example(ImageList)
    
    with tf.Session(graph=graph) as sess:
        a=0
        res=[]
        ProbMatrix = np.zeros((0,len(SpeciesDict)), dtype=None)
        if num <= n:
            y_out = sess.run(y, feed_dict={
                x: pics,
                z: 0
            })
            res = np.argmax(y_out, 1)
            res.astype(np.int8)

            ProbMatrix = sess.run(tf.nn.softmax(y_out))

        elif num % n == 0:
            for i in range(0, num // n):
                y_out = sess.run(y, feed_dict={
                    x: pics[a:a + n],
                    z: 0
                })
                res2 = np.argmax(y_out, 1)
                res = np.append(res, res2, axis=0)
                ProbMatrix = np.append(ProbMatrix, 
                    sess.run(tf.nn.softmax(y_out)),axis=0)
                a = a+n

        elif num % n != 0:
            for i in range(0, num // n):
                y_out = sess.run(y, feed_dict={
                    x: pics[a:a + n],
                    z: 0
                })
                res2 = np.argmax(y_out, 1)
                res = np.append(res, res2, axis=0)
                ProbMatrix  = np.append(ProbMatrix, 
                    sess.run(tf.nn.softmax(y_out)),axis=0)
                a = a+n

            y_out = sess.run(y, feed_dict={
                x: pics[a:],
                z: 0
            })
            res2 = np.argmax(y_out, 1)
            res = np.append(res, res2, axis=0)
            res.astype(np.int8)
            ProbMatrix = np.append(ProbMatrix, 
                sess.run(tf.nn.softmax(y_out)), axis=0)
        return num, res, ProbMatrix

def SecondChkBlock(Block, signal, sr, model, SpeciesDict, ImgDir2, args):
    '''
    process recheck episodes block by block
    '''
    res = []
    prob = []
    image_list2 = args.output+'/predic_pic2.csv'
    signal_matrix, labels = ops.GetEpisodes(signal, args.SegmentLength, 
            Block, args.Repeat, args.ChkRange)
    image_files = []
    for i in range(len(labels)):
        ResultFile2 = ImgDir2+'/'+str(i)+'.jpg' # conver array to image
        img_utils.ConvJPG(ResultFile2, signal_matrix[i], sr, 
            args.ImageWidth, args.ImageHeight) 
        image_files.append(ResultFile2)

    # tmp file to store image path
    ops.TMPList(ImgDir2, image_list2)
    
    # prediction
    _, res, prob = predict(model, SpeciesDict, image_list2, args.PredictLength)

    # delete images 
    for i in image_files:
        os.remove(i)
    return res, prob

def main():
    time_start = time.time()
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    Path = Path.replace('\\', '/')
    print (Path)
    args = parse_args(Path)

    # Creat folders 
    if os.path.exists(args.output):
        shutil.rmtree(args.output) # delete exists work file
    ops.ChkDir(args.output) # create work dir
    ImgDir = args.output+'/img' # temp image dir
    ImgDir2 = args.output+'/img_round2'  # temp image dir 2
    ops.ChkDir(ImgDir)
    ops.ChkDir(ImgDir2)

    # write log file
    LogFile = args.output+'/log_predict.txt'
    print ('LogFile: ', LogFile)
    if os.path.isfile(LogFile):
        os.remove(LogFile)

    log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)
    log.info("\n===============================================================")
    log.info("This pipeline is to predict bat for audio file.")
    log.info('Find updates in website and see README.md for more information.\n')
    log.info(' '.join(sys.argv) + '\n')

    # Step 1. Check files and folders, read files
    model = Path+'/model.ckpt/'+args.Model+'/frozen_model.pb' # model

    ops.ChkFile(args.AudioFile)
    ops.ChkFile(model)
    ops.ChkFile(args.SpeciesLabel)

    df = pd.read_excel(args.SpeciesLabel, skiprows=[0], header=None) # read species and label
    SpeciesDict = df[0]

    # sys.exit(1)
    try:
        signal, SampleRate = librosa.load(args.AudioFile, sr=None) # read audio file
    except:
        print ('WARNING: error in wav file')
        sys.exit(1)

    # Step 2. Convert audio to image
    log = ops.Logs(log, LogFile)
    log.info('First round prediction ...')
    log.info('Convert audio to image ... Please Waits for a while')

    # get signal episode array
    signal_matrix, labels = ops.GetEpisodes(signal, args.SegmentLength) 
    for i in range(len(labels)):
        ResultFile = ImgDir+'/'+str(labels[i])+'.jpg' # conver array to image
        img_utils.ConvJPG(ResultFile, signal_matrix[i], SampleRate, 
            args.ImageWidth, args.ImageHeight) 

    # step 3. predict
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # GPU first, otherwise use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # only print  error information!

    ## create file for storing images path
    image_list = args.output+'/predic_pic.csv'
    ops.TMPList(ImgDir, image_list)

    ## predict
    NumberOfImage, PredictLabel, Prob = predict(model, SpeciesDict, 
        image_list, args.PredictLength)
    
    ## first round result
    FirstRound = args.output+'/result_first_round.txt'
    ResultHandle = open(FirstRound, 'w')

    count = 0 # to count bat present times in first round prediction
    for i in range(NumberOfImage):
        ResultHandle.write("%d\t%d\t%f\n"%(i, PredictLabel[i], np.max(np.squeeze(Prob)[i])))
        # statistic the result 
        if np.max(np.squeeze(Prob)[i]) > args.Probability:
            if PredictLabel[i] > 1:
                count += 1

    ResultHandle.close()

    log.info('First round result finished!\n')

    # step 4. Second round result check
    log.info('Second round: Checking ...')

    final_matrix = []
    SecondChkPool = []

    for record in open(FirstRound, 'r'):
        index, label, prob = record.strip('\n').split('\t')
        if int(label) <= 1: # pass weak and blank class 
            final_matrix.append(index+'\t'+label+'\t'+prob)
        else:
            if float(prob) < args.Probability: # pass the low probability prediction
                final_matrix.append(index+'\t1\t'+prob+'\t'+label)
            elif args.SecondChk == False: # false means pass the second check
                final_matrix.append(index+'\t'+label+'\t'+prob)
            else:
                ## Store all the episodes require second check
                SecondChkPool.append((index, label, prob))

    log.info('Total {} images need to check ...'.format(str(count)))

    MaxValue = int(args.PredictLength/(2*args.Repeat)) ## need first get the max value for each block
    count_second = 0 
    # recheck bat signal units block by block
    TmpLabelBlock = []
    TmpIndexBlock = []
    TmpProbBlock = []

    for count, (index, label, prob) in enumerate(SecondChkPool):
        TmpLabelBlock.append(label)
        TmpIndexBlock.append(int(index))
        TmpProbBlock.append(prob)

        if count != 0 and count % MaxValue == 0:
            PredictLabel, Prob = SecondChkBlock(TmpIndexBlock, signal, SampleRate, 
                model, SpeciesDict, ImgDir2, args)

            for i in range(len(TmpIndexBlock)):
                start = i*args.Repeat
                end = (i+1)*args.Repeat
                ChkRes = ops.CheckResult(PredictLabel[start:end], Prob[start:end], 
                    args.Repeat, int(TmpLabelBlock[i]), args.Probability)                    

                if ChkRes == True: # true means pass the check
                    final_matrix.append(str(TmpIndexBlock[i])+'\t'+TmpLabelBlock[i]\
                        +'\t'+TmpProbBlock[i])
                else: # false means not pas the check
                    final_matrix.append(str(TmpIndexBlock[i])+'\t1\t'+\
                        TmpProbBlock[i]+'\t'+TmpLabelBlock[i])
           
            TmpLabelBlock = []
            TmpIndexBlock = []
            TmpProbBlock = []

            log.info('Processed %.2f%%' % (count/len(SecondChkPool)*100)) # log the process
            
        elif count == (len(SecondChkPool)-1): # to check the remain episodes
            PredictLabel, Prob = SecondChkBlock(TmpIndexBlock, signal, SampleRate, 
                model, SpeciesDict, ImgDir2, args)

            for i in range(len(TmpIndexBlock)):
                start = i*args.Repeat
                end = (i+1)*args.Repeat
                ChkRes = ops.CheckResult(PredictLabel[start:end], Prob[start:end], 
                    args.Repeat, int(TmpLabelBlock[i]), args.Probability)                    

                if ChkRes == True: # true means pass the check
                    final_matrix.append(str(TmpIndexBlock[i])+'\t'+TmpLabelBlock[i]\
                        +'\t'+TmpProbBlock[i])
                else: # false means not pas the check
                    final_matrix.append(str(TmpIndexBlock[i])+'\t1\t'+\
                        TmpProbBlock[i]+'\t'+TmpLabelBlock[i])
            
            log.info('Processed 100%') # log the process
    log.info('Second round done!\n') # log the process

    # step 5. summary the result
    SecondRound = args.output+'/Result_summary.xls'
    ResultHandle2 = open(SecondRound, 'w')

    UnitTimeDuration = 1/SampleRate*args.SegmentLength # time units according to sample rate
    TotalNumber, FreqDict = ops.statistic(final_matrix)
    FreqDictSort = sorted(FreqDict.items(), key=lambda x:x[1], reverse=True) # sort

    ResultHandle2.write('# Result Summary\n')
    ResultHandle2.write('# Total of %d species detected in the audio file.\n' % (TotalNumber))

    ResultHandle2.write('Species\tOccurrence Number\n') # write title
    for index, (sp, PresentTimes) in enumerate(FreqDictSort):
        ResultHandle2.write('{}\t{}\n'.format(SpeciesDict[sp], str(PresentTimes)))

    ResultHandle2.write('\n# Detail bats present\n')
    ResultHandle2.write('Time(s)\tSpecies\tProbability\tFirst_round\n') # write title

    # write result
    result_dict = {}
    for record in final_matrix:
        i = record.split('\t')
        if len(i) == 4: 
            content = '%.2f\t%s\t%.2f\t%s' % ((int(i[0])+1)*UnitTimeDuration, # time calculate
                SpeciesDict[int(i[1])], float(i[2]), SpeciesDict[int(i[3])])
            result_dict.update({content: (int(i[0])+1)*UnitTimeDuration})
        elif len(i) == 3:
            content = '%.2f\t%s\t%.2f' % ((int(i[0])+1)*UnitTimeDuration, 
                SpeciesDict[int(i[1])], float(i[2]))
            result_dict.update({content: (int(i[0])+1)*UnitTimeDuration})

    result_dict_Sort = sorted(result_dict.items(), key=lambda x:x[1]) # sort

    for _, (key, value) in enumerate(result_dict_Sort):
        ResultHandle2.write(key+'\n') # write details

    log.info('Summary Done!')

    log = ops.Logs(log, LogFile, logging.DEBUG)
    log.info("\n---------------------------------------------------------------")
    log.info("Total cost " + str(time.time() - time_start))
    log.debug('\nPlease contract us if you find bugs! \nE-mail: chenxing@mail.xtbg.ac.cn ')

    ResultHandle2.close()
    logging.shutdown()

if __name__ == '__main__':
    main()

''' Copyright 2018 '''
