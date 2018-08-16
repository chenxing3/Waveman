import os
import sys
import time
import argparse
import logging
import tensorflow as tf
import numpy as np
from utils import MODEL, ops

def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('Training.py')
    parser.add_argument('--TrainFile', dest='TrainFile', type=str,
                        default = path+'/dataset/tf/train.tfrecords',
                        help='Please enter train tfrecord file')
    parser.add_argument('--ValidFile', dest='ValidFile', type=str,
                        default = path+'/dataset/tf/valid.tfrecords',
                        help='Please enter valid tfrecord file')
    parser.add_argument('--ModelDir', dest='ModelDir', type=str,
                        default = path+'/logs',
                        help='(optimal) folder to store models')

    ## parameters for network
    parser.add_argument('--ClassNum', dest='ClassNum', type=int,
                        default= 38,
                        help='(optimal) class number, default is 38')
    parser.add_argument('--BatchSize', dest='BatchSize', type=int,
                        default= 64,
                        help='(optimal) batch size, default is 64')
    parser.add_argument('--ValidNum', dest='ValidNum', type=int,
                        default= 300,
                        help='(optimal) valid image number, default is 300')
    parser.add_argument('--TrainNum', dest='TrainNum', type=int,
                        default= 111244,
                        help='(optimal) train image number, default is 300')
    parser.add_argument('--PoolSize', dest='PoolSize', type=int,
                        default= 2,
                        help='(optimal) pool size, default is 2')
    parser.add_argument('--NetworkType', dest='NetworkType', type=str,
                        default = 'ResVggNet',
                        help='(optimal) ResVggNet or VggNet')
    parser.add_argument('--LearningRate', dest='LearningRate', type=float,
                        default= 1e-5,
                        help='(optimal) learning rate, default is 1e-5')
    parser.add_argument('--MaxStep', dest='MaxStep', type=int,
                        default= 20,
                        help='(optimal) max train step, default is 100000')
    parser.add_argument('--SaveStep', dest='SaveStep', type=int,
                        default= 0,
                        help='(optimal) max train step, default 0 means \
                                it is TrainNum divide BatchSize')
    parser.add_argument('--TFImageWidth', dest='TFImageWidth', type=int,
                        default= 64,
                        help='(optimal) image width, default is 64')
    parser.add_argument('--TFImageHeight', dest='TFImageHeight', type=int,
                        default= 64,
                        help='(optimal) image height, default is 64')
    parser.add_argument('--Channel', dest='Channel', type=int,
                        default= 3,
                        help='(optimal) image channel, default is 3')
    parser.add_argument('--MinAfterQueue', dest='MinAfterQueue', type=int,
                        default= 100,
                        help='(optimal) min after queue, default is 500')
    parser.add_argument('--Keep', dest='Keep', type=int,
                        default= 100,
                        help='(optimal) keep rate, default is 100')

    args = parser.parse_args()

    if args.NetworkType not in ['ResVggNet', 'VggNet']:
        print('Invalid feature: ', args.NetworkType)
        print("Please input a action: ResVggNet or VggNet")
    if args.ClassNum < 1:
        print('\n-------------->')
        print('Error! class number must be an integer that >= 1')
        sys.exit(1)
    if args.BatchSize < 1:
        print('\n-------------->')
        print('Error! batch size must be an integer that >= 1')
        sys.exit(1)
    if args.ValidNum < 1:
        print('\n-------------->')
        print('Error! valid image number must be an integer that >= 1')
        sys.exit(1)
    if args.TrainNum < 1:
        print('\n-------------->')
        print('Error! total train image number must be an integer that >= 1')
        sys.exit(1)
    if args.PoolSize < 1:
        print('\n-------------->')
        print('Error! pool size must be an integer that >= 1')
        sys.exit(1)
    if args.LearningRate > 1:
        print('\n-------------->')
        print('Error! pool size must be <= 1')
        sys.exit(1)
    if args.MaxStep < 1:
        print('\n-------------->')
        print('Error! max train step must be an integer that >= 1')
        sys.exit(1)
    if args.SaveStep < 0:
        print('\n-------------->')
        print('Error! max save step must be an integer that >= 1')
        sys.exit(1)
    if args.TFImageWidth < 1:
        print('\n-------------->')
        print('Error! image width must be an integer that >= 1')
        sys.exit(1)
    if args.TFImageHeight < 1:
        print('\n-------------->')
        print('Error! image height must be an integer that >= 1')
        sys.exit(1)
    if args.TFImageHeight < 1:
        print('\n-------------->')
        print('Error! image channel must be an integer, colorful image is 3')
        sys.exit(1)
    if args.MinAfterQueue < 1:
        print('\n-------------->')
        print('Error! min after queue be an integer that >= 1')
        sys.exit(1)
    if args.Keep < 1:
        print('\n-------------->')
        print('Error! keep rate be an integer that >= 1')
        sys.exit(1)

    return args

def read_and_decode(filename, args, shuffle=True): 
    '''
    decode tfrecords
    '''
    filename_queue = tf.train.string_input_producer([filename])  # generate a queue

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue) 
    features = tf.parse_single_example(serialized_example,
                                       features={
                                           'label': tf.FixedLenFeature([], tf.int64),
                                           'img_raw': tf.FixedLenFeature([], tf.string),
                                       })  # read image and label

    label = tf.cast(features['label'], tf.int32)
    img = tf.decode_raw(features['img_raw'], tf.uint8)
    img = tf.reshape(img, [args.TFImageWidth, args.TFImageHeight, args.Channel])
    img = tf.cast(img, tf.float32)
    img = (img - 128) / 128.0

    if shuffle:
        imgs, label_batch = tf.train.shuffle_batch(
            [img, label],
            batch_size=args.BatchSize,
            capacity=20000,
            min_after_dequeue=args.MinAfterQueue)
    else:
        imgs, label_batch = tf.train.batch(
            [img, label],
            batch_size=args.BatchSize,
            capacity=20000)

    label_batch = tf.one_hot(label_batch, depth=args.ClassNum)
    label_batch = tf.cast(label_batch, dtype=tf.int32)
    label_batch = tf.reshape(label_batch, [args.BatchSize, args.ClassNum])

    return imgs, label_batch


def main():
    time_start = time.time()
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    args = parse_args(Path)

    # log file 
    LogFile = Path+'/logs/log_training.txt'
    if os.path.isfile(LogFile):
        os.remove(LogFile)
    log = ops.Logs(logging.getLogger(), LogFile, logging.NOTSET)

    log.info("\nThis pipeline is to train dataset.")
    log.info('Find updates in website and see README.md for more information.\n')
    log.info(' '.join(sys.argv) + '\n')

    # load data
    log = ops.Logs(log, LogFile)
    log.info('Loading dataset!')
    tra_image_batch, tra_label_batch = read_and_decode(args.TrainFile, args)
    val_image_batch, val_label_batch = read_and_decode(args.ValidFile, args)

    if args.SaveStep == 0:
        EPOCHS = args.TrainNum // args.BatchSize
    else:
    	EPOCHS = args.SaveStep

    # training
    log.info('Beging training!')
    X = tf.placeholder(tf.float32, [None, args.TFImageWidth, 
        args.TFImageHeight, args.Channel])
    Y = tf.placeholder(tf.float32, [None, args.ClassNum])
    Z = tf.placeholder(tf.bool, name='training')

    network = MODEL.Model(args.ClassNum, (args.PoolSize, args.PoolSize), Z)
    logits1 = getattr(network, args.NetworkType)(X)
    logits = tf.add(logits1, 0, name='output_node')

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(args.LearningRate).minimize(cost)

    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    saver = tf.train.Saver(max_to_keep=args.Keep)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    try:
        for step in np.arange(args.MaxStep):
            if coord.should_stop():
                break

            tra_images, tra_labels = sess.run([tra_image_batch, tra_label_batch])
            feed = {X: tra_images,
                    Y: tra_labels,
                    Z: 1}
            _, tra_loss = sess.run([optimizer, cost], feed)
            if step % 20 == 0 or (step + 1) == args.MaxStep:
                val_images, val_labels = sess.run([val_image_batch, val_label_batch])
                tra_acc = sess.run(accuracy, feed_dict={X: val_images,
                                                        Y: val_labels,
                                                        Z: 0})

                log.info('Step: %d, loss: %.8f, acc: %.8f' % (step, tra_loss, tra_acc))
            if step % EPOCHS == 0 or (step + 1) == args.MaxStep:
                checkpoint_path = args.ModelDir+'/model.ckpt'
                saver.save(sess, checkpoint_path, global_step=step)
                log.info("Model saved!")

    except tf.errors.OutOfRangeError:
        log.info('Done training -- epoch limit reached')
    finally:
        coord.request_stop()

    coord.join(threads)
    sess.close()

    log = ops.Logs(log, LogFile, logging.DEBUG)
    log.info("Total cost " + str(time.time() - time_start))
    log.debug('\nPlease contract us if you find bugs! \nE-mail: chenxing@mail.xtbg.ac.cn ')

if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # GPU first, otherwise use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # only print error information!
    main()
