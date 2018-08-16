import os
import sys
import argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util


def parse_args(path):
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser('freeze.py')
    parser.add_argument('--ModelDir', dest='ModelDir', type=str,
                        default = path+'/logs',
                        help='Please enter train tfrecord file')
    parser.add_argument('--Output', dest='Output', type=str,
                        default = path+'/model.ckpt/frozen_model.pb',
                        help='Please enter frozen file')
    args = parser.parse_args()

    return args

def freeze_graph():
    # get absolute path of Waveman and parse input arguments
    Path, _ = os.path.split(os.path.abspath(sys.argv[0]))
    args = parse_args(Path)

    checkpoint = tf.train.get_checkpoint_state(args.ModelDir)
    input_checkpoint = checkpoint.model_checkpoint_path
    output_node_names = "output_node"

    clear_devices = True

    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )


        with tf.gfile.GFile(args.Output, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        print('Done!')


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # GPU first, otherwise use CPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
    os.environ["TF_CPP_MIN_LOG_LEVEL"]='1' # only print error information!
    freeze_graph()
