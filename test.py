import numpy as np
import scipy.io
import tensorflow as tf
import argparse
from model import RRAM

SAVE_PATH = '/models/'
DATA_PATH = 'relaxation_data.mat'

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results-folder", default="results")
parser.add_argument("-m", "--model-name", default="RRAM-200")
args = parser.parse_args()
MODEL_PATH = args.results_folder + SAVE_PATH + args.model_name

#load data
mat = scipy.io.loadmat(DATA_PATH)
data_full = mat.get('relaxation_data').astype(np.float32)
print('loading mat file, shape=')
print(data_full.shape)
W = np.expand_dims(data_full[:5, 0], axis=1)
R = data_full[:5, 1:]
M = R.shape[0]
N = R.shape[1]

#load model
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
rram = RRAM(
            M=M, N=N, W=W, R=R,
            num_state=3, non_drift_var=.001,
            p_a_layers=(16,),
            p_v_layers=(16,),
            q_v_layers=(32, 16,),
            learning_rate=.01,
            gradient_clipping_norm=100.0,
            train=True,
            reuse=False
        )
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    print("Model restore from {} ...".format(MODEL_PATH))
    saver.restore(sess, MODEL_PATH)
    v_var, v_mean = sess.run([rram.v_var, rram.v_mean])
    print('Done.')