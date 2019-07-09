import os
import shutil
import argparse
import numpy as np
import scipy.io
import tensorflow as tf
from model import RRAM

EPOCHS = 100
TRAIN_TEST_SPLIT = .9
BATCH_SIZE = 64
PRINT_EACH_ITERATIONS = 1
SUMMARY_EACH_ITERATIONS = 10
SAVE_EACH_ITERATIONS = 200
TEST_EACH_ITERATIONS = 20
SAVE_PATH = '/models/'
LOG_PATH = '/log/'
DATA_PATH = '/resources/relaxation_data.mat'

#create folder
parser = argparse.ArgumentParser()
parser.add_argument("-r", "--results-folder", default="results")
parser.add_argument("-o", "--overwrite-results", type=int, choices=[0, 1], default=0)
args = parser.parse_args()
if os.path.exists(args.results_folder):
    if args.overwrite_results:
        shutil.rmtree(args.results_folder, ignore_errors=True)
    else:
        folder, i = args.results_folder, 0
        args.results_folder = "{}_{}".format(folder, i)
        while os.path.exists(args.results_folder):
            i += 1
            args.results_folder = "{}_{}".format(folder, i)

SAVE_FOLDER = args.results_folder + SAVE_PATH
LOG_FOLDER = args.results_folder + LOG_PATH
os.makedirs(args.results_folder)
os.makedirs(SAVE_FOLDER)
os.makedirs(LOG_FOLDER)


#load data
mat = scipy.io.loadmat(DATA_PATH)
data_full = mat.get('relaxation_data').astype(np.float32)
print('loading mat file, shape=')
print(data_full.shape)
ntrain = int(data_full.shape[0] * TRAIN_TEST_SPLIT)
ntest = data_full.shape[0] - ntrain
print('use {} data for training and {} data for testing'.format(ntrain, ntest))
data_train = data_full[:ntrain]
data_test = data_full[ntrain:]
W_train = np.expand_dims(data_train[:, 0], axis=1)
R_train = data_train[:, 1:]
M_train = R_train.shape[0]
N_train = R_train.shape[1]
W_test = np.expand_dims(data_test[:, 0], axis=1)
R_test = data_test[:, 1:]
M_test = R_test.shape[0]
N_test = R_test.shape[1]
dataset = tf.data.Dataset.from_tensor_slices({'W': W_train, 'R': R_train}).repeat(EPOCHS)
dataset = dataset.batch(BATCH_SIZE)
iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()
W_placeholder_train = next_element.get('W')
R_placeholder_train = next_element.get('R')
model_inputs = [[W_placeholder_train, R_placeholder_train, BATCH_SIZE, N_train],
                [W_test, R_test, M_test, N_test]]


#create model
models = []
for i in range(2):
    print("Creating {0} model...".format("training" if i == 0 else "testing"))
    models.append(
        RRAM(
            M=model_inputs[i][2], N=model_inputs[i][3], W=model_inputs[i][0], R=model_inputs[i][1],
            num_state=3, non_drift_var=.001,
            p_a_layers=(16,),
            p_v_layers=(16,),
            q_v_layers=(32, 16,),
            learning_rate=3e-4,
            gradient_clipping_norm=100.0,
            train=(i==0),
            reuse=(i==1)
        )
    )
train_model, test_model = models
train_summaries = tf.summary.merge(train_model.summaries)
test_summaries = tf.summary.merge(test_model.summaries)

#train
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    print("Initializing variables...\n")
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=5)
    writer = tf.summary.FileWriter(LOG_FOLDER, graph=sess.graph)

    print("Training...\n")
    step = 0
    while True:
        try:
            # test step
            if step % TEST_EACH_ITERATIONS == 0:
                ELBO, summ = sess.run([test_model.ELBO, test_summaries])
                print("Test ELBO {:.2f}".format(ELBO))
                writer.add_summary(summ, step)
            # train step
            _, ELBO, step, summ = sess.run([
                train_model.training, train_model.ELBO, train_model.global_step, train_summaries])
            if step % SAVE_EACH_ITERATIONS == 0:
                saver.save(
                    sess, SAVE_FOLDER + "RRAM",
                    global_step=step
                )
            if step % PRINT_EACH_ITERATIONS == 0:
                print("Iteration {}\t Train ELBO {:.2f}".format(step, ELBO))
            if step % SUMMARY_EACH_ITERATIONS == 0:
                writer.add_summary(summ, step)
        except tf.errors.OutOfRangeError:
            break

    print("Finished.")
