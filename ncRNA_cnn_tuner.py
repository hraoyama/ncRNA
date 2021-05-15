import tensorflow as tf
<<<<<<< HEAD:ncRNA_cnn_tuner.py
=======
import numpy as np
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, \
    MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import SimpleRNN, GRU, LeakyReLU
from tensorflow.keras.layers import Concatenate, Average
>>>>>>> parent of 3267b1c (base models checked in):rnn_cnn_tb_tuner_setup.py
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Bidirectional
<<<<<<< HEAD:ncRNA_cnn_tuner.py
from tensorflow.keras.models import Model
=======
from timeit import default_timer as timer
import h5py as h5
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
import errno
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import random
import warnings
import gpflow
from gpflow.utilities import ops, print_summary, set_trainable
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter
>>>>>>> parent of 3267b1c (base models checked in):rnn_cnn_tb_tuner_setup.py
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, \
    MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply, \
    SimpleRNN, GRU, LeakyReLU
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
<<<<<<< HEAD:ncRNA_cnn_tuner.py
import kerastuner as kt
import site
import pandas as pd
=======
from tensorflow.keras import regularizers
import os
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import kerastuner as kt
from kerastuner import HyperModel
from tensorflow.summary import create_file_writer
from functools import partial
>>>>>>> parent of 3267b1c (base models checked in):rnn_cnn_tb_tuner_setup.py

"""Data Preparation"""

warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow
set_default_float('float64')
os.chdir("D:/papers/RNA_NN_GP_ARD/ncRFP/ncRFP_Model/")

# load the input data
INPUT_DIM = 8  #
TIME_STEPS = 500  # The step of RNN


# data extraction
def getData(is500=True):
    hf_Train = h5.File(f'Fold_10_Train_Data_{str(500) if is500 else str(1000)}.h5', 'r')
    hf_Test = h5.File(f'Fold_10_Test_Data_{str(500) if is500 else str(1000)}.h5', 'r')
    X_train = hf_Train['Train_Data']  # Get train set
    X_train = np.array(X_train)
    Y_train = hf_Train['Label']  # Get train label
    Y_train = np.array(Y_train)
    X_test = hf_Test['Train_Data']  # Get test set
    X_test = np.array(X_test)
    Y_test = hf_Test['Label']  # Get test label
    Y_test = np.array(Y_test)
    Y_train = to_categorical(Y_train, 13)  # Process the label of tain
    Y_test = to_categorical(Y_test, 13)  # Process the label of te
    return (X_train, Y_train, X_test, Y_test)


# rnn3 model creation
def model_with_pure_rnn3():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIM,))
    lstm_one = Bidirectional(
        GRU(256, return_sequences=True, kernel_initializer='RandomNormal', dropout=0.5, recurrent_dropout=0.5,
            recurrent_initializer='RandomNormal', bias_initializer='zero'))(inputs)
    lstm_two = Bidirectional(
        GRU(128, return_sequences=True, kernel_initializer='RandomNormal', dropout=0.5, recurrent_dropout=0.5,
            recurrent_initializer='RandomNormal', bias_initializer='zero'))(lstm_one)
    lstm_two = Bidirectional(
        GRU(64, return_sequences=True, kernel_initializer='RandomNormal', dropout=0.5, recurrent_dropout=0.5,
            recurrent_initializer='RandomNormal', bias_initializer='zero'))(lstm_two)
    attention_mul = SeqWeightedAttention()(lstm_two)
    attention_mul = Flatten()(attention_mul)
    dense_one = Dense(256, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu',
                      name="antepenultimate_dense")(attention_mul)
    dense_one = Dropout(0.5)(dense_one)
    dense_two = Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu',
                      name="penultimate_dense")(dense_one)
    dense_two = Dropout(0.4)(dense_two)
    dense_three = Dense(64, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu',
                        name="last_dense")(dense_two)
    dense_three = Dropout(0.3)(dense_two)
    output = Dense(13, activation='softmax', name="last_softmax")(dense_three)
    model = Model([inputs], output, name="pure_rnn3")
    return model


# plotting history of a fit
def plot_history(history):
    acc_keys = [k for k in history.history.keys() if k in ('accuracy', 'val_accuracy')]
    loss_keys = [k for k in history.history.keys() if not k in acc_keys]
    for k, v in history.history.items():
        if k in acc_keys:
            plt.figure(1)
            plt.plot(v)
        else:
            plt.figure(2)
            plt.plot(v)
    plt.figure(1)
    plt.title('Accuracy vs. epochs')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(acc_keys, loc='upper right')
    plt.figure(2)
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loss_keys, loc='upper right')
    plt.show()


def get_layer_by_name(layers, name, return_first=True):
    matching_named_layers = [l for l in layers if l.name == name]
    if not matching_named_layers:
        return None
    return matching_named_layers[0] if return_first else matching_named_layers


def get_combined_features_from_models(
        to_combine,
        X_train, Y_train,
        X_test, Y_test,
        reverse_one_hot=False,
        normalize_X_func=None):
    models = dict()
    X_trains_out = []
    X_test_out = []
    XY_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))

    if reverse_one_hot:
        Y_train_new = np.apply_along_axis(np.argmax, 1, Y_train) + 1
        Y_test_new = np.apply_along_axis(np.argmax, 1, Y_test) + 1
    else:
        Y_train_new = Y_train.copy()
        Y_test_new = Y_test.copy()

    for model_file_name, layer_name, kwargs in to_combine:
        model_here = None
        if isinstance(model_file_name, tf.keras.models.Model):
            model_here = model_file_name
            model_file_name = model_here.name
        else:
            if model_file_name in models.keys():
                model_here = models[model_file_name]
            else:
                model_here = tf.keras.models.load_model(model_file_name,
                                                        **kwargs) if kwargs is not None else tf.keras.models.load_model(
                    model_file_name)
        features_model = Model(model_here.input,
                               get_layer_by_name(model_here.layers, layer_name).output)
        if normalize_X_func is None:
            X_trains_out.append(np.array(features_model.predict(X_train), dtype='float64'))
            X_test_out.append(np.array(features_model.predict(X_test), dtype='float64'))
        else:
            X_trains_out.append(np.array(normalize_X_func(features_model.predict(X_train)), dtype='float64'))
            X_test_out.append(np.array(normalize_X_func(features_model.predict(X_test)), dtype='float64'))
        XY_dict[model_file_name][layer_name]['Train']['X'] = X_trains_out[-1]
        XY_dict[model_file_name][layer_name]['Test']['X'] = X_test_out[-1]
        XY_dict[model_file_name][layer_name]['Train']['Y'] = Y_train_new
        XY_dict[model_file_name][layer_name]['Test']['Y'] = Y_test_new
        models[model_file_name] = model_here

    X_train_new = np.concatenate(tuple(X_trains_out), axis=1)
    X_test_new = np.concatenate(tuple(X_test_out), axis=1)

    data_train = (X_train_new, Y_train_new)
    data_test = (X_test_new, Y_test_new)

    return (models, data_train, data_test, XY_dict)


def make_dir_if_not_exist(used_path):
    if not os.path.isdir(used_path):
        try:
            os.mkdir(used_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise exc
            else:
                raise ValueError(f'{used_path} directoy cannot be created because its parent directory does not exist.')


def model_with_cnn_2(model_name, input_shape):
    model = Sequential([
        Conv1D(128, 3, padding='same', input_shape=input_shape),
        LeakyReLU(),
        MaxPooling1D(3),
        BatchNormalization(),
        GaussianNoise(0.05),
        # Bidirectional(GRU(128, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.3, recurrent_dropout = 0.3, recurrent_initializer='RandomNormal', bias_initializer='zero')),
        Conv1D(128, 3, padding='same'),
        LeakyReLU(),
        Conv1D(128, 3, padding='same'),
        LeakyReLU(),
        MaxPooling1D(3),
        BatchNormalization(),
        GaussianNoise(0.05),
        Conv1D(256, 3, padding='same'),
        LeakyReLU(),
        Conv1D(256, 3, padding='same'),
        LeakyReLU(name="last_leakyrelu"),
        MaxPooling1D(3),
        BatchNormalization(name="last_batchnorm"),
        GaussianNoise(0.05),
        Flatten(name="last_flatten"),
        Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu',
              name="penultimate_dense"),
        Dropout(0.2, name="penultimate_dropout"),
        Dense(64, kernel_initializer='RandomNormal', bias_initializer='zeros', activation='relu', name="last_dense"),
        Dropout(0.2, name="last_dropout"),
        Dense(13, activation='softmax', name="last_softmax")
    ], name=model_name)
    return model


<<<<<<< HEAD:ncRNA_cnn_tuner.py
if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow

    # os.chdir("D:/papers/RNA_NN_GP_ARD/ncRFP/ncRFP_Model/")
    # load the input data

    # X_train, Y_train, X_test, Y_test, X_validation, Y_validation  = getData(is500=False, validation_split=0.1)
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation = getData(is500=False)
    X_test_shuffled, Y_test_shuffled = coShuffled_vectors(X_test, Y_test)
    X_train_shuffled, Y_train_shuffled = coShuffled_vectors(X_train, Y_train)
    X_train_rev = reverse_tensor(X_train)
    Y_train_rev = reverse_tensor(Y_train)

    # model_gen_func = partial(model_with_cnn_2, model_name="cnn_to_hp_with_kt", input_shape=X_train[0].shape)
    # tuner = kt.Hyperband(
    #     model_gen_func,
    #     objective='val_accuracy',
    #     max_epochs=60,
    #     hyperband_iterations=4)

    NUM_CLASSES = Y_train[0].shape[0]
    INPUT_SHAPE = X_train[0].shape
    hypermodel_cnn = CNNHyperModel(model_name="cnn_to_hp_with_kt", input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
    tuner = kt.Hyperband(
        hypermodel_cnn,
        max_epochs=60,
        objective='val_accuracy',
        executions_per_trial=4,
        directory='hyperband'
    )
    HYPERBAND_MAX_EPOCHS = 40
    MAX_TRIALS = 20
    EXECUTION_PER_TRIAL = 2
    # tuner.search(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=50)
    tuner.search(X_train, Y_train, epochs=50, validation_split=0.2,
                 callbacks=[tf.keras.callbacks.EarlyStopping(patience=2)])

    # Show a summary of the search
    tuner.results_summary()
    # Retrieve the best model.
    best_model = tuner.get_best_models(num_models=1)[0]
    # Evaluate the best model.
    loss, accuracy = best_model.evaluate(X_test, Y_test)
=======
def run_and_save_model(model_func, model_name, input_shape, X_train, Y_train, kwargs):
    make_dir_if_not_exist(model_name)
    m = model_func(model_name, input_shape)
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, **kwargs)
    m.save(f"{m.name}_Tenth_Fold_New_Model_500_8")  # Save the model
    return (m, history)


def source_model(model_func, model_name, input_shape):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)
    return m


def compile_model_and_fit_with_custom_loop(model_func,
                                           model_name,
                                           input_shape,
                                           X_train,
                                           Y_train,
                                           send_to_tb=True,
                                           **kwargs):
    if not isinstance(model_func, tf.keras.models.Model):
        make_dir_if_not_exist(model_name)
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)

    train_writer = create_file_writer(f'{m.name}_logs/train/')
    test_writer = create_file_writer(f'{m.name}_logs/test/')
    train_step = test_step = 0

    acc_metric = tf.keras.metrics.CategoricalAccuracy()
    optimizer = tf.keras.optimizers.Adam()
    num_epochs = kwargs.get("epochs", 10)

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    BATCH_SIZE = kwargs.get("batch_size", 32)
    X_test, Y_test = kwargs.get("validation_data", (None, None))
    if X_test is None:
        raise ValueError("Missing X validation data")
    if Y_test is None:
        raise ValueError("Missing Y validation data")

    train_dataset_tf = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    train_dataset_tf = train_dataset_tf.batch(BATCH_SIZE)
    train_dataset_tf = train_dataset_tf.prefetch(AUTOTUNE)

    test_dataset_tf = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    test_dataset_tf = train_dataset_tf.batch(BATCH_SIZE)
    test_dataset_tf = train_dataset_tf.prefetch(AUTOTUNE)

    loss_fn = tf.keras.losses.CategoricalCrossentropy()

    for epoch in range(num_epochs):
        # Iterate through training set
        for batch_idx, (x, y) in enumerate(train_dataset_tf):
            with tf.GradientTape() as tape:
                y_pred = m(x, training=True)
                loss = loss_fn(y, y_pred)

            gradients = tape.gradient(loss, m.trainable_weights)
            optimizer.apply_gradients(zip(gradients, m.trainable_weights))
            acc_metric.update_state(y, y_pred)

            if send_to_tb:
                with train_writer.as_default():
                    tf.summary.scalar("Loss", loss, step=train_step)
                    tf.summary.scalar(
                        "Accuracy", acc_metric.result(), step=train_step,
                    )
                    train_step += 1
        # Reset accuracy in between epochs (and for testing and test)
        acc_metric.reset_states()

        # Iterate through test set
        for batch_idx, (x, y) in enumerate(test_dataset_tf):
            y_pred = m(x, training=False)
            loss = loss_fn(y, y_pred)
            acc_metric.update_state(y, y_pred)
            if send_to_tb:
                with test_writer.as_default():
                    tf.summary.scalar("Loss", loss, step=test_step)
                    tf.summary.scalar(
                        "Accuracy", acc_metric.result(), step=test_step,
                    )
                    test_step += 1

        acc_metric.reset_states()  # Reset accuracy in between epochs (and for testing and test)

    return m


def compile_and_fit_model_with_tb(model_func,
                                  model_name,
                                  input_shape,
                                  X_train,
                                  Y_train,
                                  save_every_epoch=True,
                                  **kwargs):
    if not isinstance(model_func, tf.keras.models.Model):
        make_dir_if_not_exist(model_name)
    m = source_model(model_func, model_name, input_shape)
    tb_callback = TensorBoard(log_dir=f'{m.name}_logs', histogram_freq=kwargs.pop("histogram_freq", 1))
    callbacks_used = [tb_callback]
    if save_every_epoch:
        callbacks_used.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{val_accuracy:0.2f}'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=callbacks_used, verbose=2, **kwargs)
    return (m, history)
    # m.save(f"{m.name}_Tenth_Fold_New_Model_500_8") #Save the model

# optimize a data set on tensorboard
X_train, Y_train, X_test, Y_test = getData(is500=True)
cnn_2, history_cnn_2 = compile_and_fit_model_with_tb(model_with_cnn_2,
                                                     "cnn2_input_base_20210501",
                                                     X_train[0].shape,
                                                     X_train,
                                                     Y_train,
                                                     batch_size=128,
                                                     epochs=50,
                                                     class_weight=None,
                                                     validation_data=(X_test, Y_test))
cnn_2.evaluate(X_test, Y_test)

class CNNHyperModel(HyperModel):

    def __init__(self, model_name, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs

        for idx, i in enumerate(range(hp.Int('conv128_blocks_with_normalizations', 1, 6, default=4))):
            x = Conv1D(128, 3, padding='same', name=f"conv1D_128_{idx}")(x)
            if hp.Boolean(f'conv128_has_leaky_relu_{idx}', default=True):
                x = LeakyReLU()(x)
            if hp.Boolean(f'conv128_has_max_pooling_{idx}', default=True):
                x = MaxPooling1D()(x)
            if hp.Boolean(f'conv128_has_batchnorm_{idx}', default=True):
                x = BatchNormalization()(x)
            if hp.Boolean(f'conv128_has_gaussiannoise_{idx}', default=True):
                x = GaussianNoise(hp.Float(f'conv128_gaussiannoise_{idx}',
                                       min_value=1e-5,
                                       max_value=1e1,
                                       sampling='LOG',
                                       default=0.05
                                       ))(x)
        for idx, i in enumerate(range(hp.Int('conv256_blocks_with_normalizations', 1, 4, default=2))):
            x = Conv1D(256, 3, padding='same', name=f"conv1D_256_{idx}")(x)
            if hp.Boolean(f'conv256_has_leaky_relu_{idx}', default=True):
                x = LeakyReLU()(x)
            if hp.Boolean(f'conv256_has_max_pooling_{idx}', default=True):
                x = MaxPooling1D()(x)
            if hp.Boolean(f'conv256_has_batchnorm_{idx}', default=True):
                x = BatchNormalization()(x)
            if hp.Boolean(f'conv256_has_gaussiannoise_{idx}', default=True):
                x = GaussianNoise(hp.Float(f'conv256_gaussiannoise_{idx}',
                                       min_value=1e-5,
                                       max_value=1e1,
                                       sampling='LOG',
                                       default=0.05
                                       ))(x)
        x = Flatten(name="last_flatten")(x)
        for idx, i in enumerate(range(hp.Int('final_dense', 1, 5, default=2))):
            x = Dense(units=hp.Choice(f'final_dense_num_nodes_{idx}', values=[16, 32, 64, 128], default=128),
                  activation=hp.Choice(f'final_dense_kernel_init_{idx}',
                                       values=['exponential', 'gelu', 'elu', 'relu', 'tanh'], default='relu'),
                  kernel_initializer='RandomNormal',
                  bias_initializer='zeros',
                  name=f"final_dense_{idx}")(x)
            if hp.Boolean(f'final_dense_has_dropout_{idx}', default=True):
                x = Dropout(hp.Float(f'final_dense_dropout_{idx}',
                                 min_value=0.05,
                                 max_value=0.75,
                                 step=0.05,
                                 default=0.2
                                 ), name=f"final_dense_dropout_{idx}")(x)
        outputs = Dense(self.num_classes, activation='softmax', name="last_softmax")(x)
        model = tf.keras.Model(inputs, outputs, name=self.model_name)
        #  m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        if hp.Boolean('optimize_adam', default=True):
            model.compile(
                optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', 1e-5, 1e-1, sampling='log')),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        else:
            model.compile(
                optimizer=hp.Choice('final_optimizer',
                                    values=['adam', 'SGD', 'RMSprop', 'Adadelta', 'Nadam', 'Adamax', 'Adagrad'],
                                    default='adam'),
                loss='categorical_crossentropy',
                metrics=['accuracy'])
        return model


# using hyperparameters
model_gen_func = partial(hp_cnn2, model_name="cnn_to_hp_with_kt", input_shape=X_train[0].shape)
tuner = kt.Hyperband(
    model_gen_func,
    objective='val_accuracy',
    max_epochs=60,
    hyperband_iterations=4)

NUM_CLASSES = Y_train[0].shape[0]
INPUT_SHAPE = X_train[0].shape
hypermodel_cnn = CNNHyperModel(model_name="cnn_to_hp_with_kt", input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES)
tuner = kt.Hyperband(
    hypermodel_cnn,
    max_epochs=60,
    objective='val_accuracy',
    executions_per_trial=4,
    directory='hyperband'
)

HYPERBAND_MAX_EPOCHS = 40
MAX_TRIALS = 20
EXECUTION_PER_TRIAL = 2

tuner.search(x=X_train, y=Y_train, validation_data=(X_test, Y_test), epochs=50)

tuner.search(X_train, Y_train, epochs=50, validation_split=0.2,
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])

# Show a summary of the search
tuner.results_summary()
# Retrieve the best model.
best_model = tuner.get_best_models(num_models=1)[0]
# Evaluate the best model.
loss, accuracy = best_model.evaluate(x_test, y_test)

X_train, Y_train, X_test, Y_test = getData(is500=False)
mRNN3x = load_model("RNN_3stacked_23_epochs.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
rnn_2, history_rnn_2 = compile_and_fit_model_with_tb(mRNN3x,
                                                     "rnn3_input_base_fbg_20210501",
                                                     X_train[0].shape,
                                                     X_train,
                                                     Y_train,
                                                     batch_size=128,
                                                     epochs=10,
                                                     class_weight=None,
                                                     validation_data=(X_test, Y_test))

# mRNN3, mRNN3_fit_history = run_and_save_model(model_with_pure_rnn3, X_train, Y_train,  { "batch_size":512, "epochs":2, "class_weight":None, "validation_data":(X_test, Y_test)} )
# mRNN3 = load_model("pure_rnn3_Tenth_Fold_New_Model_500_8")
# tf.keras.utils.plot_model(mRNN3, show_shapes=True)
# mCNNx = load_model("CNN_new.h5")
# mCNNx._name = "CNN_base_input"
# mRNN3_checkpoint = ModelCheckpoint(f'{mRNN3.name}' + '_model_{epoch:02d}_{val_accuracy:0.2f}')
# mRNN3_tensorboard = TensorBoard(log_dir=f'{mRNN3.name}_logs')
# mRNN3.fit(X_train, Y_train, batch_size=512,
#           epochs=20, class_weight=None, validation_data=(X_test, Y_test),
#           callbacks=[mRNN3_checkpoint, mRNN3_tensorboard])
# mRNN3x = load_model("RNN_3stacked_23_epochs.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
# mRNN3x.evaluate(X_test, Y_test)
# mRNN = tf.keras.models.load_model("PureRNN.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
# mCNN = tf.keras.models.load_model("CNN_new.h5")
# mRNN._name = "rnn_model"
# mCNN._name = "cnn_model"
>>>>>>> parent of 3267b1c (base models checked in):rnn_cnn_tb_tuner_setup.py
