from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
from gpflow.config import set_default_float
import warnings
from tensorflow.keras.layers import Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, \
    MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply, \
    SimpleRNN, GRU, LeakyReLU
from keras_self_attention import SeqWeightedAttention
import site
import pandas as pd

site.addsitedir("D:/Code/ncRNA")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import getData, coShuffled_vectors, reverse_tensor, compile_and_fit_model_with_tb

def model_with_pure_rnn3(input_shape=(500, 8,)):
    inputs = Input(shape=input_shape)
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


if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow
    set_default_float('float64')

    # os.chdir("D:/papers/RNA_NN_GP_ARD/ncRFP/ncRFP_Model/")
    # load the input data
    X_train, Y_train, X_test, Y_test, X_validation, Y_validation  = getData(is500=False, validation_split=0.1)
    # X_train, Y_train, X_test, Y_test, X_validation, Y_validation = getData(is500=False)
    X_test_shuffled, Y_test_shuffled = coShuffled_vectors(X_test, Y_test)
    X_train_shuffled, Y_train_shuffled = coShuffled_vectors(X_train, Y_train)
    X_train_rev = reverse_tensor(X_train)
    Y_train_rev = reverse_tensor(Y_train)
    cnn_2, history_cnn_2 = compile_and_fit_model_with_tb(model_with_cnn_2,
                                                  "cnn2_input_base_20210501",
                                                  X_train[0].shape,
                                                  X_train,
                                                  Y_train,
                                                  save_every_epoch=False,
                                                  save_final=True,
                                                  batch_size=128,
                                                  epochs=50,
                                                  class_weight=None,
                                                  validation_data=(X_validation, Y_validation))
    cnn_2.evaluate(X_test, Y_test)

    # X_train, Y_train, X_test, Y_test = getData(is500=False)
    # mRNN3x = load_model("RNN_3stacked_23_epochs.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    # rnn_2, history_rnn_2 = compile_and_fit_model_with_tb(mRNN3x,
    #                                                      "rnn3_input_base_fbg_20210501",
    #                                                      X_train[0].shape,
    #                                                      X_train,
    #                                                      Y_train,
    #                                                      batch_size=128,
    #                                                      epochs=10,
    #                                                      class_weight=None,
    #                                                      validation_data=(X_test, Y_test))

    # INPUT_DIM = 8  #
    # TIME_STEPS = 500  # The step of RNN
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
    # # mRNN3x = load_model("RNN_3stacked_23_epochs.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    # mRNN3x.evaluate(X_test, Y_test)
    # mRNN = tf.keras.models.load_model("PureRNN.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    # mCNN = tf.keras.models.load_model("CNN_new.h5")
    # mRNN._name = "rnn_model"
    # mCNN._name = "cnn_model"