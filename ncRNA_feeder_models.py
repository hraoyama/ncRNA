import tensorflow as tf
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.models import Model
from tensorflow.keras.layers import  Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply, SimpleRNN, GRU, LeakyReLU
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import site
import pandas as pd
import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard

site.addsitedir("D:/Code/ncRNA")
os.chdir("D:/Code/ncRNA")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import getData, coShuffled_vectors, reverse_tensor, get_combined_features_from_models, getE2eData



def model_with_pure_rnn_finalist(model_name, input_shape = (1000, 8,)):

    # RNN part
    # (TIME_STEPS, INPUT_DIM,)
    inputs = Input(shape=input_shape)
    lstm_one = Bidirectional \
        (GRU(256, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.5, recurrent_dropout = 0.5, recurrent_initializer='RandomNormal', bias_initializer='zero'))(inputs)
    # lstm_one = Dropout(0.3)(lstm_one)
    # lstm_one = GaussianNoise(0.05)(lstm_one)
    lstm_two = Bidirectional \
        (GRU(128, return_sequences=True, kernel_initializer='RandomNormal', dropout= 0.5, recurrent_dropout = 0.5, recurrent_initializer='RandomNormal', bias_initializer='zero'))(lstm_one)
    # lstm_two = Dropout(0.3)(lstm_two)
    # lstm_two = GaussianNoise(0.05)(lstm_two)
    attention = SeqWeightedAttention()(lstm_two)
    # attention_mul = attention_3d_block(lstm_two)
    attention = Flatten()(attention)
    rnnoutput = Dense(256 ,kernel_initializer='RandomNormal', bias_initializer='zeros')(attention)
    rnnoutput = BatchNormalization()(rnnoutput)
    rnnoutput = GaussianNoise(1)(rnnoutput)
    rnnoutput = Dropout(0.4)(rnnoutput)

    # Dense Feed-forward
    dense_one = Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros')(rnnoutput)
    dense_one = LeakyReLU()(dense_one)
    dense_one = Dropout(0.5)(dense_one)
    dense_one = BatchNormalization()(dense_one)
    dense_two = Dense(64, kernel_initializer='RandomNormal', bias_initializer='zeros')(dense_one)
    dense_two = LeakyReLU()(dense_two)
    dense_two = Dropout(0.4)(dense_two)

    # Output
    output = Dense(13, activation='softmax')(dense_two)
    model = Model([inputs], output, name = model_name)
    return model

def baseline_CNN_finalist(model_name, inshape, num_classes = 13):

    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv1D(128 ,10 ,padding='same' ,input_shape=inshape))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(128 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(2))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Conv1D(256 ,10 ,padding='same'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))
    model.add(tf.keras.layers.MaxPooling1D(4))

    model.add(tf.keras.layers.GaussianNoise(1))
    model.add(tf.keras.layers.Dropout(rate=0.5))

    model.add(tf.keras.layers.Flatten())

    model.add(tf.keras.layers.Dense(128))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(64))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.LeakyReLU(alpha=0.5))

    model.add(tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax))
    model._name = model_name

    return model


def compile_and_fit_model_basic(  model_func,
                                  model_name,
                                  input_shape,
                                  X_train,
                                  Y_train,
                                  save_max_epoch=True,
                                  save_final=False,
                                  patience_count = None,
                                  **kwargs):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)

    callbacks_used = []
    if save_max_epoch:
        callbacks_used.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{val_accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='val_accuracy',
                                              mode='max',
                                              save_best_only=True))
    if patience_count is not None:
        callbacks_used.append(tf.keras.callbacks.EarlyStopping(patience=patience_count))

    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=callbacks_used, verbose=2, **kwargs)
    if save_final:
        make_dir_if_not_exist(model_name)
        m.save(f"{m.name}_saved_model_after_fit")  # Save the model
    return (m, history)


if __name__ == "__main__":
    
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData(is500=False,
                                                                                                    include_secondary=False)

    X_train_1000e, Y_train_1000e = coShuffled_vectors(X_train_1000e, Y_train_1000e)
    X_test_1000e, Y_test_1000e = coShuffled_vectors(X_test_1000e, Y_test_1000e)

    cnn_2, history_cnn_2 = compile_and_fit_model_basic(baseline_CNN_finalist,
                                                  "cnn_newdata_20210516",
                                                  X_train_1000e[0].shape,
                                                  X_train_1000e,
                                                  Y_train_1000e,
                                                  save_max_epoch=True,
                                                  save_final=True,
                                                  patience_count=10,
                                                  batch_size=128,
                                                  epochs=150,
                                                  class_weight=None,
                                                  validation_data=(X_val_1000e, Y_val_1000e))
    cnn_2.evaluate(X_test_1000e, Y_test_1000e)




