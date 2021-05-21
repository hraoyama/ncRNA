import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import BatchNormalization, Dense, LeakyReLU, Dropout
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import site
import pandas as pd
import numpy as np
import matplotlib
import os

site.addsitedir("D:/Code/ncRNA")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import *


def get_features_model(model_name, combined_models):
    for ids, models in combined_models:
        if models[0].name == model_name:
            return models[1]
    return None

def get_features_from_combined_models(combined_models, X_input):
    data_features = []
    for ids, models in combined_models:
        data_features.append(np.array(models[1].predict(X_input), dtype='float64'))
    return np.concatenate(tuple(data_features), axis=1)


def model_combination(model_name, input_shape):
    model = Sequential([
        tf.keras.Input(shape=input_shape),
        BatchNormalization(),
        Dense(256, kernel_initializer='RandomNormal', bias_initializer='zeros'),
        LeakyReLU(),
        Dropout(0.6),
        #GaussianNoise(0.1),
        Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-3)),
        LeakyReLU(),
        #Dense(32, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-2)),
        #LeakyReLU(),
        Dropout(0.6),
        Dense(32, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-2)),
        LeakyReLU(),
        Dropout(0.5),
        # Dense(128, kernel_initializer='RandomNormal', bias_initializer='zeros', kernel_regularizer = tf.keras.regularizers.l1(1e-2)),
        # LeakyReLU(),
        Dense(13, activation='softmax')
    ], name=model_name)
    return model


if __name__ == "__main__":
    
    os.chdir("D:/Code/ncRNA")
    
    # 'new' data
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData(is500=False,
                                                                                                    include_secondary=False)
    X_train_1000e_w2nd, Y_train_1000e_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd, X_val_1000e_w2nd, Y_val_1000e_w2nd = getE2eData(is500=False,
                                                                                                    include_secondary=True)
    
    # merge into a new train:
    X_new_train = np.concatenate( (X_train_1000e, X_val_1000e), axis=0 )
    Y_new_train = np.concatenate( (Y_train_1000e, Y_val_1000e), axis=0 )    
    
    X_new_train_w2nd = np.concatenate( (X_train_1000e_w2nd, X_val_1000e_w2nd), axis=0 )
    Y_new_train_w2nd = np.concatenate( (Y_train_1000e_w2nd, Y_val_1000e_w2nd), axis=0 )    

    # CNNs no secondary
    mCNN1_1000 = load_model("./models/CNN_baseline_May16_e2e1000_256.h5") # CNN on 256 1st dens
    mCNN1_1000._name = "cnn_merged_newdata_finalist_1"

    mCNN2_1000 = load_model("./models/CNN_baseline_May16_e2e.h5")   # CNN on 128 1st dens
    mCNN2_1000._name = "cnn_merged_newdata_finalist_2"
    
    mCNN_1000 = load_model("cnn_noTest_20210516_model_445_0.998")   # CNN on 128 1st dens
    mCNN_1000._name = "cnn_merged_newdata_colab_finalist"

    
    # RNN no secondary
    mCNN_1000_w2nd = load_model("./models/CNN_baseline_May16_e2e_secondary.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    mCNN_1000_w2nd._name = "cnn_merged_newdata_w_secondary_finalist"

    
    mCNN1_1000.evaluate(X_test_1000e, Y_test_1000e)  # 96.15% 
    mCNN2_1000.evaluate(X_test_1000e, Y_test_1000e)  # 95.80 %  
    mCNN_1000.evaluate(X_test_1000e, Y_test_1000e)  # 95.57% 
    mCNN_1000_w2nd.evaluate(X_test_1000e_w2nd, Y_test_1000e_w2nd)  # 94.64 %


    # tf.keras.utils.plot_model(mCNN_1000_w2nd, show_shapes=True, to_file="C:/temp/test.png")
    # tf.keras.utils.plot_model(mCNN1_1000, show_shapes=True, to_file="C:/temp/test.png")

    # only the final layers... with a secondary structure CNN
    to_combine_last_layers = [
        (mCNN_1000, "dense_2", None),
        (mCNN1_1000, "dense_26", None),
        (mCNN2_1000, "dense_14", None),
        (mCNN_1000_w2nd, "dense_17", None)
    ]

    combined_models, data_train_ll, data_test_ll, data_access_ll = get_combined_features_from_models(
        to_combine_last_layers,
        [ X_new_train, X_new_train, X_new_train, X_new_train_w2nd ],
        [ Y_new_train, Y_new_train, Y_new_train, Y_new_train_w2nd ], 
        [ X_test_1000e, X_test_1000e, X_test_1000e, X_test_1000e_w2nd],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e, Y_test_1000e_w2nd],
        reverse_one_hot=False)
    
    
    cnn_combine_model = model_combination("combine_cnns_into_dense", data_train_ll[0][0].shape  )
    cnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_cnn_combine = [ModelCheckpoint(f'{cnn_combine_model.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='accuracy',
                                              mode='max',
                                              save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(patience=10)
                      ]
    cnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_cnn_combine = cnn_combine_model.fit(data_train_ll[0], 
                                                data_train_ll[1][0], 
                                                callbacks=callbacks_used_cnn_combine, 
                                                verbose=2, 
                                                epochs = 500, 
                                                batch_size=64)
    
    cnn_combine_model.evaluate(data_test_ll[0],data_test_ll[1][0]) # 96.27%
    
    
    # plot_history(history_cnn_combine)
    cnn_combine_model = load_model("combine_cnns_into_dense_model_350_0.999")
    cnn_combine_model = load_model("combine_cnns_into_dense_model_493_0.998")
    cnn_combine_model.evaluate(data_test_ll[0],data_test_ll[1][0]) # 96.27%   (anywhere between 96% and 96.5%)


    # only the final layers... no secondary structure CNN
    to_combine_last_layers_no2nd = [
        (mCNN_1000, "dense_2", None),
        (mCNN1_1000, "dense_26", None),
        (mCNN2_1000, "dense_14", None)
    ]

    combined_models_no2nd, data_train_ll_no2nd, data_test_ll_no2nd, data_access_ll_no2nd = get_combined_features_from_models(
        to_combine_last_layers_no2nd,
        [ X_new_train, X_new_train, X_new_train],
        [ Y_new_train, Y_new_train, Y_new_train], 
        [ X_test_1000e, X_test_1000e, X_test_1000e],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e],
        reverse_one_hot=False)
    
    
    cnn_combine_model_no2nd = model_combination("combine_cnns_into_dense_no2nd", data_train_ll_no2nd[0][0].shape  )
    cnn_combine_model_no2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_cnn_combine_no2nd = [ModelCheckpoint(f'{cnn_combine_model_no2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='accuracy',
                                              mode='max',
                                              save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(patience=10)
                      ]
    cnn_combine_model_no2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_cnn_combine_no2nd = cnn_combine_model_no2nd.fit(data_train_ll_no2nd[0], 
                                                data_train_ll_no2nd[1][0], 
                                                callbacks=callbacks_used_cnn_combine_no2nd, 
                                                verbose=2, 
                                                epochs = 500, 
                                                batch_size=64)
    
    cnn_combine_model_no2nd.evaluate(data_test_ll_no2nd[0],data_test_ll_no2nd[1][0]) # 96.5%

    plot_history(history_cnn_combine_no2nd)


    cnn_combine_model_no2nd = load_model("combine_cnns_into_dense_no2nd_model_276_0.998")
    cnn_combine_model_no2nd.evaluate(data_test_ll_no2nd[0],data_test_ll_no2nd[1][0]) # 96.5%   
    cnn_combine_model_no2nd = load_model("combine_cnns_into_dense_no2nd_model_271_0.998")
    cnn_combine_model_no2nd.evaluate(data_test_ll_no2nd[0],data_test_ll_no2nd[1][0]) # 96.5%   
    cnn_combine_model_no2nd = load_model("combine_cnns_into_dense_no2nd_model_142_0.997")
    cnn_combine_model_no2nd.evaluate(data_test_ll_no2nd[0],data_test_ll_no2nd[1][0]) # 96.39%

    

    # only the penultimate dense layers... no secondary structure CNN
    to_combine_penul_layers_no2nd = [
        (mCNN_1000, "dense_1", None),
        (mCNN1_1000, "dense_25", None),
        (mCNN2_1000, "dense_13", None)
    ]

    combined_models_penul_no2nd, data_train_ll_penul_no2nd, data_test_ll_penul_no2nd, data_access_ll_penul_no2nd = get_combined_features_from_models(
        to_combine_last_layers_no2nd,
        [ X_new_train, X_new_train, X_new_train],
        [ Y_new_train, Y_new_train, Y_new_train], 
        [ X_test_1000e, X_test_1000e, X_test_1000e],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e],
        reverse_one_hot=False # ,
        # normalize_X_func = tf.math.l2_normalize # this destroys predictive power!
        )
    
    
    cnn_combine_model_penul_no2nd = model_combination("combine_cnns_into_dense_penul_no2nd", data_train_ll_penul_no2nd[0][0].shape  )
    cnn_combine_model_penul_no2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_cnn_combine_penul_no2nd = [ModelCheckpoint(f'{cnn_combine_model_penul_no2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='accuracy',
                                              mode='max',
                                              save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(patience=10)
                      ]
    cnn_combine_model_penul_no2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_cnn_combine_penul_no2nd = cnn_combine_model_penul_no2nd.fit(data_train_ll_penul_no2nd[0], 
                                                data_train_ll_penul_no2nd[1][0], 
                                                callbacks=callbacks_used_cnn_combine_penul_no2nd, 
                                                verbose=2, 
                                                epochs = 1000, 
                                                batch_size=64)
    
    cnn_combine_model_penul_no2nd.evaluate(data_test_ll_penul_no2nd[0],data_test_ll_penul_no2nd[1][0]) # 96.15% (without normalization)
    plot_history(history_cnn_combine_penul_no2nd)

    # only the penultimate dense layers... with secondary structure CNN
    to_combine_penul_layers = [
        (mCNN_1000, "dense_1", None),
        (mCNN1_1000, "dense_25", None),
        (mCNN2_1000, "dense_13", None),
        (mCNN_1000_w2nd, "dense_16", None)
    ]

    combined_models_penul, data_train_ll_penul, data_test_ll_penul, data_access_ll_penul = get_combined_features_from_models(
        to_combine_penul_layers,
        [ X_new_train, X_new_train, X_new_train, X_new_train_w2nd],
        [ Y_new_train, Y_new_train, Y_new_train, Y_new_train_w2nd], 
        [ X_test_1000e, X_test_1000e, X_test_1000e, X_test_1000e_w2nd],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e, Y_test_1000e_w2nd],
        reverse_one_hot=False # ,
        # normalize_X_func = tf.math.l2_normalize # this destroys predictive power!
        )
    
    
    cnn_combine_model_penul = model_combination("combine_cnns_into_dense_penul", data_train_ll_penul[0][0].shape  )
    cnn_combine_model_penul.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_cnn_combine_penul = [ModelCheckpoint(f'{cnn_combine_model_penul.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                              save_weights_only=False,
                                              monitor='accuracy',
                                              mode='max',
                                              save_best_only=True),
                      tf.keras.callbacks.EarlyStopping(patience=10)
                      ]
    cnn_combine_model_penul.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_cnn_combine_penul = cnn_combine_model_penul.fit(data_train_ll_penul[0], 
                                                data_train_ll_penul[1][0], 
                                                callbacks=callbacks_used_cnn_combine_penul, 
                                                verbose=2, 
                                                epochs = 1000, 
                                                batch_size=64)
    
    cnn_combine_model_penul.evaluate(data_test_ll_penul[0],data_test_ll_penul[1][0]) # 96.04% (without normalization)
    plot_history(history_cnn_combine_penul)

    cnn_combine_model_penul = load_model("combine_cnns_into_dense_penul_model_653_0.998")
    cnn_combine_model_penul.evaluate(data_test_ll_penul[0],data_test_ll_penul[1][0]) # 96.39%


    




    # combine_cnns_into_dense_model_350_0.999
    
    tf.keras.utils.plot_model(cnn_combine_model, show_shapes=True)
    
