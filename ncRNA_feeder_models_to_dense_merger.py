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

def model_combination_X88(model_name, input_shape):
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
        Dense(88, activation='softmax')
    ], name=model_name)
    return model


if __name__ == "__main__":
    
    os.chdir("D:/Code/ncRNA")
    
    
    ## 88K data
    
    mRNNx88_1000 = load_model("./models/RNN_baseline_20May_88_4.h5", custom_objects=SeqWeightedAttention.get_custom_objects()) 
    X88_train, Y88_train, X88_test, Y88_test, X88_validation, Y88_validation = get88KData()
    Y88_test_cleaned = Y88_test[:,0:88].copy()
    Y88_train_cleaned = Y88_train[:,0:88].copy()
    Y88_validation_cleaned = Y88_validation[:,0:88].copy()
    
    # get_sp_pr_rc_f1_acc(mRNNx88_1000 , X88_test, Y88_test)
    y_pred = np.apply_along_axis(np.argmax, 1, mRNNx88_1000.predict(X88_test))
    y_true = np.apply_along_axis(np.argmax, 1, Y88_test)
    cmres = ConfusionMatrix(actual_vector=y_true,predict_vector=y_pred)
    print(cmres)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")   
    acc_data = mRNNx88_1000.evaluate(X88_test,Y88_test_cleaned)
    cm88 = confusion_matrix(y_pred,y_true)
    # np.trace(cm88)/np.sum(cm88)
    print(cmres.TNR_Macro, pr, rc, f1, np.trace(cm88)/np.sum(cm88))

    mCNNx88_1000 = load_model("./models/CNN_baseline_May18_e2e1000_256_88_50epochs.h5") 
    y_pred_Cx88 = np.apply_along_axis(np.argmax, 1, mCNNx88_1000.predict(X88_test))
    y_true_Cx88 = np.apply_along_axis(np.argmax, 1, Y88_test)
    # np.unique(y_true_Cx88)
    cmres_Cx88 = ConfusionMatrix(actual_vector=y_true_Cx88,predict_vector=y_pred_Cx88)
    print(cmres_Cx88)
    pr_Cx88, rc_Cx88, f1_Cx88, _ = precision_recall_fscore_support(y_true_Cx88, y_pred_Cx88, average="weighted")   
    cm88_Cx88 = confusion_matrix(y_pred_Cx88,y_true_Cx88)

    print(cmres_Cx88.TNR_Macro, pr_Cx88, rc_Cx88, f1_Cx88, np.trace(cm88_Cx88)/np.sum(cm88_Cx88))
    

    accCNNx88_data = mCNNx88_1000.evaluate(X88_test,Y88_test_cleaned)
    
    
    tf.keras.utils.plot_model(mRNNx88_1000, show_shapes=True)
    
    
# LD(A88)+LD(E88)
    to_combine_penul_x88_layers_no2nd_rNo2nd = [
        (mCNNx88_1000, "dense_1", None),
        (mRNNx88_1000,"dense_10", None)
    ]

    combined_models_x88_no2nd_rNo2nd_penul, data_train_x88_penul_no2nd_rNo2nd, data_test_x88_penul_no2nd_rNo2nd, data_access_x88_penul_no2nd_rNo2nd = get_combined_features_from_models(
        to_combine_penul_x88_layers_no2nd_rNo2nd,
        [ X88_train, X88_train],
        [ Y88_train_cleaned,  Y88_train_cleaned], 
        [ X88_test,  X88_test],
        [ Y88_test_cleaned,  Y88_test_cleaned],
        reverse_one_hot=False)

    rcnn_combine_x88_penul_models_no2nd_rNo2nd = model_combination_X88("combine_rcnns_x88_penul_no2nd_rNo2nd_into_dense", data_train_x88_penul_no2nd_rNo2nd[0][0].shape  )
    rcnn_combine_x88_penul_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_x88_penul_no2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_combine_x88_penul_models_no2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=20)
                        ]
    rcnn_combine_x88_penul_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_x88_combine_penul_no2nd_rNo2nd = rcnn_combine_x88_penul_models_no2nd_rNo2nd.fit(data_train_x88_penul_no2nd_rNo2nd[0], 
                                                  data_train_x88_penul_no2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_x88_penul_no2nd_rNo2nd, 
                                                  verbose=2, 
                                                  epochs = 500, 
                                                  batch_size=256)
      
    rcnn_combine_x88_penul_models_no2nd_rNo2nd.evaluate(data_test_penul_no2nd_rNo2nd[0],data_test_penul_no2nd_rNo2nd[1][0]) # 99.5%    

    rcnn_combine_x88_penul_models_no2nd_rNo2nd.save(f"./data/{rcnn_combine_penul_models_no2nd_rNo2nd.name}.h5")
    rcnn_combine_x88_penul_models_no2nd_rNo2nd = load_model("combine_rcnns_x88_penul_no2nd_rNo2nd_into_dense_model_003_0.977")
    get_sp_pr_rc_f1_acc(rcnn_combine_x88_penul_models_no2nd_rNo2nd,data_test_x88_penul_no2nd_rNo2nd[0],data_test_x88_penul_no2nd_rNo2nd[1][0])

    data_test_penul_no2nd_rNo2nd[0].shape
    data_test_penul_no2nd_rNo2nd[1][0].shape




    
    ###########    
    
    # 'new' data
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData(is500=False,
                                                                                                    include_secondary=False)
    X_train_1000e_w2nd, Y_train_1000e_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd, X_val_1000e_w2nd, Y_val_1000e_w2nd = getE2eData(is500=False,
                                                                                                    include_secondary=True)
    
    X_train_1000e_j2nd, Y_train_1000e_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd, X_val_1000e_j2nd, Y_val_1000e_j2nd = getE2eDataJustSecondary(isColab=False)
    
    
    X_test12CL_1000e, Y_test12CL_1000e = getTest12Data()
    
    
    # merge into a new train:
    X_new_train = np.concatenate( (X_train_1000e, X_val_1000e), axis=0 )
    Y_new_train = np.concatenate( (Y_train_1000e, Y_val_1000e), axis=0 )    
    
    X_new_train_w2nd = np.concatenate( (X_train_1000e_w2nd, X_val_1000e_w2nd), axis=0 )
    Y_new_train_w2nd = np.concatenate( (Y_train_1000e_w2nd, Y_val_1000e_w2nd), axis=0 )    
    
    X_new_train_j2nd = np.concatenate( (X_train_1000e_j2nd, X_val_1000e_j2nd), axis=0 )
    Y_new_train_j2nd = np.concatenate( (Y_train_1000e_j2nd, Y_val_1000e_j2nd), axis=0 )    
    
    
    ## CNN FEEDER MODELS
        
    # CNNs no secondary
    
        # A  #
    mCNN1_1000 = load_model("D:/GooDrive/data_papers/ncRNA/CNN_baseline_May16_e2e1000_256.h5") # CNN on 256 1st dens
    mCNN1_1000._name = "cnn_merged_newdata_finalist_1"
    
        # B  #
    mCNN2_1000 = load_model("D:/GooDrive/data_papers/ncRNA/CNN_baseline_May16_e2e.h5")   # CNN on 128 1st dens
    mCNN2_1000._name = "cnn_merged_newdata_finalist_2"
    
    mCNN_1000 = load_model("D:/GooDrive/data_papers/ncRNA/cnn_noTest_20210516_model_445_0.998")   # CNN on 128 1st dens
    mCNN_1000._name = "cnn_merged_newdata_colab_finalist"
    

        # C  #    
    # CNN w/ secondary
    mCNN_1000_w2nd = load_model("D:/GooDrive/data_papers/ncRNA/CNN_baseline_May16_e2e_secondary.h5")
    mCNN_1000_w2nd._name = "cnn_merged_newdata_w_secondary_finalist"
    
        # D  #    
    # CNN secondary only
    mCNN_1000_j2nd = load_model("D:/GooDrive/data_papers/ncRNA/cnn_j2nd_noTest_20210516_model_488_0.990")
    mCNN_1000_j2nd._name = "cnn_merged_newdata_j_secondary_finalist"
    
    
    mCNN1_1000.evaluate(X_test_1000e, Y_test_1000e)  # 96.15% 
    mCNN2_1000.evaluate(X_test_1000e, Y_test_1000e)  # 95.80 %  
    mCNN_1000.evaluate(X_test_1000e, Y_test_1000e)  # 95.57% 
    mCNN_1000_w2nd.evaluate(X_test_1000e_w2nd, Y_test_1000e_w2nd)  # 94.64 %
    mCNN_1000_j2nd.evaluate(X_test_1000e_j2nd, Y_test_1000e_j2nd)  # 72.96 %
    
    get_sp_pr_rc_f1_acc(mCNN1_1000,X_test_1000e,Y_test_1000e)
    get_sp_pr_rc_f1_acc(mCNN2_1000,X_test_1000e,Y_test_1000e)
    get_sp_pr_rc_f1_acc(mCNN_1000_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd)  
    get_sp_pr_rc_f1_acc(mCNN_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)  
    
    
    # on test12CL
    get_sp_pr_rc_f1_acc(mCNN1_1000,X_test12CL_1000e, Y_test12CL_1000e )
    get_sp_pr_rc_f1_acc(mCNN2_1000,X_test12CL_1000e, Y_test12CL_1000e )
    get_sp_pr_rc_f1_acc(mCNN_1000_w2nd,X_test12CL_1000e, Y_test12CL_1000e )
    get_sp_pr_rc_f1_acc(mCNN_1000_j2nd,X_test12CL_1000e, Y_test12CL_1000e )
    
    
   # get_sp_pr_rc_f1(mCNN1_1000, X_test_1000e, Y_test_1000e)
    

    ## CNN FEEDER MODELS
    from keras_self_attention import SeqWeightedAttention
    
    
        # E # 
    # RNN on the same data
    RNN_1000 = load_model("D:/GooDrive/data_papers/ncRNA/RNN_baseline_17May_180.h5", custom_objects=SeqWeightedAttention.get_custom_objects())   # RNN on 180ep 1000 ts
    RNN_1000._name = "rnn_merged_newdata_colab_finalist"
    
        # F #
    # RNN with secondary still needs to continue fit ## requires the validation in the fit data!
    RNN_1000_w2nd = load_model("D:/GooDrive/data_papers/ncRNA/rnn_newdata_w2nd_colab_continue_fit_epoch_22_accuracy_0.959.h5", custom_objects=SeqWeightedAttention.get_custom_objects())   # RNN on 180ep 1000 ts
    RNN_1000_w2nd._name = "rnn_newdata_w2nd_colab_continue_fit"
    
        # G # 
    # RNN on the e2e data just secondary
    RNN_1000_j2nd = load_model("D:/GooDrive/data_papers/ncRNA/rnn_newdata_j2nd_colab_continue_fit_epoch_25_accuracy_0.828.h5", custom_objects=SeqWeightedAttention.get_custom_objects())   # RNN on 180ep 1000 ts
    RNN_1000_j2nd._name = "rnn_newdata_j2nd_colab_continue_fit"
    
    
       # GX # 
    RNN2_1000_j2nd = load_model("./models/RNN_baseline_20May_JustSecondary_30.h5", custom_objects=SeqWeightedAttention.get_custom_objects())   
    RNN2_1000_j2nd._name = "rnn2_newdata_j2nd_colab_continue_fit"
    
    
    # RNN_1000.evaluate(X_test_1000e, Y_test_1000e)  # 95.45 %
    # RNN_1000_w2nd.evaluate(X_test_1000e_w2nd, Y_test_1000e_w2nd)  # 91.72 %
    # RNN_1000_j2nd.evaluate(X_test_1000e_j2nd, Y_test_1000e_j2nd)  # 61.04 %

    get_sp_pr_rc_f1_acc(RNN_1000, X_test_1000e, Y_test_1000e)
    get_sp_pr_rc_f1_acc(RNN_1000_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd)
    get_sp_pr_rc_f1_acc(RNN_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)    
    
    # get_sp_pr_rc_f1_acc(RNN2_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)    
    # get_sp_pr_rc_f1_acc(RNN_1000, X_test12CL_1000e, Y_test12CL_1000e)
   
    
    #### confusion matrices

    def num_to_label(inputNum):
        if inputNum==0:
            return '5S_rRNA'
        if inputNum==1:
            return '5_8S_rRNA'
        if inputNum==2:
            return 'tRNA'
        if inputNum==3:
            return 'ribozyme'
        if inputNum==4:
            return 'CD-box'
        if inputNum==5:
            return 'miRNA'
        if inputNum==6:
            return 'Intron_gpI'
        if inputNum==7:
            return 'HACA-box'
        if inputNum==8:
            return '5S_rRNA'
        if inputNum==9:
            return 'riboswitch'
        if inputNum==10:
            return 'IRES'
        if inputNum==11:
            return 'leader'
        if inputNum==12:
            return 'scaRNA'
        return 'NA'

    cnnA_CM, cnnA_pred, cnnA_true =  get_confusion_matrix_classification(mCNN1_1000, X_test_1000e, Y_test_1000e)
    pd.DataFrame(cnnA_CM).to_csv("cnnA_CM.csv")

    cnnB_CM, cnnB_pred, cnnB_true =  get_confusion_matrix_classification(mCNN2_1000, X_test_1000e, Y_test_1000e)
    pd.DataFrame(cnnB_CM).to_csv("cnnB_CM.csv")

    cnnC_CM, cnnC_pred, cnnC_true =  get_confusion_matrix_classification(mCNN_1000_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd)
    pd.DataFrame(cnnC_CM).to_csv("cnnC_CM.csv")

    cnnD_CM, cnnD_pred, cnnD_true =  get_confusion_matrix_classification(mCNN_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)
    pd.DataFrame(cnnD_CM).to_csv("cnnD_CM.csv")

    rnnE_CM, rnnE_pred, rnnE_true =  get_confusion_matrix_classification(RNN_1000, X_test_1000e, Y_test_1000e)    
    pd.DataFrame(rnnE_CM).to_csv("rnnE_CM.csv")

    rnnF_CM, rnnF_pred, rnnF_true =  get_confusion_matrix_classification(RNN_1000_w2nd, X_test_1000e_w2nd, Y_test_1000e_w2nd)    
    pd.DataFrame(rnnF_CM).to_csv("rnnF_CM.csv")

    rnnG1_CM, rnnG1_pred, rnnG1_true =  get_confusion_matrix_classification(RNN_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)    
    pd.DataFrame(rnnG1_CM).to_csv("rnnG1_CM.csv")
    
    component_test_frame = pd.DataFrame( { "TrueLabel" : list(map(num_to_label,cnnA_true)),
                    "TrueValue" : cnnA_true,
                    "A" : np.array(cnnA_pred==cnnA_true, dtype=int),
                    "B" : np.array(cnnB_pred==cnnB_true, dtype=int),
                    "C" : np.array(cnnC_pred==cnnC_true, dtype=int),
                    "D" : np.array(cnnD_pred==cnnD_true, dtype=int),
                    "E" : np.array(rnnE_pred==rnnE_true, dtype=int),
                    "F" : np.array(rnnF_pred==rnnF_true, dtype=int),
                    "G" : np.array(rnnG1_pred==rnnG1_true, dtype=int)                    
                    } )

    component_test_frame.to_csv("component_test_frame.csv",index=False)    
    
        
    # rnnG_CM, rnnG_pred, rnnG_true =  get_confusion_matrix_classification(RNN2_1000_j2nd, X_test_1000e_j2nd, Y_test_1000e_j2nd)    
    # pd.DataFrame(rnnG_CM).to_csv("rnnG_CM.csv")


    
  # O(A)+O(B)+O(D)
    
    to_combine_last_layers_no2nd_j2nd = [
        (mCNN1_1000, "dense_26", None),
        (mCNN2_1000, "dense_14", None),
        (mCNN_1000_j2nd,"dense_5", None)
    ]

    combined_models_no2nd_j2nd, data_train_ll_no2nd_j2nd, data_test_ll_no2nd_j2nd, data_access_ll_no2nd_j2nd = get_combined_features_from_models(
        to_combine_last_layers_no2nd_j2nd,
        [ X_new_train, X_new_train, X_new_train_j2nd],
        [ Y_new_train, Y_new_train, Y_new_train_j2nd], 
        [ X_test_1000e, X_test_1000e, X_test_1000e_j2nd],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e_j2nd],
        reverse_one_hot=False)
    
        cnn_combine_model = model_combination("combine_cnns_no2nd_j2nd_into_dense", data_train_ll_no2nd_j2nd[0][0].shape  )
        cnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
        callbacks_used_cnn_combine = [ModelCheckpoint(f'{cnn_combine_model.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                    save_weights_only=False,
                                                    monitor='accuracy',
                                                    mode='max',
                                                    save_best_only=True),
                            tf.keras.callbacks.EarlyStopping(patience=10)
                            ]
        cnn_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history_cnn_combine = cnn_combine_model.fit(data_train_ll_no2nd_j2nd[0], 
                                                      data_train_ll_no2nd_j2nd[1][0], 
                                                      callbacks=callbacks_used_cnn_combine, 
                                                      verbose=2, 
                                                      epochs = 500, 
                                                      batch_size=64)
          
        cnn_combine_model.evaluate(data_test_ll_no2nd_j2nd[0],data_test_ll_no2nd_j2nd[1][0]) # 96.62%
    #     cnn_combine_model.save(f"./models/{cnn_combine_model.name}.h5")    
    
    # cnn_combine_model = load_model("./models/combine_cnns_no2nd_j2nd_into_dense.h5")
    cnn_combine_model = load_model("./models/combine_cnns_no2nd_j2nd_into_dense_model_194_0.998")
    
    get_sp_pr_rc_f1_acc(cnn_combine_model,data_test_ll_no2nd_j2nd[0],data_test_ll_no2nd_j2nd[1][0])
    
    
    
# O(A) + O(B) + O(E)
    to_combine_last_layers_no2nd_j2nd_rNo2nd = [
        (mCNN1_1000, "dense_26", None),
        (mCNN_1000_j2nd,"dense_5", None),
        (RNN_1000,"dense_3", None)
    ]
    
    combined_models_no2nd_j2nd_rNo2nd, data_train_ll_no2nd_j2nd_rNo2nd, data_test_ll_no2nd_j2nd_rNo2nd, data_access_ll_no2nd_j2nd_rNo2nd = get_combined_features_from_models(
        to_combine_last_layers_no2nd_j2nd_rNo2nd,
        [ X_new_train, X_new_train_j2nd, X_new_train],
        [ Y_new_train, Y_new_train_j2nd, Y_new_train], 
        [ X_test_1000e, X_test_1000e_j2nd, X_test_1000e],
        [ Y_test_1000e, Y_test_1000e_j2nd, Y_test_1000e],
        reverse_one_hot=False)
    
    rcnn_combine_models_no2nd_j2nd_rNo2nd = model_combination("combine_rcnns_no2nd_j2nd_rNo2nd_into_dense", data_train_ll_no2nd_j2nd_rNo2nd[0][0].shape  )
    rcnn_combine_models_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_no2nd_j2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_combine_models_no2nd_j2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=10)
                        ]
    rcnn_combine_models_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_combine_no2nd_j2nd_rNo2nd = rcnn_combine_models_no2nd_j2nd_rNo2nd.fit(data_train_ll_no2nd_j2nd_rNo2nd[0], 
                                                  data_train_ll_no2nd_j2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_no2nd_j2nd_rNo2nd, 
                                                  verbose=2, 
                                                  epochs = 250, 
                                                  batch_size=64)
      
    rcnn_combine_models_no2nd_j2nd_rNo2nd.evaluate(data_test_ll_no2nd_j2nd_rNo2nd[0],data_test_ll_no2nd_j2nd_rNo2nd[1][0]) # 96.39%    
    # rcnn_combine_models_no2nd_j2nd_rNo2nd.save(f"./data/{rcnn_combine_models_no2nd_j2nd_rNo2nd.name}.h5")
    
    get_sp_pr_rc_f1_acc(rcnn_combine_models_no2nd_j2nd_rNo2nd,data_test_ll_no2nd_j2nd_rNo2nd[0],data_test_ll_no2nd_j2nd_rNo2nd[1][0])
    
    
# LD(A)+LD(B)+LD(D)
    to_combine_penul_layers_no2nd_j2nd = [
        (mCNN1_1000, "dense_25", None),
        (mCNN2_1000, "dense_13", None),
        (mCNN_1000_j2nd,"dense_4", None)
    ]

    combined_models_penul_no2nd_j2nd, data_train_ll_penul_no2nd_j2nd, data_test_ll_penul_no2nd_j2nd, data_access_ll_penul_no2nd_j2nd = get_combined_features_from_models(
        to_combine_penul_layers_no2nd_j2nd,
        [ X_new_train, X_new_train, X_new_train_j2nd],
        [ Y_new_train, Y_new_train, Y_new_train_j2nd], 
        [ X_test_1000e, X_test_1000e, X_test_1000e_j2nd],
        [ Y_test_1000e, Y_test_1000e, Y_test_1000e_j2nd],
        reverse_one_hot=False)    
    
        # cnn_penul_combine_model = model_combination("combine_cnns_penul_no2nd_j2nd_into_dense", data_train_ll_penul_no2nd_j2nd[0][0].shape  )
        # cnn_penul_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
        # callbacks_used_cnn_combine_penul = [ModelCheckpoint(f'{cnn_penul_combine_model.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
        #                                     save_weights_only=False,
        #                                     monitor='accuracy',
        #                                     mode='max',
        #                                     save_best_only=True),
        #             tf.keras.callbacks.EarlyStopping(patience=10)
        #             ]
        # cnn_penul_combine_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        # history_cnn_penul_combine = cnn_penul_combine_model.fit(data_train_ll_penul_no2nd_j2nd[0], 
        #                                       data_train_ll_penul_no2nd_j2nd[1][0], 
        #                                       callbacks=callbacks_used_cnn_combine_penul, 
        #                                       verbose=2, 
        #                                       epochs = 250, 
        #                                       batch_size=64)
          
        # cnn_penul_combine_model.evaluate(data_test_ll_penul_no2nd_j2nd[0],data_test_ll_penul_no2nd_j2nd[1][0]) # 96.27%
        # cnn_penul_combine_model.save(f"{cnn_penul_combine_model.name}.h5")
        
    cnn_penul_combine_model = load_model("./models/combine_cnns_penul_no2nd_j2nd_into_dense.h5")
        
    cnn_penul_combine_model = load_model("/content/combine_cnns_penul_no2nd_j2nd_into_dense_model_250_0.997")    
    get_sp_pr_rc_f1_acc(cnn_penul_combine_model,data_test_ll_penul_no2nd_j2nd[0],data_test_ll_penul_no2nd_j2nd[1][0])
    
    
# LD(A) + LD(B) + LD(E)
    to_combine_penul_layers_no2nd_j2nd_rNo2nd = [
        (mCNN1_1000, "dense_25", None),
        (mCNN_1000_j2nd,"dense_4", None),
        (RNN_1000,"dense_2", None)
    ]

    combined_models_penul_no2nd_j2nd_rNo2nd, data_train_ll_penul_no2nd_j2nd_rNo2nd, data_test_ll_penul_no2nd_j2nd_rNo2nd, data_access_ll_penul_no2nd_j2nd_rNo2nd = get_combined_features_from_models(
        to_combine_penul_layers_no2nd_j2nd_rNo2nd,
        [ X_new_train, X_new_train_j2nd, X_new_train],
        [ Y_new_train, Y_new_train_j2nd, Y_new_train], 
        [ X_test_1000e, X_test_1000e_j2nd, X_test_1000e],
        [ Y_test_1000e, Y_test_1000e_j2nd, Y_test_1000e],
        reverse_one_hot=False)
    
    
    rcnn_penul_combine_no2nd_j2nd_rNo2nd = model_combination("combine_rcnns_penul_no2nd_j2nd_rNo2nd_into_dense", data_train_ll_penul_no2nd_j2nd_rNo2nd[0][0].shape  )
    rcnn_penul_combine_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_penul = [ModelCheckpoint(f'{rcnn_penul_combine_no2nd_j2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=10)
                        ]
    rcnn_penul_combine_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_penul_combine = rcnn_penul_combine_no2nd_j2nd_rNo2nd.fit(data_train_ll_penul_no2nd_j2nd_rNo2nd[0], 
                                                  data_train_ll_penul_no2nd_j2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_penul, 
                                                  verbose=2, 
                                                  epochs = 250, 
                                                  batch_size=64)      
    rcnn_penul_combine_no2nd_j2nd_rNo2nd.evaluate(data_test_ll_penul_no2nd_j2nd_rNo2nd[0],data_test_ll_penul_no2nd_j2nd_rNo2nd[1][0]) # 97.32%

    rcnn_penul_combine_no2nd_j2nd_rNo2nd.save(f"./data/{rcnn_penul_combine_no2nd_j2nd_rNo2nd.name}.h5")
    get_sp_pr_rc_f1_acc(rcnn_penul_combine_no2nd_j2nd_rNo2nd,data_test_ll_penul_no2nd_j2nd_rNo2nd[0],data_test_ll_penul_no2nd_j2nd_rNo2nd[1][0])
    
    
# O(A)+O(E)
     to_combine_last_layers_no2nd_rNo2nd = [
            (mCNN1_1000, "dense_26", None),
            (RNN_1000,"dense_3", None)
        ]
    
    combined_models_no2nd_rNo2nd, data_train_ll_no2nd_rNo2nd, data_test_ll_no2nd_rNo2nd, data_access_ll_no2nd_rNo2nd = get_combined_features_from_models(
        to_combine_last_layers_no2nd_rNo2nd,
        [ X_new_train, X_new_train],
        [ Y_new_train,  Y_new_train], 
        [ X_test_1000e,  X_test_1000e],
        [ Y_test_1000e,  Y_test_1000e],
        reverse_one_hot=False)
    
    # rcnn_combine_models_no2nd_rNo2nd = model_combination("combine_rcnns_ll_no2nd_rNo2nd_into_dense", data_train_ll_no2nd_rNo2nd[0][0].shape  )
    # rcnn_combine_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    # callbacks_used_rcnn_combine_no2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_combine_models_no2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
    #                                             save_weights_only=False,
    #                                             monitor='accuracy',
    #                                             mode='max',
    #                                             save_best_only=True),
    #                     tf.keras.callbacks.EarlyStopping(patience=10)
    #                     ]
    # rcnn_combine_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # history_rcnn_combine_no2nd_rNo2nd = rcnn_combine_models_no2nd_rNo2nd.fit(data_train_ll_no2nd_rNo2nd[0], 
    #                                               data_train_ll_no2nd_rNo2nd[1][0], 
    #                                               callbacks=callbacks_used_rcnn_combine_no2nd_rNo2nd, 
    #                                               verbose=2, 
    #                                               epochs = 250, 
    #                                               batch_size=64)    
    # rcnn_combine_models_no2nd_rNo2nd.save(f"./models/{rcnn_combine_models_no2nd_rNo2nd.name}.h5")
    
    rcnn_combine_models_no2nd_rNo2nd = load_model(f"./data/combine_rcnns_ll_no2nd_rNo2nd_into_dense.h5")
    
    rcnn_combine_models_no2nd_rNo2nd.evaluate(data_test_ll_no2nd_rNo2nd[0],data_test_ll_no2nd_rNo2nd[1][0]) # 96.50%
    get_sp_pr_rc_f1_acc(rcnn_combine_models_no2nd_rNo2nd,data_test_ll_no2nd_rNo2nd[0],data_test_ll_no2nd_rNo2nd[1][0])

# LD(A)+LD(E)
    to_combine_penul_layers_no2nd_rNo2nd = [
        (mCNN1_1000, "dense_25", None),
        (RNN_1000,"dense_2", None)
    ]

    combined_models_no2nd_rNo2nd_penul, data_train_penul_no2nd_rNo2nd, data_test_penul_no2nd_rNo2nd, data_access_penul_no2nd_rNo2nd = get_combined_features_from_models(
        to_combine_penul_layers_no2nd_rNo2nd,
        [ X_new_train, X_new_train],
        [ Y_new_train,  Y_new_train], 
        [ X_test_1000e,  X_test_1000e],
        [ Y_test_1000e,  Y_test_1000e],
        reverse_one_hot=False)

    
    rcnn_combine_penul_models_no2nd_rNo2nd = model_combination("combine_rcnns_penul_no2nd_rNo2nd_into_dense", data_train_penul_no2nd_rNo2nd[0][0].shape  )
    rcnn_combine_penul_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_penul_no2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_combine_penul_models_no2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=10)
                        ]
    rcnn_combine_penul_models_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_combine_penul_no2nd_rNo2nd = rcnn_combine_penul_models_no2nd_rNo2nd.fit(data_train_penul_no2nd_rNo2nd[0], 
                                                  data_train_penul_no2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_penul_no2nd_rNo2nd, 
                                                  verbose=2, 
                                                  epochs = 250, 
                                                  batch_size=64)
      
    rcnn_combine_penul_models_no2nd_rNo2nd.evaluate(data_test_penul_no2nd_rNo2nd[0],data_test_penul_no2nd_rNo2nd[1][0]) # 97.20%    

    rcnn_combine_penul_models_no2nd_rNo2nd.save(f"./data/{rcnn_combine_penul_models_no2nd_rNo2nd.name}.h5")
    get_sp_pr_rc_f1_acc(rcnn_combine_penul_models_no2nd_rNo2nd,data_test_penul_no2nd_rNo2nd[0],data_test_penul_no2nd_rNo2nd[1][0])

# LD(A)+LD(E)+O(A)+O(E)
    
    to_combine_last2_layers_no2nd_rNo2nd = [
        (mCNN1_1000, "dense_25", None),
        (mCNN1_1000, "dense_26", None),
        (RNN_1000,"dense_2", None),
        (RNN_1000,"dense_3", None)
    ]

    combined_models_last2_no2nd_rNo2nd, data_train_last2_no2nd_rNo2nd, data_test_last2_no2nd_rNo2nd, data_access_last2_no2nd_rNo2nd = get_combined_features_from_models(
        to_combine_last2_layers_no2nd_rNo2nd,
        [ X_new_train, X_new_train, X_new_train, X_new_train],
        [ Y_new_train,  Y_new_train, Y_new_train,  Y_new_train], 
        [ X_test_1000e,  X_test_1000e, X_test_1000e,  X_test_1000e],
        [ Y_test_1000e,  Y_test_1000e, Y_test_1000e,  Y_test_1000e],
        reverse_one_hot=False)

    rcnn_last2_combine_model_no2nd_rNo2nd = model_combination("combine_rcnns_last2_no2nd_rNo2nd_into_dense", data_train_last2_no2nd_rNo2nd[0][0].shape  )
    rcnn_last2_combine_model_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_last2_no2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_last2_combine_model_no2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=10)
                        ]
    rcnn_last2_combine_model_no2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_combine_last2_no2nd_rNo2nd = rcnn_last2_combine_model_no2nd_rNo2nd.fit(data_train_last2_no2nd_rNo2nd[0], 
                                                  data_train_last2_no2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_last2_no2nd_rNo2nd, 
                                                  verbose=2, 
                                                  epochs = 250, 
                                                  batch_size=64)
      
    rcnn_last2_combine_model_no2nd_rNo2nd.evaluate(data_test_last2_no2nd_rNo2nd[0],data_test_last2_no2nd_rNo2nd[1][0]) # 97.20%

    rcnn_last2_combine_model_no2nd_rNo2nd.save(f"./data/{rcnn_last2_combine_model_no2nd_rNo2nd.name}.h5")
    get_sp_pr_rc_f1_acc(rcnn_last2_combine_model_no2nd_rNo2nd, data_test_last2_no2nd_rNo2nd[0],data_test_last2_no2nd_rNo2nd[1][0])


# LD(A)+LD(D)+LD(E)+O(A)+0(D)+O(E)

    to_combine_last2_layers_no2nd_j2nd_rNo2nd = [
        (mCNN1_1000, "dense_25", None),
        (mCNN1_1000, "dense_26", None),
        (mCNN_1000_j2nd,"dense_4", None),
        (mCNN_1000_j2nd,"dense_5", None),
        (RNN_1000,"dense_2", None),
        (RNN_1000,"dense_3", None)
    ]

    combined_models_last2_no2nd_j2nd_rNo2nd, data_train_last2_no2nd_j2nd_rNo2nd, data_test_last2_no2nd_j2nd_rNo2nd, data_access_last2_no2nd_j2nd_rNo2nd = get_combined_features_from_models(
        to_combine_last2_layers_no2nd_j2nd_rNo2nd,
        [ X_new_train, X_new_train, X_new_train_j2nd, X_new_train_j2nd, X_new_train, X_new_train],
        [ Y_new_train,  Y_new_train, Y_new_train_j2nd, Y_new_train_j2nd, Y_new_train, Y_new_train], 
        [ X_test_1000e,  X_test_1000e, X_test_1000e_j2nd, X_test_1000e_j2nd, X_test_1000e, X_test_1000e],
        [ Y_test_1000e,  Y_test_1000e, Y_test_1000e_j2nd, Y_test_1000e_j2nd, Y_test_1000e, Y_test_1000e],
        reverse_one_hot=False)


    rcnn_last2_combine_model_no2nd_j2nd_rNo2nd = model_combination("combine_rcnns_last2_no2nd_j2nd_rNo2nd_into_dense", data_train_last2_no2nd_j2nd_rNo2nd[0][0].shape  )
    rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])  
    callbacks_used_rcnn_combine_last2_no2nd_j2nd_rNo2nd = [ModelCheckpoint(f'{rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.name}' + '_model_{epoch:03d}_{accuracy:0.3f}',
                                                save_weights_only=False,
                                                monitor='accuracy',
                                                mode='max',
                                                save_best_only=True),
                        tf.keras.callbacks.EarlyStopping(patience=10)
                        ]
    rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history_rcnn_combine_last2_no2nd_j2nd_rNo2nd = rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.fit(data_train_last2_no2nd_j2nd_rNo2nd[0], 
                                                  data_train_last2_no2nd_j2nd_rNo2nd[1][0], 
                                                  callbacks=callbacks_used_rcnn_combine_last2_no2nd_j2nd_rNo2nd, 
                                                  verbose=2, 
                                                  epochs = 250, 
                                                  batch_size=64)
      
    rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.evaluate(data_test_last2_no2nd_j2nd_rNo2nd[0],data_test_last2_no2nd_j2nd_rNo2nd[1][0]) 
    
    rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.save(f"./data/{rcnn_last2_combine_model_no2nd_j2nd_rNo2nd.name}.h5")
    get_sp_pr_rc_f1_acc(rcnn_last2_combine_model_no2nd_j2nd_rNo2nd, data_test_last2_no2nd_j2nd_rNo2nd[0],data_test_last2_no2nd_j2nd_rNo2nd[1][0])




data_access_last2_no2nd_j2nd_rNo2nd.keys()


data_access_last2_no2nd_j2nd_rNo2nd["cnn_merged_newdata_finalist_1"]["dense_25"].keys()


features_seq_cnn = data_access_last2_no2nd_j2nd_rNo2nd["cnn_merged_newdata_finalist_1"]["dense_25"]["Test"]["X"]
features_j2nd_cnn = data_access_last2_no2nd_j2nd_rNo2nd["cnn_merged_newdata_j_secondary_finalist"]["dense_4"]["Test"]["X"]
features_seq_rnn = data_access_last2_no2nd_j2nd_rNo2nd["rnn_merged_newdata_colab_finalist"]["dense_2"]["Test"]["X"]
 features_seq_cnn = pd.DataFrame(features_seq_cnn)
 features_seq_cnn.columns= [ f"CNN_LD_{x}" for x in range(features_seq_cnn.shape[1])]

 features_j2nd_cnn = pd.DataFrame(features_j2nd_cnn)
 features_j2nd_cnn.columns= [ f"CNNJ2_LD_{x}" for x in range(features_j2nd_cnn.shape[1])]

 features_seq_rnn = pd.DataFrame(features_seq_rnn)
 features_seq_rnn.columns= [ f"RNN_LD_{x}" for x in range(features_seq_rnn.shape[1])]


from heatmap import heatmap, corrplot


features_j2nd_cnn.shape


features_3_mods = pd.concat( (features_seq_cnn,features_j2nd_cnn,features_seq_rnn), axis=1 )

features_2_mods = pd.concat( (features_seq_cnn,features_j2nd_cnn), axis=1 )

features_2_crnn_mods = pd.concat( (features_seq_cnn,features_seq_rnn), axis=1 )

corrplot(features_2_mods.corr())
corrplot(features_3_mods.corr())


corrplot(features_seq_cnn.corr())

import seaborn as sns

ax = sns.heatmap(
    features_2_mods.corr(), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.color_palette("coolwarm",10),
    # cmap=sns.color_palette("RdBu",200),
    square=True
)
ax.tick_params(left=False, bottom=False)
ax.set(xticklabels=[],yticklabels=[])
ax.set(xlabel=None,ylabel=None)


ax = sns.heatmap(
    features_2_crnn_mods.corr(), 
    vmin=-1, vmax=1, center=0,
    cmap=sns.color_palette("coolwarm",10),
    # cmap=sns.color_palette("hot",10),
    # cmap=sns.color_palette("RdBu",200),
    square=True
)
ax.tick_params(left=False, bottom=False)
ax.set(xticklabels=[],yticklabels=[])
ax.set(xlabel=None,ylabel=None)


plt.imshow(np.triu(np.array(features_seq_cnn.corr())), cmap='hot', interpolation='nearest')
plt.show()


np.mean([ x for x in np.triu(np.array(features_seq_cnn.corr())).flatten().tolist() if np.abs(x)!=0. ])
np.std([ x for x in np.triu(np.array(features_seq_cnn.corr())).flatten().tolist() if np.abs(x)!=0. ])

np.mean([ x for x in np.triu(np.array(features_j2nd_cnn.corr())).flatten().tolist() if np.abs(x)!=0. ])
np.std([ x for x in np.triu(np.array(features_j2nd_cnn.corr())).flatten().tolist() if np.abs(x)!=0. ])

np.mean([ x for x in np.triu(np.array(features_seq_rnn.corr())).flatten().tolist() if np.abs(x)!=0. ])
np.std([ x for x in np.triu(np.array(features_seq_rnn.corr())).flatten().tolist() if np.abs(x)!=0. ])


features_2_crnn_mods.shape


features_2_crnn_mods.shape

np_features_2_crnn_mods = np.array(features_2_crnn_mods)

np.mean([ x for x in np.triu(np.array(pd.DataFrame(np_features_2_crnn_mods[0:64,64:128]).corr())).flatten().tolist() if np.abs(x)!=0. ])
np.std([ x for x in np.triu(np.array(pd.DataFrame(np_features_2_crnn_mods[0:64,64:128]).corr())).flatten().tolist() if np.abs(x)!=0. ])






ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    size=0,
    horizontalalignment='right'
);
ax.set_yticklabels(
    ax.get_yticklabels(),
    size=0,
    horizontalalignment='right'
);





    
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
    
