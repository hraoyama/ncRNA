import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import site
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

    # import time
    # start = time.time()
    # t1 = [ x for x in a1a[:10] if x not in a2a ]
    # end = time.time()
    # print(end - start)


site.addsitedir("D:/Code/ncRNA")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import getData, coShuffled_vectors, reverse_tensor, compile_and_fit_model_with_tb, getE2eData, sparse_setdiff

def confirm_no_duplicates_in_data(Xs):
    # check no duplicates in the data
    # should all be zero
    number_of_duplicates_within_each_array = [len(x) - len(np.unique(x, axis=0)) for x in Xs]
    assert (np.all([x == 0 for x in number_of_duplicates_within_each_array]))
    all_X = np.concatenate(Xs, axis=0)
    assert ((len(all_X) - len(np.unique(all_X, axis=0))) == 0)
    pass


if __name__ == "__main__":
    os.chdir("D:/Code/ncRNA")
    
    # 'old' data
    X_train_1000, Y_train_1000, X_test_1000, Y_test_1000, X_val, Y_val = getData(is500=False, shuffle=True, ise2e=False)
    confirm_no_duplicates_in_data((X_train_1000, X_test_1000))  # X_val is None
    # 'new' data
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData(is500=False,
                                                                                                    include_secondary=False)
    confirm_no_duplicates_in_data((X_train_1000e, X_test_1000e, X_val_1000e))

    # now let's get some validation data that was not used for creating these saved models
    X_test_for_all, idxs_kept = sparse_setdiff(X_train_1000e.copy(),X_train_1000.numpy().copy())
    Y_test_for_all = Y_train_1000e[idxs_kept,:]
    X_test_for_all, idxs_kept = sparse_setdiff(X_test_for_all.copy(),X_test_1000.numpy().copy())
    Y_test_for_all = Y_test_for_all[idxs_kept,:]
    # validation is none for the original so this should be it...

    # np.save("D:/Code/ncRNA/data/X_test_for_all.npy", X_test_for_all)
    # np.save("D:/Code/ncRNA/data/Y_test_for_all.npy", Y_test_for_all)


    # CNN no secondary - great
    mCNN2_1000 = load_model("./models/CNN_baseline.h5")
    mCNN2_1000.evaluate(X_test_1000, Y_test_1000)  # 93.51%
    mCNN2_1000.evaluate(X_train_1000e, Y_train_1000e)  # 98.02% (should be the training accuracy)
    mCNN2_1000.evaluate(X_test_1000e, Y_test_1000e)  # 97.55% (should be test set used to fit model?)
    mCNN2_1000.evaluate(X_val_1000e, Y_val_1000e)  # 97.20% (the new validation data)    
    mCNN2_1000.evaluate(X_test_for_all, Y_test_for_all)  # 94.21% (the real validation data)
    
    # RNN no secondary - great
    mRNN_1000l = load_model("./models/PureRNN_no_sec.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    mRNN_1000l.evaluate(X_train_1000e, Y_train_1000e)  # 97.24% (should be the training accuracy)
    mRNN_1000l.evaluate(X_test_1000e, Y_test_1000e)  # 97.79% (should be the test accuracy - used to fit the model?)
    mRNN_1000l.evaluate(X_val_1000e, Y_val_1000e)  # 96.15% (the new validation data)
    mRNN_1000l.evaluate(X_test_for_all, Y_test_for_all)  # 92.93% (the real validation data)
    

    # bad => need more data and overfits with hyperband ...
    X_train_500, Y_train_500, X_test_500, Y_test_500, X_val, Y_val = getData(is500=True, shuffle=True, ise2e=False)
    mCNNt_500 = load_model("./models/CNN_tuned_hp_500.h5")
    mCNNt_500.evaluate(X_train_500, Y_train_500)  # 94.92%
    mCNNt_500.evaluate(X_test_500, Y_test_500)  # 78.01%
    mCNN1_500 = load_model("./models/CNN_Tenth_Fold_New_Model_500_8_updated_large_20210426.h5")
    mCNN1_500.evaluate(X_train_500, Y_train_500)  # 71.99%
    mCNN1_500.evaluate(X_test_500, Y_test_500)  # 61.39%

    # rnn no secondary, 45 epochs, not good, too few epochs?
    mRNN_1000 = load_model("./models/RNN_3stacked_45_epochs.h5",
                           custom_objects=SeqWeightedAttention.get_custom_objects())
    mRNN_1000.evaluate(X_train_1000e, Y_train_1000e)  # 89.17%
    mRNN_1000.evaluate(X_test_1000e, Y_test_1000e)  # 89.86%
    mRNN_1000.evaluate(X_val_1000e, Y_val_1000e)  # 87.51%

    # rnn with secondary - great but just as good (if not better) without secondary
    mRNN_secondary = load_model("./models/RNN_baseline_150.h5",
                                custom_objects=SeqWeightedAttention.get_custom_objects())
    X_train_1136e, Y_train_1136e, X_test_1136e, Y_test_1136e, X_val_1136e, Y_val_1136e = getE2eData(is500=False,
                                                                                                    include_secondary=True)
    mRNN_secondary.evaluate(X_train_1136e, Y_train_1136e)  # 96.52%
    mRNN_secondary.evaluate(X_test_1136e, Y_test_1136e)   # 97.20%
    mRNN_secondary.evaluate(X_val_1136e, Y_val_1136e)    # 96.62%
