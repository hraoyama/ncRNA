import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import  Dense, Flatten, Activation, Dropout, Embedding, Conv1D, MaxPooling1D, GRU
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute,Reshape, Lambda, RepeatVector, Input,Multiply
from tensorflow.keras.layers import  Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import SimpleRNN, GRU, LeakyReLU
from tensorflow.keras.layers import Concatenate, Average 
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import  Bidirectional
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
import warnings
from functools import partial
from multiprocessing import Pool, cpu_count
from tensorflow.keras.layers import  Dense, Flatten, Activation, Dropout, Embedding, Conv1D, Conv2D, MaxPooling2D, MaxPooling1D, Concatenate, BatchNormalization, GaussianNoise
from tensorflow.keras.layers import LSTM, TimeDistributed, Permute, Reshape, Lambda, RepeatVector, Input, Multiply, SimpleRNN, GRU, LeakyReLU
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from tensorflow.keras import regularizers
import os
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
import kerastuner as kt
from kerastuner import HyperModel
from tensorflow.summary import create_file_writer
from functools import partial
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

"""Data Preparation"""
warnings.filterwarnings("ignore")  # ignore DeprecationWarnings from tensorflow
set_default_float('float64')
os.chdir("D:/papers/RNA_NN_GP_ARD/ncRFP/ncRFP_Model/")

# load the input data
INPUT_DIM = 8    # 
TIME_STEPS = 500  # The step of RNN

def coShuffled_vectors(X,Y):
    if tf.shape(X)[0] ==  tf.shape(Y)[0]:
        test_idxs = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_test_idxs = tf.random.shuffle(test_idxs)
        return ( tf.gather(X, shuffled_test_idxs), tf.gather(Y, shuffled_test_idxs) )
    else:
        raise ValueError(f"0-dimension has to be the same {tf.shape(X)[0]} != {tf.shape(Y)[0]}")

# data extraction
def getData(is500=True, shuffle=False, validation_split=None):
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
    
    if shuffle:
        X_train, Y_train = coShuffled_vectors(X_train, Y_train)
        X_test, X_test = coShuffled_vectors(X_test, Y_test)

    X_validation = Y_validation = None        
    if validation_split is not None:
        # sklearn split shuffles anyway
        X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_split)        
    
    return (X_train, Y_train, X_test, Y_test, X_validation, Y_validation)



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
    plt.legend(acc_keys, loc='lower right')
    plt.figure(2)
    plt.title('Loss vs. epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loss_keys, loc='upper right')
    plt.show()


def get_layer_by_name(layers, name, return_first = True):
    matching_named_layers = [ l for l in layers if l.name==name]
    if not matching_named_layers:
        return None
    return matching_named_layers[0] if return_first else matching_named_layers


def get_combined_features_from_models(
        to_combine, 
        X_train, Y_train, 
        X_test, Y_test,
        reverse_one_hot = False,
        normalize_X_func = None):
    
    models = []
    models_dict = {}
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
            if model_file_name in models_dict.keys():
                model_here = models_dict[model_file_name]
            else:
                model_here = tf.keras.models.load_model(model_file_name, **kwargs) if kwargs is not None else tf.keras.models.load_model(model_file_name)
                
        features_model = Model(model_here.input, 
                               get_layer_by_name(model_here.layers,layer_name).output)
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
        models.append( ( (model_file_name, layer_name), model_here) )
        models_dict[model_file_name] = model_here            
        
    X_train_new = np.concatenate(tuple(X_trains_out), axis = 1)
    X_test_new = np.concatenate(tuple(X_test_out), axis = 1)
    
    data_train = (X_train_new, Y_train_new)
    data_test = (X_test_new, Y_test_new)    
    
        
    return (models, data_train, data_test,XY_dict)



def make_dir_if_not_exist(used_path):
    if not os.path.isdir(used_path):
        try:
            os.mkdir(used_path)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise exc
            else:
                raise ValueError(f'{used_path} directoy cannot be created because its parent directory does not exist.')


def source_model(model_func, model_name, input_shape):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)
    return m


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
                  activation=hp.Choice(f'final_dense_kernel_activation_{idx}',
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
    

class BatchSizeTuner(kt.tuners.Hyperband):
   def run_trial(self, trial, *args, **kwargs):
#     # You can add additional HyperParameters for preprocessing and custom training loops via overriding `run_trial`
     kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 16, 256, step=32)
     super(BatchSizeTuner, self).run_trial(trial, *args, **kwargs)
#     kwargs['epochs'] = trial.hyperparameters.Int('epochs', 10, 30)



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

def reinitialize_weights(model):
    for ix, layer in enumerate(model.layers):
        if hasattr(model.layers[ix], 'kernel_initializer') and hasattr(model.layers[ix], 'bias_initializer'):
            weight_initializer = model.layers[ix].kernel_initializer
            bias_initializer = model.layers[ix].bias_initializer
    
            old_weights, old_biases = model.layers[ix].get_weights()
    
            model.layers[ix].set_weights([
                weight_initializer(shape=old_weights.shape),
                bias_initializer(shape=len(old_biases))])            
    return model

def reverse_tensor(X):
    return tf.gather(X, tf.reverse(tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32),(0,)) )


class DNNFeatureMergeModel(HyperModel):
    
    def __init__(self, model_name, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_name = model_name

    def build(self, hp):
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        for idx, i in enumerate(range(hp.Int('dense_blocks_with_normalizations', 1, 5, default=1))):
            if hp.Boolean(f'has_batchnormalization_{idx}', default=True):
                x = BatchNormalization()(x)
            x = Dense(units= hp.Choice(f'dense_block_nunits_{idx}', values=[32,64,128,256],default=128), 
                      activation=hp.Choice(f'dense_activation_{idx}',
                                       values=['selu', 'gelu', 'elu', 'relu', 'tanh', 'linear'], default='relu'),
                  kernel_initializer=hp.Choice(f'dense_kernel_init_{idx}',
                                       values=['HeNormal', 'VarianceScaling', 'GlorotUniform', 'RandomNormal'], default='RandomNormal'),
                  bias_initializer='zeros',
                  name=f'dense_{idx}')(x)
            if hp.Boolean(f'has_leakyrelu_{idx}', default=True):
                x = LeakyReLU()(x)
            if hp.Boolean(f'has_dropout_{idx}', default=True):
                x = Dropout(hp.Float(f'dense_dropout_value_{idx}',
                                 min_value=0.1,
                                 max_value=0.9,
                                 step=0.1,
                                 default=0.6
                                 ), name=f"dense_dropout_{idx}")(x)
        outputs = Dense(self.num_classes, activation='softmax', name="last_softmax")(x)
        model = tf.keras.Model(inputs, outputs, name=self.model_name)
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
    

class BatchSizeTuner(kt.tuners.Hyperband):
   def run_trial(self, trial, *args, **kwargs):
#     # You can add additional HyperParameters for preprocessing and custom training loops via overriding `run_trial`
     kwargs['batch_size'] = trial.hyperparameters.Int('batch_size', 16, 256, step=32)
     super(BatchSizeTuner, self).run_trial(trial, *args, **kwargs)

# X_train, Y_train, X_test, Y_test, X_validation, Y_validation  = getData(is500=False, validation_split=0.1)
X_train, Y_train, X_test, Y_test, X_validation, Y_validation  = getData(is500=False)
X_test_shuffled, Y_test_shuffled = coShuffled_vectors(X_test, Y_test)
X_train_shuffled, Y_train_shuffled = coShuffled_vectors(X_train, Y_train)
X_train_rev = reverse_tensor(X_train)
Y_train_rev = reverse_tensor(Y_train)

mRNNx = tf.keras.models.load_model("RNN_3stacked_45_epochs.h5", custom_objects=SeqWeightedAttention.get_custom_objects()) 
mRNNx._name = "rnn_3stacked_45ep_model"
# mRNNx.evaluate(X_test, Y_test)
mRNNx.save(f'./combination_models/{mRNNx.name}.h5')


mRNN = tf.keras.models.load_model("PureRNN.h5", custom_objects=SeqWeightedAttention.get_custom_objects()) 
mRNN._name = "pure_rnn_model"
# mRNN.evaluate(X_test_shuffled, Y_test_shuffled)
# mRNN.evaluate(X_test, Y_test)
mRNN.save(f'./combination_models/{mRNN.name}.h5')


mCNN = tf.keras.models.load_model("CNN_new.h5")
mCNN._name = "cnn_model"
# mCNN.evaluate(X_test, Y_test)
mCNN.save(f'./combination_models/{mCNN.name}.h5')


mCNNb = tf.keras.models.load_model("CNN_baseline.h5")
mCNNb._name = "CNN_baseline_model"
mCNNb.save(f'./combination_models/{mCNNb.name}.h5')


# learn in reverse sequence... then do shuffled in smaller batches
# mCNNr = tf.keras.models.clone_model(mCNN)
# mCNNr._name = f'{mCNNr.name}_rev'
# mCNNr = reinitialize_weights(mCNNr)
# mCNNr.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history_cnn_rev = mCNNr.fit(X_train_rev, Y_train_rev, batch_size=128, epochs=50, class_weight=None, validation_data=(X_test, Y_test))
# history_cnn_rev_2 = mCNNr.fit(X_train_rev, Y_train_rev, batch_size=256, epochs=30, class_weight=None, validation_data=(X_test, Y_test))
# history_cnn_rev_3 = mCNNr.fit(X_train_shuffled, Y_train_shuffled, batch_size=64, epochs=40, class_weight=None, validation_data=(X_test, Y_test))
# mCNNr.save(f'./combination_models/{mCNNr.name}.h5')
# plot_history(history_cnn_rev_3)
mCNNr = tf.keras.models.load_model("./combination_models/cnn_model_rev.h5")


# # learn in shuffled sequence...
# mCNNs = tf.keras.models.clone_model(mCNN)
# mCNNs._name = f'{mCNNs.name}_shuff'
# mCNNs = reinitialize_weights(mCNNs)
# mCNNs.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# history_cnn_shuff = mCNNs.fit(X_train_shuffled, Y_train_shuffled, batch_size=128, epochs=50, class_weight=None, validation_data=(X_test, Y_test))
# mCNNs.save(f'./combination_models/{mCNNs.name}.h5')
# plot_history(history_cnn_shuff)
mCNNs = tf.keras.models.load_model("./combination_models/cnn_model_shuff.h5")

# tf.keras.utils.plot_model(mCNNb, show_shapes=True)

# crazy idea of only the final layers...
to_combine_last_layers = [ 
                (mRNN, "dense_3", None),
                (mRNNx, "dense_3", None),
                (mCNN, "dense_8", None),
                (mCNNr, "dense_8", None),
                (mCNNs, "dense_8", None),
                (mCNNb, "dense_11", None),
              ]

combined_models, data_train_ll, data_test_ll, data_access_ll = get_combined_features_from_models(to_combine_last_layers, 
                                  X_train_shuffled.numpy(), Y_train_shuffled.numpy(), 
                                  X_test, Y_test, 
                                  reverse_one_hot = False)

hypermodel_lastlayers_merge_1 = DNNFeatureMergeModel(model_name="combine_last_layers", input_shape=(data_train_ll[0].shape[1],), num_classes=data_train_ll[1].shape[-1] )
tuner = BatchSizeTuner(
    hypermodel_lastlayers_merge_1,
    max_epochs=60,
    objective='val_accuracy',
    executions_per_trial=1,
    hyperband_iterations=3,
    seed=123,
    directory=f'{hypermodel_lastlayers_merge_1.model_name}_hyperband'
)

tuner.search(data_train_ll[0], data_train_ll[1],
             epochs=50,
             validation_data=(data_test_ll[0], data_test_ll[1]),
             callbacks=[tf.keras.callbacks.EarlyStopping(patience=1)])


# Show a summary of the search
tuner.results_summary()
# Retrieve the best model.

tuner.get_best_models()

best_model = tuner.get_best_models(num_models=1)[0]
best_model.save("hypermodel_merge_6lastlayers_m1.h5")

# Evaluate the best model.
best_model.evaluate(data_test_ll[0],data_test_ll[1])

def get_confusion_matrix_classification(model, X, Y_true):
    y_pred = model.predict(X)
    y_true = np.apply_along_axis(np.argmax, 1, Y_true)
    y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
    return (confusion_matrix(y_true, y_pred), y_pred, y_true)


test_conf_matrix, test_pred, test_act = get_confusion_matrix_classification(best_model, data_test_ll[0], data_test_ll[1])
train_conf_matrix, train_pred, train_act = get_confusion_matrix_classification(best_model, data_train_ll[0], data_train_ll[1])
    

Y_test_pred = best_model.predict(data_test_ll[0])
diffs = np.apply_along_axis(np.argmax, 1, Y_test_pred) - np.apply_along_axis(np.argmax, 1, data_test_ll[1])
from sklearn.metrics import confusion_matrix
y_true = np.apply_along_axis(np.argmax, 1, data_test_ll[1])
y_pred = np.apply_along_axis(np.argmax, 1, Y_test_pred)
confusion_matrix(y_true, y_pred)
# y_true_train = np.apply_along_axis(np.argmax, 1, data_train_ll[1])
# Y_train_pred = best_model.predict(data_train_ll[0])
# y_pred_train = np.apply_along_axis(np.argmax, 1, best_model.predict(data_train_ll[0]))
# confusion_matrix(y_true_train, y_pred_train)
print(sum(1 for x in diffs if x == 0)/Y_test_pred.shape[0])


## now let's see if class weight on the training set improves the input models from where they are...

def misclass_perc_to_weight(input_confusion, add_base=True, func=None):
    perc_misclassified = 1.0 - np.array([ input_confusion[x,x] for x in np.arange(input_confusion.shape[0]).tolist() ])/input_confusion.sum(axis=1)
    
    base_val = min(perc_misclassified[perc_misclassified>0.0])
    if add_base:        
        perc_misclassified = perc_misclassified + base_val
    
    perc_misclassified = [ x/base_val for x in perc_misclassified]
    
    return dict([ (idx, func(perc_val)) if func is not None else (idx, perc_val) for idx, perc_val in enumerate(perc_misclassified) ])
    

train_conf_matrix_input_cnn, train_pred_input_cnn, train_act_input_cnn = get_confusion_matrix_classification(mCNN,X_train_shuffled, Y_train_shuffled)
mCNN_cw_after = tf.keras.models.clone_model(mCNN)
mCNN_cw_after.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history_cnn_weight_continueFit = mCNN_cw_after.fit(X_train_shuffled, Y_train_shuffled, 
                                          batch_size=128, 
                                          epochs=100, 
                                          class_weight=misclass_perc_to_weight(train_conf_matrix_input_cnn, func = lambda x: pow(1.1,x) ), 
                                          validation_data=(X_test_shuffled, Y_test_shuffled),
                                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=10)])
history_cnn_weight_continueFit = mCNN_cw_after.fit(X_train_shuffled, Y_train_shuffled, 
                                          batch_size=32, 
                                          epochs=100, 
                                          class_weight=misclass_perc_to_weight(train_conf_matrix_input_cnn, func = lambda x: pow(1.1,x) ), 
                                          validation_data=(X_test_shuffled, Y_test_shuffled),
                                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=25)])
mCNN_cw_after._name  = f'{mCNN_cw_after.name}_after_class_weightAdj'
mCNN_cw_after.save(f'./combination_models/{mCNN_cw_after.name}.h5')



train_conf_matrix_input_cnnb, train_pred_input_cnnb, train_act_input_cnnb = get_confusion_matrix_classification(mCNNb,X_train_shuffled, Y_train_shuffled)
# no confusion...


plot_history(history_cnn_weight_continueFit)





















# to_combine = [ (mRNN, "dense_1", None),
#                (mRNNx, "dense_1", None),
#                (mCNN, "dense_6", None),
#                (mCNN, "dense_7", None)
#              ]

# combined_models, data_train, data_test, data_access = get_combined_features_from_models(to_combine, 
#                                   X_train, Y_train, 
#                                   X_test, Y_test, 
#                                   reverse_one_hot = False)
# # ,                                  normalize_X_func = tf.math.l2_normalize)                                  
                                  

    

# def run_post_combine_model(model_func, input_shape, X_train, Y_train, kwargs = None, save_name = None):    
#     m = model_func(input_shape)
#     m.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
#     history = m.fit(X_train, Y_train, **kwargs) if kwargs is not None else m.fit(X_train, Y_train)
#     if save_name is not None:
#         m.save(save_name) 
#     return (m,history)

# combined_model, history_dense_post_combined = run_post_combine_model(model_combination_1,
#                     (data_train[0].shape[1],),
#                     data_train[0],
#                     to_categorical(data_train[1]),
#                     { "batch_size":512, "epochs":500, "class_weight":None, "validation_data":(data_test[0], to_categorical(data_test[1]))} )

# combined_model_2, history_dense_post_combined_2 = run_post_combine_model(model_combination_1,
#                     (data_train[0].shape[1],),
#                     data_train[0],
#                     to_categorical(data_train[1]),
#                     { "batch_size":4, "epochs":30, "class_weight":None, "validation_data":(data_test[0], to_categorical(data_test[1]))} )


# # tf.keras.utils.plot_model(combined_model, show_shapes=True)

# to_combine_from_post = [(combined_model, "dense_49", None),
#                        (combined_model_2, "dense_52", None),
#                        (combined_model_2, "dense_53", None)
#                ] 

# combined_post_models, data_train_post, data_test_post, data_access_post = get_combined_features_from_models(to_combine_from_post, 
#                                   data_train[0], data_train[1], 
#                                   data_test[0], data_test[1], 
#                                   reverse_one_hot = False , 
#                                   normalize_X_func = tf.math.l2_normalize)                                  

# # adding tf.math.l2_normalize improves results for the GP that uses layers from the combining layer
# # plot_history(history_dense_post_combined)
# # Commented out IPython magic to ensure Python compatibility.
# # %matplotlib inline
# #from tensorflow2_work.multiclass_classification import plot_posterior_predictions, colors

# # reproducibility:
# np.random.seed(0)
# tf.random.set_seed(123)

# data_gp_train = np.concatenate((data_train[0],data_train_post[0]), axis = 1)
# data_gp_test = np.concatenate((data_test[0],data_test_post[0]), axis = 1)

# # actually removing the combining dense layer elements improves results!
# # data_gp_train = data_train[0]
# # data_gp_test = data_test[0]

# data_gp = ( data_gp_train, data_train[1] )

# # sum kernel: Matern32 + White
# lengthscales = tf.convert_to_tensor([1.0] * data_gp_train.shape[1], dtype=default_float())
# # kernel = gpflow.kernels.Matern32(lengthscales=lengthscales) #+ gpflow.kernels.White(variance=0.01)
# kernel = gpflow.kernels.Matern52(lengthscales=lengthscales)

# # Robustmax Multiclass Likelihood
# invlink = gpflow.likelihoods.RobustMax(13)  # Robustmax inverse link function
# likelihood = gpflow.likelihoods.MultiClass(13, invlink=invlink)  # Multiclass likelihood
# # M = 20  # Number of inducing locations
# # Z = data_gp_train[::M].copy()  # inducing inputs CHECK DIMENSIONS
# M = 20  # Number of inducing locations
# # Z = data_gp_train[sorted(random.sample(range(0,data_gp_train.shape[0]),M)),:].copy()
# Z = data_gp_train[::M].copy()  # inducing inputs CHECK DIMENSIONS

# mGP = gpflow.models.SVGP(
#     kernel=kernel,
#     likelihood=likelihood,
#     inducing_variable=Z,
#     num_latent_gps=13,
#     whiten=False,
#     q_diag=False,
# )

# # Only train the variational parameters
# # set_trainable(mGP.kernel.kernels[1].variance, True)
# set_trainable(mGP.inducing_variable, False)

# # print_summary(mGP)

# opt = gpflow.optimizers.Scipy()
# opt_logs = opt.minimize(
#     mGP.training_loss_closure(data_gp), mGP.trainable_variables, options=dict(maxiter=ci_niter(400),method="trust-ncg")
# )
# # print_summary(mGP)

# # plt.plot(opt_logs['x'])

# (Y_test_GP_mean, Y_test_GP_variance) = mGP.predict_y(data_gp_test)
# Y_test_GP_mean = Y_test_GP_mean.numpy()
# Y_test_pred = []
# for i in range(data_gp_test.shape[0]):
#   Y_test_pred.append(np.argmax(Y_test_GP_mean[i,]))
  
# diffs = Y_test_pred  - data_test[1]
# test_acc = sum(1 for x in diffs if x == 0)/data_gp_test.shape[0]
# print(test_acc)







