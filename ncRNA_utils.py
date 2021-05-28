import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import h5py as h5
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, LeakyReLU, MaxPooling1D, BatchNormalization, GaussianNoise, Dropout, Dense, Flatten
import errno
import os
from collections import defaultdict
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.summary import create_file_writer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import kerastuner as kt
from kerastuner import HyperModel
import numpy as np
import itertools
import multiprocessing
from numpy import genfromtxt
import seaborn as sns

def coShuffled_vectors(X, Y):
    if tf.shape(X)[0] == tf.shape(Y)[0]:
        test_idxs = tf.range(start=0, limit=tf.shape(X)[0], dtype=tf.int32)
        shuffled_test_idxs = tf.random.shuffle(test_idxs)
        return (tf.gather(X, shuffled_test_idxs), tf.gather(Y, shuffled_test_idxs))
    else:
        raise ValueError(f"0-dimension has to be the same {tf.shape(X)[0]} != {tf.shape(Y)[0]}")

def reverse_one_hot(Y_input):
    return np.apply_along_axis(np.argmax, 1, Y_input) + 1

def getNpArrayFromH5(hf_Data):
    X_train = hf_Data['Train_Data']  # Get train set
    X_train = np.array(X_train)
    Y_train = hf_Data['Label']  # Get train label
    Y_train = np.array(Y_train)
    return X_train, Y_train

def get_confusion_matrix_classification(model, X, Y_true):
    y_pred = model.predict(X)
    y_true = np.apply_along_axis(np.argmax, 1, Y_true)
    y_pred = np.apply_along_axis(np.argmax, 1, y_pred)
    return (confusion_matrix(y_true, y_pred), y_pred, y_true)

def misclass_perc_to_weight(input_confusion, add_base=True, func=None):
    perc_misclassified = 1.0 - np.array([ input_confusion[x,x] for x in np.arange(input_confusion.shape[0]).tolist() ])/input_confusion.sum(axis=1)
    
    base_val = min(perc_misclassified[perc_misclassified>0.0])
    if add_base:        
        perc_misclassified = perc_misclassified + base_val
    
    perc_misclassified = [ x/base_val for x in perc_misclassified]
    return dict([ (idx, func(perc_val)) if func is not None else (idx, perc_val) for idx, perc_val in enumerate(perc_misclassified) ])


from sklearn.metrics import precision_recall_fscore_support
from pycm import ConfusionMatrix

def prf(model,xtest, ytest):
  y_pred = np.apply_along_axis(np.argmax, 1, model.predict(xtest))
  y_true = np.apply_along_axis(np.argmax, 1, ytest)
  return precision_recall_fscore_support(y_true, y_pred, average="weighted")

def get_sp_pr_rc_f1(model,xtest, ytest):  
    y_pred = np.apply_along_axis(np.argmax, 1, model.predict(xtest))
    y_true = np.apply_along_axis(np.argmax, 1, ytest)
    cmres = ConfusionMatrix(actual_vector=y_true,predict_vector=y_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted")   
    return cmres.TNR_Macro, pr, rc, f1
 
def get_sp_pr_rc_f1_acc(model,xtest, ytest):  
    spec, pr, rc, f1 = get_sp_pr_rc_f1(model,xtest, ytest)
    acc = model.evaluate(xtest,ytest)[-1]        
    return spec, pr, rc, f1, acc

# cmres = [ ConfusionMatrix(actual_vector=np.apply_along_axis(np.argmax, 1, ytest)  ,predict_vector=np.apply_along_axis(np.argmax, 1, model.predict(xtest))) for model, xtest, ytest in to_eval ]
# [ (x.TNR_Macro,x.TPR_Macro, x.PPV_Macro, x.F1_Macro)  for x in cmres ]  # specificity/true neg rate, sensitivity/recall, precision/predictive pos value, F1= harmonic(precision/recall)

# data extraction
def getData(is500=True, shuffle=False, ise2e=False, include_secondary=False, validation_split=None, isColab=False):
    if not include_secondary:
        hf_Train = h5.File(
            f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/{"e2e_Train_Data" if ise2e else "Fold_10_Train_Data"}_{str(500) if is500 else str(1000)}.h5', 'r')
        hf_Test = h5.File(
            f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/{"e2e_Test_Data" if ise2e else "Fold_10_Test_Data"}_{str(500) if is500 else str(1000)}.h5', 'r')
    else:
        hf_Train = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Train_Secondary_Data_1136.h5', 'r')
        hf_Test = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Test_Secondary_Data_1136.h5', 'r')

    X_train, Y_train = getNpArrayFromH5(hf_Train)
    X_test, Y_test = getNpArrayFromH5(hf_Test)
    Y_train = to_categorical(Y_train, 13)  # Process the label of tain
    Y_test = to_categorical(Y_test, 13)  # Process the label of te

    if shuffle:
        X_train, Y_train = coShuffled_vectors(X_train, Y_train)
        X_test, Y_test = coShuffled_vectors(X_test, Y_test)

    X_validation = Y_validation = None
    if validation_split is not None:
        # sklearn split shuffles anyway
        X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=validation_split)

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


def getE2eData(is500=True, shuffle=False, include_secondary=False, isColab=False):
    if not include_secondary:
        hf_Train = h5.File(
            f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Train_Data_{str(500) if is500 else str(1000)}.h5', 'r')
        hf_Test = h5.File(
            f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Test_Data_{str(500) if is500 else str(1000)}.h5', 'r')
    else:
        hf_Train = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Train_Secondary_Data_1136.h5', 'r')
        hf_Test = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Test_Secondary_Data_1136.h5', 'r')

    X_train, Y_train = getNpArrayFromH5(hf_Train)
    X_test, Y_test = getNpArrayFromH5(hf_Test)
    Y_train = to_categorical(Y_train, 13)  # Process the label of tain
    Y_test = to_categorical(Y_test, 13)  # Process the label of te

    if shuffle:
        X_train, Y_train = coShuffled_vectors(X_train, Y_train)
        X_test, Y_test = coShuffled_vectors(X_test, Y_test)

    hf_Val = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Val_Secondary_Data_1136.h5', 'r') if include_secondary else h5.File(
        f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Val_Data_{str(500) if is500 else str(1000)}.h5', 'r')
    X_validation, Y_validation = getNpArrayFromH5(hf_Val)
    Y_validation = to_categorical(Y_validation, 13)  # Process the label of tain

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


def getE2eDataJustSecondary(shuffle=False,isColab=False):
    hf_Train = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Train_just_Secondary_Data_1000.h5', 'r')
    hf_Test = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Test_just_Secondary_Data_1000.h5', 'r')

    X_train, Y_train = getNpArrayFromH5(hf_Train)
    X_test, Y_test = getNpArrayFromH5(hf_Test)
    
    
    Y_train = to_categorical(Y_train, Y_test.shape[-1])  # Process the label of tain
    Y_test = to_categorical(Y_test, Y_test.shape[-1])  # Process the label of te

    if shuffle:
        X_train, Y_train = coShuffled_vectors(X_train, Y_train)
        X_test, Y_test = coShuffled_vectors(X_test, Y_test)

    hf_Val = h5.File(f'./{"data" if not isColab else "drive/MyDrive/data_papers/ncRNA"}/e2e_Val_just_Secondary_Data_1000.h5', 'r')
    
    X_validation, Y_validation = getNpArrayFromH5(hf_Val)
    Y_validation = to_categorical(Y_validation, Y_test.shape[-1])  # Process the label of tain

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation


def getTest12Data():
    hf_Test = h5.File(f'./data/e2e_Test_Data_1000_12classes.h5', 'r')
    X_test, Y_test = getNpArrayFromH5(hf_Test)
    Y_test = to_categorical(Y_test, 13)  # Process the label of te

    return X_test, Y_test


def get88KData():
    hf_Train = h5.File(f'./data/e2e_Train_Data_1000_88.h5', 'r')
    hf_Test = h5.File(f'./data/e2e_Test_Data_1000_88.h5', 'r')

    X_train, Y_train = getNpArrayFromH5(hf_Train)
    X_test, Y_test = getNpArrayFromH5(hf_Test)
    Y_train = to_categorical(Y_train, Y_test.shape[-1])  # Process the label of tain
    Y_test = to_categorical(Y_test, Y_test.shape[-1])  # Process the label of te

    hf_Val = h5.File(f'./data/e2e_Val_Data_1000_88.h5', 'r') 
    X_validation, Y_validation = getNpArrayFromH5(hf_Val)
    Y_validation = to_categorical(Y_validation, Y_test.shape[-1])  # Process the label of tain

    return X_train, Y_train, X_test, Y_test, X_validation, Y_validation
    


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


def get_layer_by_name(layers, name, return_first=True):
    matching_named_layers = [l for l in layers if l.name == name]
    if not matching_named_layers:
        return None
    return matching_named_layers[0] if return_first else matching_named_layers


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


def compile_and_fit_model_with_tb(model_func,
                                  model_name,
                                  input_shape,
                                  X_train,
                                  Y_train,
                                  save_every_epoch=True,
                                  save_final=False,
                                  **kwargs):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)
    tb_callback = TensorBoard(log_dir=f'{m.name}_logs', histogram_freq=kwargs.pop("histogram_freq", 1))
    if save_every_epoch:
        tb_callback.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{val_accuracy:0.2f}'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=[tb_callback], verbose=2, **kwargs)
    if save_final:
        make_dir_if_not_exist(model_name)
        m.save(f"{m.name}_saved_model_after_fit")  # Save the model
    return (m, history)
    # m.save(f"{m.name}_Tenth_Fold_New_Model_500_8") #Save the model


def compile_and_fit_model(model_func,
                          model_name,
                          input_shape,
                          X_train,
                          Y_train,
                          save_every_epoch=True,
                          save_final=False,
                          **kwargs):
    m = None
    if isinstance(model_func, tf.keras.models.Model):
        m = model_func
        m._name = model_name
    else:
        m = model_func(model_name, input_shape)

    callbacks_used = []
    if save_every_epoch:
        callbacks_used.append(ModelCheckpoint(f'{m.name}' + '_model_{epoch:03d}_{val_accuracy:0.2f}'))
    m.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = m.fit(X_train, Y_train, callbacks=callbacks_used, verbose=2, **kwargs)
    if save_final:
        make_dir_if_not_exist(model_name)
        m.save(f"{m.name}_saved_model_after_fit")  # Save the model
    return (m, history)


def compile_model_and_fit_with_custom_loop(model_func,
                                           model_name,
                                           input_shape,
                                           X_train,
                                           Y_train,
                                           **kwargs):
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
            with test_writer.as_default():
                tf.summary.scalar("Loss", loss, step=test_step)
                tf.summary.scalar(
                    "Accuracy", acc_metric.result(), step=test_step,
                )
                test_step += 1

        acc_metric.reset_states()  # Reset accuracy in between epochs (and for testing and test)

    return m



def run_mirrored_strategy(model_func, base_batch_size, nepochs, x_train, y_train, x_test, y_test, **kwargs):
    strategy = tf.distribute.MirroredStrategy()
    with strategy.scope():
        model = model_func()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=tf.keras.metrics.SparseCategoricalAccuracy()
        )
    batch_size_mirr_strat = base_batch_size * strategy.num_replicas_in_sync
    history = model.fit(x_train, y_train, epochs=nepochs, batch_size=batch_size_mirr_strat,
                        validation_data=(x_test, y_test),
                        **kwargs)
    return model, history


def sparse_setdiff(a1, a2):
    a1a = a1.reshape(a1.shape[0], -1)
    a2a = a2.reshape(a2.shape[0], -1)
    spa2a = [np.where(x)[0].tolist() for x in a2a]
    spa1a = [np.where(x)[0].tolist() for x in a1a]
    idxs_to_keep = []
    for idx, sample in enumerate(spa1a):
        try:
            spa2a.index(sample)
        except ValueError:
            # not in list
            idxs_to_keep.append(idx)
    return a1[idxs_to_keep], idxs_to_keep



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

def get_combined_features_from_models(
    
        to_combine,
        X_train, Y_train,
        X_test, Y_test,
        reverse_one_hot=False,
        normalize_X_func=None):
    
    models = []
    models_dict = {}
    X_trains_out = []
    X_test_out = []
    XY_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: None))))

    models_have_different_inputs = isinstance(Y_train,list)

    if reverse_one_hot:
        if models_have_different_inputs:
            Y_train_new = np.apply_along_axis(np.argmax, 1, Y_train) + 1
            Y_test_new = np.apply_along_axis(np.argmax, 1, Y_test) + 1
        else:
            Y_train_new = [ np.apply_along_axis(np.argmax, 1, y_train) + 1 for y_train in Y_train ]  
            Y_test_new = [ np.apply_along_axis(np.argmax, 1, y_test) + 1 for y_test in Y_test ]              
    else:
        if models_have_different_inputs:
            Y_train_new = Y_train.copy()
            Y_test_new = Y_test.copy()
        else:
            Y_train_new = [ y_train.copy() for y_train in Y_train ] 
            Y_test_new = [ y_test.copy() for y_test in Y_train ] 
            

    extraction_counter =0
    for model_file_name, layer_name, kwargs in to_combine:
        model_here = None
        if isinstance(model_file_name, tf.keras.models.Model):
            model_here = model_file_name
            model_file_name = model_here.name
        else:
            if model_file_name in models_dict.keys():
                model_here = models_dict[model_file_name]
            else:
                model_here = tf.keras.models.load_model(model_file_name,
                                                        **kwargs) if kwargs is not None else tf.keras.models.load_model \
                    (model_file_name)

        features_model = Model(model_here.input,
                               get_layer_by_name(model_here.layers, layer_name).output)
        
        if normalize_X_func is None:
            X_trains_out.append(np.array(features_model.predict(X_train if not models_have_different_inputs else X_train[extraction_counter]), dtype='float64'))
            X_test_out.append(np.array(features_model.predict(X_test if not models_have_different_inputs else X_test[extraction_counter]), dtype='float64'))
        else:
            X_trains_out.append(np.array(normalize_X_func(features_model.predict(X_train if not models_have_different_inputs else X_train[extraction_counter])), dtype='float64'))
            X_test_out.append(np.array(normalize_X_func(features_model.predict(X_test if not models_have_different_inputs else X_test[extraction_counter])), dtype='float64'))
        XY_dict[model_file_name][layer_name]['Train']['X'] = X_trains_out[-1]
        XY_dict[model_file_name][layer_name]['Test']['X'] = X_test_out[-1]
        XY_dict[model_file_name][layer_name]['Train']['Y'] = Y_train_new
        XY_dict[model_file_name][layer_name]['Test']['Y'] = Y_test_new
        models.append(((model_file_name, layer_name), (model_here, features_model)))
        models_dict[model_file_name] = model_here
        extraction_counter += 1

    X_train_new = np.concatenate(tuple(X_trains_out), axis=1)
    X_test_new = np.concatenate(tuple(X_test_out), axis=1)

    data_train = (X_train_new, Y_train_new)
    data_test = (X_test_new, Y_test_new)

    return models, data_train, data_test, XY_dict



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
    

class SaveBestOverCombinedThresholds(tf.keras.callbacks.Callback):

    def __init__(self, colab_download = False, observed_values = [ ('accuracy',0.9) ] ):
        self.thresholds = dict(observed_values)
        self.last_best_values = dict([ (obs_name, np.nan) for obs_name in self.thresholds.keys()] )
        self.colab_download = colab_download
        
    def on_epoch_end(self, epoch, logs=None):        
        register = None
        for k,v in self.thresholds.items():
            if k not in logs.keys():
                raise ValueError(f"{k} not found in logs")
            passes_threshold = logs[k] > self.thresholds[k]                 
            register = passes_threshold if register is None else (register and passes_threshold)
        
        if register:
            for k,v in self.thresholds.items():
                if np.isnan(self.last_best_values[k]):
                    self.last_best_values[k] = logs[k]
                else:
                    if logs[k] < self.last_best_values[k]:
                        register = False
                        break
            if register:
                for k,v in self.thresholds.items():
                    self.last_best_values[k] = logs[k]
                base_name = f'{self.model.name}_epoch_{str(epoch)}_{"_".join(["{}_{:.3f}".format(k,v) for k,v in self.last_best_values.items()])}'
                self.model.save(f'{base_name}.h5')                
                history_df = pd.DataFrame(self.model.history.history) 
                history_df.to_csv(f'{base_name}_history.csv',header=True, index=False)
                
                if self.colab_download:
                    from google.colab import files
                    files.download(f'{base_name}.h5')
                    files.download(f'{base_name}_history.csv')
                
                
                
# gpout_data = genfromtxt("D:/Code/ncRNA/data/gp_no2nd256_no2nd128_j2nd256.csv", delimiter=',')        
# gpout_data = gpout_data[gpout_data>0.0]
       
# sns.distplot(gpout_data, hist=True, kde=False, 
#              bins=int(180/5), color = 'blue',
#              hist_kws={'edgecolor':'black'})
# # Add labels
# plt.title('Histogram of Arrival Delays')
# plt.xlabel('Delay (min)')
# plt.ylabel('Flights')


# sns.distplot(gpout_data, hist=True, kde=True, 
#      bins=int(180/5), color = 'darkblue', 
#      hist_kws={'edgecolor':'black'},
#      kde_kws={'linewidth': 4})

    

# def unpacking_apply_along_axis(all_args):
#     """
#     Like numpy.apply_along_axis(), but with arguments in a tuple
#     instead.

#     This function is useful with multiprocessing.Pool().map(): (1)
#     map() only handles functions that take a single argument, and (2)
#     this function can generally be imported from a module, as required
#     by map().
#     """
#     (func1d, axis, arr, args, kwargs) = all_args
#     # return np.apply_along_axis(func1d, axis, arr, *args, **kwargs)


# def parallel_apply_along_axis(func1d, axis, arr, *args, **kwargs):
#     """
#     Like numpy.apply_along_axis(), but takes advantage of multiple
#     cores.
#     """
#     # Effective axis where apply_along_axis() will be applied by each
#     # worker (any non-zero axis number would work, so as to allow the use
#     # of `np.array_split()`, which is only done on axis 0):
#     effective_axis = 1 if axis == 0 else axis
#     if effective_axis != axis:
#         arr = arr.swapaxes(axis, effective_axis)

#     # Chunks for the mapping (only a few chunks):
#     chunks = [(func1d, effective_axis, sub_arr, args, kwargs)
#               for sub_arr in np.array_split(arr, multiprocessing.cpu_count())]

#     pool = multiprocessing.Pool()
#     individual_results = pool.map(unpacking_apply_along_axis, chunks)
#     # Freeing the workers:
#     pool.close()
#     pool.join()

#     return np.concatenate(individual_results)

# ## not convinced these are correct, a check on the in/output does not pan out ... do not use now
# https://github.com/phantomachine/numpy-setdiff-intersect/blob/master/setops.py

# #---------- SET-DIFFERENCE OF ARRAYS (not applied here...) -------------------------------------
# def setdiff(a1,a2):
#     a1a = a1.reshape(a1.shape[0], -1)
#     a2a = a2.reshape(a2.shape[0], -1)
#     a1_rows = a1a.view([('', a1a.dtype)] * a1a.shape[-1])
#     a2_rows = a2a.view([('', a2a.dtype)] * a2a.shape[-1])
#     ad = np.setdiff1d(a1_rows, a2_rows).view(a1a.dtype).reshape(-1, a1a.shape[-1])
#     # ad.shape
#     # index_bool.shape
#     # np.where(index_bool)
#     index_bool = np.in1d(a1_rows, a2_rows, invert=True)
#     index = np.arange(index_bool.size)[index_bool]
#     return ad, index, index_bool


# def setdiff2ndIdx(a1, a2):
#     """Simpler version of Matlab/R's SETDIFF for ND arrays """
#     a1 = a1.ravel().reshape(-1,a1.shape[-1])
#     a2 = a2.ravel().reshape(-1,a1.shape[-1])
#     a1_rows = a1.view([('', a1.dtype)] * a1.shape[-1])
#     a2_rows = a2.view([('', a2.dtype)] * a2.shape[-1])
#     ad = np.setdiff1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[-1])
#     index_bool = np.in1d(a1_rows, a2_rows, invert=True)
#     index = np.arange(index_bool.size)[index_bool]
#     return ad, index, index_bool

# #---------- INTERSECTION OF ARRAYS --------------------------------------
# def intersect(a1, a2):
#     """Simpler version of Matlab/R's INTERSECT for ND arrays """
#     a1 = a1.ravel().reshape(-1,a1.shape[-1])
#     a2 = a2.ravel().reshape(-1,a1.shape[-1])
#     a1_rows = a1.view([('', a1.dtype)] * a1.shape[-1])
#     a2_rows = a2.view([('', a2.dtype)] * a2.shape[-1])
#     ax=np.intersect1d(a1_rows, a2_rows).view(a1.dtype).reshape(-1, a1.shape[-1])
#     index_bool = np.in1d(a1_rows, a2_rows, invert=False)
#     index = np.arange(index_bool.size)[index_bool]
#     return ax, index, index_bool

# #------------- PYTHON LIST or NUMPY ARRAYS: Flatten vs. Reconstruct -------
# def flatten2(nl):
#     """
#     To flatten Python List of lists / numpy arrays (2 levels). (See also reverse operation in RECONSTRUCT() below.)
#     Usage: L_flat,l1,l2 = flatten2(L)
#     Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
#     """
#     l1 = [len(s) for s in itertools.chain.from_iterable(nl)]
#     l2 = [len(s) for s in nl]

#     nl = list(itertools.chain.from_iterable(itertools.chain.from_iterable(nl)))

#     return nl,l1,l2

# def reconstruct2(nl, l1, l2):
#     """
#     To reconstruct Python List of lists / numpy arrays. Inverse operation of FLATTEN() above.
#     Usage: L_reconstructed = reconstruct2(L_flat,l1,l2)
#     Source: http://stackoverflow.com/questions/27982432/flattening-and-unflattening-a-nested-list-of-numpy-arrays
#     """
#     return np.split(np.split(nl,np.cumsum(l1)),np.cumsum(l2))[:-1]
