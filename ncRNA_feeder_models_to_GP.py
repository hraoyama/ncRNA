import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
import site
import pandas as pd
import numpy as np
import matplotlib
import gpflow
from gpflow.utilities import ops, print_summary, set_trainable
from gpflow.config import set_default_float, default_float, set_default_summary_fmt
from gpflow.ci_utils import ci_niter

matplotlib.use('Qt5Agg')

site.addsitedir("D:/Code/ncRNA")
pd.set_option('display.width', 400)
pd.set_option('display.max_columns', 40)

from ncRNA_utils import getData, coShuffled_vectors, reverse_tensor, get_combined_features_from_models, getE2eData


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


def reverse_one_hot(Y_input):
    return np.apply_along_axis(np.argmax, 1, Y_input) + 1


def get_gp_acc(mGP, data_gp):
    (Y_test_GP_mean, Y_test_GP_variance) = mGP.predict_y(data_gp[0])
    Y_test_GP_mean = Y_test_GP_mean.numpy()
    Y_test_pred = []
    for i in range(data_gp[0].shape[0]):
        Y_test_pred.append(np.argmax(Y_test_GP_mean[i,]))
    diffs = Y_test_pred - data_gp[1]
    test_acc = sum(1 for x in diffs if x == 0) / data_gp[0].shape[0]
    return test_acc, Y_test_pred, (Y_test_GP_mean, Y_test_GP_variance)


if __name__ == "__main__":
    # 'new' data
    X_train_1000e, Y_train_1000e, X_test_1000e, Y_test_1000e, X_val_1000e, Y_val_1000e = getE2eData(is500=False,
                                                                                                    include_secondary=False)
    # CNN no secondary
    mCNN2_1000 = load_model("./models/CNN_baseline.h5")
    mCNN2_1000._name = "cnn_finalist"
    # RNN no secondary
    mRNN_1000l = load_model("./models/PureRNN_no_sec.h5", custom_objects=SeqWeightedAttention.get_custom_objects())
    mRNN_1000l._name = "rnn_finalist"

    # tf.keras.utils.plot_model(mRNN_1000l, show_shapes=True, to_file="C:/temp/test.png")
    # only the final layers...
    to_combine_last_layers = [
        (mRNN_1000l, "dense_3", None),
        (mCNN2_1000, "dense_11", None)
    ]

    combined_models, data_train_ll, data_test_ll, data_access_ll = get_combined_features_from_models(
        to_combine_last_layers,
        X_train_1000e, Y_train_1000e,
        X_test_1000e, Y_test_1000e,
        reverse_one_hot=False)

    # reproducibility:
    np.random.seed(0)
    tf.random.set_seed(123)

    data_gp_train = (data_train_ll[0], reverse_one_hot(data_train_ll[1]))
    data_gp_test = (data_test_ll[0], reverse_one_hot(data_test_ll[1]))
    data_gp_val = (get_features_from_combined_models(combined_models, X_val_1000e), reverse_one_hot(Y_val_1000e))

    # sum kernel: Matern32 + White
    lengthscales = tf.convert_to_tensor([1.0] * data_gp_train[0].shape[1], dtype=default_float())
    # kernel = gpflow.kernels.Matern32(lengthscales=lengthscales) #+ gpflow.kernels.White(variance=0.01)
    kernel = gpflow.kernels.Matern52(lengthscales=lengthscales)

    # Robustmax Multiclass Likelihood
    invlink = gpflow.likelihoods.RobustMax(13)  # Robustmax inverse link function
    likelihood = gpflow.likelihoods.MultiClass(13, invlink=invlink)  # Multiclass likelihood
    # M = 20  # Number of inducing locations
    # Z = data_gp_train[::M].copy()  # inducing inputs CHECK DIMENSIONS
    M = 60
    # Z = data_gp_train[sorted(random.sample(range(0,data_gp_train.shape[0]),M)),:].copy()
    Z = data_gp_train[0][::M].copy()  # inducing inputs CHECK DIMENSIONS
    # Number of inducing locations
    Z.shape[0]

    mGP = gpflow.models.SVGP(
        kernel=kernel,
        likelihood=likelihood,
        inducing_variable=Z,
        num_latent_gps=13,
        whiten=False,
        q_diag=False,
    )

    # Only train the variational parameters
    # set_trainable(mGP.kernel.kernels[1].variance, True)
    set_trainable(mGP.inducing_variable, False)

    # print_summary(mGP)
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        mGP.training_loss_closure(data_gp_train), mGP.trainable_variables,
        options=dict(maxiter=ci_niter(25000), method="trust-ncg")
    )
    # print_summary(mGP)
    test_acc, Y_test_pred, (gp_mean, gp_var) = get_gp_acc(mGP, data_gp_test)
    test_val, Y_test_val, (gp_mean, gp_var) = get_gp_acc(mGP, data_gp_val)
