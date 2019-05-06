import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Conv1D, Dropout, Lambda, Flatten
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
import keras.backend as K
import tensorflow as tf
import os
from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasRegressor
import numpy as np
from math import tanh, atanh
from scipy import interpolate

import matplotlib
#matplotlib.use("agg")
from matplotlib import pyplot as plt

from generate_params import read_params, read_params_cont_bath, name


def meta(beta):
    return "_beta_{}".format(beta)

def rescale(data, alpha=4):
    def f(x):
        rho = 0.5 * (1 - tanh(alpha))
        return (-rho + 0.5 * (tanh(alpha * (2 * x - 1)) + 1)) / (1 - 2 * rho)
    args = np.linspace(0, 1, num=len(data), endpoint=True)
    tck = interpolate.splrep(args, data, s=0.000001)
    rescaled_args = np.array([f(x) for x in args])
    new_data = interpolate.splev(rescaled_args, tck, der=0)
    return np.array(new_data)

def back_rescale(data, alpha=5):
    def inv_f(x):
        rho = 0.5 * (1 - tanh(alpha))
        return atanh(2 * (1 - 2 * rho) * x + 2 * rho - 1) / (2 * alpha) + 0.5
    args = np.linspace(0, 1, num=len(data), endpoint=True)
    tck = interpolate.splrep(args, data, s=0.000001)
    rescaled_args = np.array([inv_f(x) for x in args])
    new_data = interpolate.splev(rescaled_args, tck, der=0)
    return np.array(new_data)

def transform(data, how="shift_and_rescale"):
    for i in range(len(data)):
        data[i, :] = rescale(data[i, :])
    if how == "minus":
        return -data
    if how == "shift":
        return data + 0.5
    if how == "shift_and_rescale":
        return (data + 0.5) * 2
    if how == "shift_and minus":
        return -(data + 0.5)
    if how == "shift_and_rescale_and_minus":
        return -(data + 0.5) * 2
    else:
        return data

def back_transform(data, how="shift_and_rescale"):
    if how == "minus":
        data = -data
    if how == "shift":
        data = data - 0.5
    if how == "shift_and_rescale":
        data = 0.5 * data - 0.5
    if how == "shift_and minus":
        data = data - 0.5
    if how == "shift_and_rescale_and_minus":
        data = -0.5 * data - 0.5
    else:
        data = data
    if hasattr(data, '__len__'):
        if len(data.shape) == 1:
            data[:] = back_rescale(data)
        else:
            for i in range(len(data)):
                data[i, :] = back_rescale(data[i, :])
    return data

def generate_data(meta=meta(1),
                  n_tau=51,
                  n_tau_in_data=201,
                  n_per_file=1000,
                  train_files=[0, 1, 2, 3, 4, 5, 6],
                  val_files=[7, 8, 9],
                  dtype='float16',
                  preprocessing="shift_and_rescale",
                  CNN=False,
                  parent="data/"):

    n_input = 2 * n_tau  # weak & strong coupling GF
    n_output = n_tau  # approximation to an exact GF

    skip = (n_tau_in_data - 1) // (n_tau - 1)

    def get_data(prefix, n):
        data = np.loadtxt(parent + prefix + meta + "_{}.csv".format(n),
                          delimiter=",")[:, ::skip]
        return transform(data, how=preprocessing)

    n_train = len(train_files) * n_per_file * 2
    n_test = len(val_files) * n_per_file * 2

    X_train = np.zeros((n_train, n_input), dtype=dtype)
    Y_train = np.zeros((n_train, n_output), dtype=dtype)
    X_test = np.zeros((n_test, n_input), dtype=dtype)
    Y_test = np.zeros((n_test, n_output), dtype=dtype)

    def fill_with_data(X, Y, files):
        for i, n in enumerate(files):
            row_start = (2 * i) * n_per_file
            row_middle = (2 * i + 1) * n_per_file
            row_end = (2 * i + 2) * n_per_file
            X[row_start:row_middle, :n_tau] = get_data("G_weak", n)
            X[row_start:row_middle, n_tau:] = get_data("G_strong", n)
            Y[row_start:row_middle, :] = get_data("G_ED", n)
            # Data augmentation - perform particle-hole transformation on gf
            X[row_middle:row_end, :n_tau] = np.fliplr(X_train[row_start:row_middle, :n_tau])
            X[row_middle:row_end, n_tau:] = np.fliplr(X_train[row_start:row_middle, n_tau:])
            Y[row_middle:row_end, :] = np.fliplr(Y_train[row_start:row_middle, :])

    fill_with_data(X_train, Y_train, train_files)
    fill_with_data(X_test, Y_test, val_files)

    if CNN == True:
        if keras.backend.image_data_format() == 'channels_first':
            X_train = X_train.reshape(X_train.shape[0], 2, n_tau)
            X_test = X_test.reshape(X_test.shape[0], 2, n_tau)
            input_shape = (2, n_tau)
        else:
            X_train = X_train.reshape(X_train.shape[0], n_tau, 2)
            X_test = X_test.reshape(X_test.shape[0], n_tau, 2)
            input_shape = (n_tau, 2)
    else:
        input_shape = (2 * n_tau, )

    return X_train, Y_train, X_test, Y_test, input_shape

# Create NN
def max_error(y_true, y_pred):
    return K.max(K.abs(y_true - y_pred))

def boundary_cond(y_true, y_pred):
    preprocessing = 'shift_and_rescale'
    return K.max(K.abs(-(back_transform(y_pred[:, 0], how=preprocessing)
             + back_transform(y_pred[:, -1], how=preprocessing)) - 1.))

def create_model(preprocessing='shift_and_rescale',
                 input_shape=(2 * 51, ),
                 n_output=51,
                 activation='elu',
                 n_hidden_layers=2,
                 n_neurons=51,
                 optimizer=Nadam,
                 init_mode='uniform',
                 learn_rate=0.0002,
                 momentum=0.9,
                 CNN=False,
                 kernel_size=5,
                 n_filters=8):

    model = Sequential()
    if CNN == False:
        model.add(Dense(n_neurons, input_shape=input_shape,
                        activation=activation, kernel_initializer=init_mode))
        for i in range(n_hidden_layers - 1):
            model.add(Dense(n_neurons, activation=activation, kernel_initializer=init_mode))
        model.add(Dense(n_output, activation=activation, kernel_initializer=init_mode))
    else:
        model.add(Conv1D(n_filters, input_shape=input_shape, kernel_size=kernel_size,
                  activation=activation, kernel_initializer=init_mode, padding="valid"))
        model.add(Flatten())
        for i in range(n_hidden_layers - 1):
            model.add(Dense(n_neurons, activation=activation, kernel_initializer=init_mode))
        model.add(Dense(n_output, activation=activation, kernel_initializer=init_mode))
        # model.add(Conv1D(1, kernel_size=kernel_size, activation=activation, kernel_initializer=init_mode, padding="same"))
        # model.add(Flatten())

    # Create optimizer
    if optimizer == SGD:
        optimizer = SGD(lr=learn_rate, momentum=momentum)
    else:
        optimizer = optimizer(lr=learn_rate)
    loss = keras.losses.mean_squared_error


    metrics = ['mae', max_error, boundary_cond]

    # Compile model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return model

# Optimize hyperparameters of the model
def perform_grid_search(beta=1, n_tau=51, preprocessing='shift_and_rescale'):
    X_train, Y_train, X_test, Y_test, input_shape = generate_data(meta=meta(beta),
                                                                  n_tau=n_tau,
                                                                  preprocessing=preprocessing,
                                                                  train_files=range(10),
                                                                  val_files=[])

    model = KerasRegressor(build_fn=create_model, epochs=100, verbose=False)
    optimizers = [Adamax, Nadam]
    learn_rates = [0.0001, 0.0005, 0.001]
    batch_sizes = [8, 16, 32]
    activations = ['elu', 'tanh']
    param_grid = dict(optimizer=optimizers, activation=activations,
                      batch_size=batch_sizes, learn_rate=learn_rates)

    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1)
    grid_result = grid.fit(X_train, Y_train)

    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
            print("%f (%f) with: %r" % (mean, stdev, param))

# Run and evaluate fit
def perform_learning(beta=1, n_tau=51, preprocessing='shift_and_rescale', CNN=False, train_files=range(1,10), val_files=[0]):

    X_train, Y_train, X_test, Y_test, input_shape = generate_data(meta=meta(beta),
                                                                  n_tau=n_tau,
                                                                  preprocessing=preprocessing,
                                                                  train_files=train_files,
                                                                  val_files=val_files,
                                                                  CNN=CNN)
    model = create_model(preprocessing=preprocessing,
                         input_shape=input_shape,
                         n_output=n_tau,
                         activation='elu',
                         optimizer=Nadam,
                         learn_rate=2 * 0.0002,
                         n_neurons=n_tau,
                         n_hidden_layers=8,
                         CNN=CNN)
    history = model.fit(X_train, Y_train,
                        batch_size = 8,
                        epochs = 80,
                        validation_data=(X_test, Y_test),
                        verbose=True)
    score = model.evaluate(X_test, Y_test, verbose=1)

    # print performance
    print('Test MSE:', score[0])
    print('Test MAE:', score[1])
    print('Test max error:', score[2])
    print('Test boundary condition', score[3])

    # summarize history for max error
    for metric in ['mean_absolute_error', 'max_error', 'boundary_cond']:
        plt.plot(history.history[metric])
        plt.plot(history.history['val_' + metric])
        plt.ylabel(metric)
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='best')
        plt.savefig(str(metric) + ".png")
        if False:
            plt.show()
        plt.close()
    return model

# Prediction
def predict(model, beta=1, n_tau=51, preprocessing='shift_and_rescale', samples=range(10),
            gen_data_kwargs={}, cont_bath=False):
    X_train, Y_train, X_test, Y_test, input_shape  = generate_data(**gen_data_kwargs)
    Y = model.predict(X_test[list(samples)])
    tau = np.linspace(0, beta, n_tau, endpoint=True)
    # We assume we draw samples from file 39 
    if not cont_bath:
        params = read_params(name("params", beta, 39))
    else:
        params = read_params_cont_bath(name("params", beta, 0, parent="data_cont_bath/"))
    params = [params[i] for i in samples]

    for i, y, p in zip(samples, Y, params):
        error = max(np.abs(Y_test[i] - y))
        bc = back_transform(y, how=preprocessing)
        bc = bc[0] + bc[-1]
        plt.axhline(0, color='k', ls=':', lw=1)
        plt.plot(tau, back_transform(Y_test[i], how=preprocessing), '-', label="ED" if not cont_bath else "QMC")
        plt.plot(tau, back_transform(y, how=preprocessing), '--', label="NN")
        plt.plot(tau, back_transform(X_test[i, :n_tau], how=preprocessing), '-.', label="weak")
        plt.plot(tau, back_transform(X_test[i, n_tau:], how=preprocessing), '-.', label="strong")
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$G(\tau)$")
        plt.ylim([-0.7,0.05])
        plt.legend(loc='best')
        #plt.title("Sample {}".format(i))
        title_str = r"$U={:.2f},\varepsilon={:.2f}".format(p["U"], p["eps"])
        if not cont_bath:
            title_str += r"$" + "\n"
            title_str += r"$(\epsilon_i, V_i)\in \{"
            for e, V in zip(p["e_list"], p["V_list"]):
                title_str += r"({:.2f}, {:.2f}),".format(e, V)
            title_str = title_str[:-1]
            title_str += r"\}$"
        else:
            title_str += r", D={:.2f}$".format(p["D"])
        title_str += "\n max error = {:.5f} \n".format(error / 2)
        title_str += r"$G(0) + G(\beta) = {:.5f}$".format(bc)
        plt.title(title_str, fontsize="medium")
        plt.tight_layout()
        if not cont_bath:
            plt.savefig("plots/sample{}.png".format(i), dpi=300)
        else:
            plt.savefig("plots_cont_bath/sample{}.png".format(i), dpi=300)
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        #plt.show()
        plt.close()

def evaluate(model, beta=1, gen_data_kwargs={}):
    X_train, Y_train, X_test, Y_test, input_shape  = generate_data(**gen_data_kwargs)
    score = model.evaluate(X_train, Y_train, batch_size=80000)
    return score

def plot_input(beta=1, n_tau=51, preprocessing='shift_and_rescale', samples=range(10),
               gen_data_kwargs={}):
    X_train, Y_train, X_test, Y_test, input_shape  = generate_data(**gen_data_kwargs)
    tau = np.linspace(0, beta, n_tau, endpoint=True)
    # We assume we draw samples from file 0
    params = read_params(name("params", beta, 0))
    params = [params[i] for i in samples]

    for i, p in zip(samples, params):
        plt.plot(tau, Y_test[i], 'o')
        plt.plot(tau, X_test[i, :n_tau], 'o', c='C2')
        plt.plot(tau, X_test[i, n_tau:], 'o', c='C3')
        plt.xlabel(r"$\tau$")
        plt.ylabel(r"$F(\tau) = \left[ G(\tau) + 0.5 \right] \cdot 2$")
        plt.ylim([-0.4,0.8])
        plt.legend(["ED", "weak", "strong"], loc='best')
        #plt.title("Sample {}".format(i))
        title_str = r"$U={:.2f},\varepsilon={:.2f}".format(p["U"], p["eps"])
        title_str += r"$" + "\n"
        title_str += r"$(\epsilon_i, V_i)\in \{"
        for e, V in zip(p["e_list"], p["V_list"]):
            title_str += r"({:.2f}, {:.2f}),".format(e, V)
        title_str = title_str[:-1]
        title_str += r"\}$"
        plt.title(title_str, fontsize="medium")
        plt.tight_layout()
        plt.savefig("plots/data{}.png".format(i), dpi=300)
        #mng = plt.get_current_fig_manager()
        #mng.resize(*mng.window.maxsize())
        #plt.show()
        plt.close()



if __name__ == "__main__":

    #from generate_params import beta
    beta = 20

    # Multithreading - optimizations
    n_thread = int(os.environ["OMP_NUM_THREADS"])
    config = tf.ConfigProto(intra_op_parallelism_threads=n_thread,
                            inter_op_parallelism_threads=2,
                            allow_soft_placement=True,
                            device_count = {'CPU': n_thread})
    session = tf.Session(config=config)
    K.set_session(session)

    os.environ["KMP_BLOCKTIME"] = "30"
    os.environ["KMP_SETTINGS"] = "1"
    os.environ["KMP_AFFINITY"]= "granularity=fine,verbose,compact,1,0"

    # perform_grid_search()

    # model = perform_learning(beta, n_tau=101, train_files=range(0, 32), val_files=range(32, 40))
    # model.save('model.h5')

    # Show how the test GF are predicted
    model = keras.models.load_model('model.h5', custom_objects={'max_error': max_error, 'boundary_cond': boundary_cond})
    predict(model, beta=beta, n_tau=101, samples=range(0, 50),
        gen_data_kwargs=dict(n_tau=101, meta=meta(beta), train_files=[39], val_files=[39]))

    # Perform comparison against benchmark QMC data
    predict(model, beta=beta, n_tau = 101, samples=range(0, 50), cont_bath=True,
            gen_data_kwargs=dict(n_per_file=50, n_tau=101, meta=meta(beta), train_files=[0], val_files=[0], parent="data_cont_bath/"))

    score_train = evaluate(model, gen_data_kwargs=dict(n_tau=101, meta=meta(beta), train_files=range(0,32), val_files=[0]))
    score_test = evaluate(model, gen_data_kwargs=dict(n_tau=101, meta=meta(beta), train_files=range(32,40), val_files=[0]))
    score_new = evaluate(model, gen_data_kwargs=dict(n_tau=101, meta=meta(beta), n_per_file=50, train_files=[0], val_files=[0], parent="data_cont_bath/"))

    with open("scores.txt", "w") as f:
        for score, m_name in zip([score_train, score_test, score_new], ["train", "test", "continuous bath"]):
            f.write(m_name + "\n")
            for s, metric in zip(score, ['MSE', 'MAE', 'max error', 'boundary cond']):
                if metric in ['MAE', 'max error']:
                    s = s / 2
                f.write(metric + ": " + str(s) + " ")
            f.write("\n")

