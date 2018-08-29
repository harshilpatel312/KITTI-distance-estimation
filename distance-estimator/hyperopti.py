import time
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import multi_gpu_model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice

def data():
    # ----------- import data and perform scaling ----------- #
    df_train = pd.read_csv('data/train.csv')
    df_test = pd.read_csv('data/test.csv')

    X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_train = df_train[['zloc']].values

    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    scalar = StandardScaler()
    x_train = scalar.fit_transform(X_train)
    y_train = scalar.fit_transform(y_train)
    x_test = scalar.fit_transform(X_test)
    y_test = scalar.fit_transform(y_test)

    return x_train, y_train, x_test, y_test

def create_model(x_train, y_train, x_test, y_test):
    # ----------- define model ----------- #
    model = Sequential()

    model.add(Dense(6, input_shape=(4,)))

    model.add(Dense({{choice([4, 5])}}, activation='relu'))

    if {{choice(['three', 'four'])}} == 'four':
        model.add(Dense(2, activation='relu'))

    model.add(Dense(1))

    model = multi_gpu_model(model, gpus=2)
    model.compile(loss='mean_squared_error', metrics=['mae'],
                  optimizer={{choice(['rmsprop', 'adam', 'sgd'])}})

    # ----------- define callbacks ----------- #
    earlyStopping = EarlyStopping(monitor='loss', patience=10, verbose=0, mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=7,
								       verbose=1, epsilon=1e-5, mode='min')
    tensorboard = TensorBoard(log_dir="logs/model@{}".format(int(time.time())))

    # ----------- start training ----------- #
    model.fit(x_train, y_train,
              batch_size={{choice([1024, 2048])}},
              epochs=5000,
              callbacks=[tensorboard],
              verbose=1,
              validation_split=0.1)

    # ----------- evaluate model ----------- #
    score, acc = model.evaluate(x_test, y_test, verbose=1)
    print('Test accuracy:', score)

    # ----------- save model and weights ----------- #
    '''
    # NOTE: WAITING ON HYPERAS ISSUE
    model_json = model.to_json()
    modelname = "{}".format(int(time.time()))
    with open("generated_files/model_{}.json".format(modelname), "w") as json_file:
        json_file.write(model_json)

    model.save_weights("generated_files/model_{}.h5".format(modelname))
    print("Saved model to disk")
    '''

    return {'loss': score, 'status': STATUS_OK, 'model': model}

if __name__ == '__main__':
    trials = Trials()
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=3,
                                          trials=trials,
                                          eval_space=True)

    X_train, Y_train, X_test, Y_test = data()
    print("Evaluation of best performing model:", best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
