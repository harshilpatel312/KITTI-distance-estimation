import pandas as pd
import numpy as np
import time

import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import multi_gpu_model
from sklearn.preprocessing import StandardScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

def main():
	# ----------- import data and scaling ----------- #
	df_train = pd.read_csv('data/train.csv')
	df_test = pd.read_csv('data/test.csv')

	X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_train = df_train[['zloc']].values

	X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
	y_test = df_test[['zloc']].values

	# standardized data
	scalar = StandardScaler()
	X_train = scalar.fit_transform(X_train)
	y_train = scalar.fit_transform(y_train)

	# ----------- create model ----------- #
	model = Sequential()
	model.add(Dense(6, input_dim=4, kernel_initializer='normal', activation='relu'))
	model.add(Dense(5, kernel_initializer='normal', activation='relu'))
	model.add(Dense(2, kernel_initializer='normal', activation='relu'))
	model.add(Dense(1, kernel_initializer='normal'))

	model = multi_gpu_model(model, gpus=2)
	model.compile(loss='mean_squared_error', optimizer='adam')

	# ----------- define callbacks ----------- #
	earlyStopping = EarlyStopping(monitor='val_loss', patience=10, verbose=0, mode='min')
	reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7,
									   verbose=1, epsilon=1e-4, mode='min')
	modelname = "model@{}".format(int(time.time()))
	tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

	# ----------- start training ----------- #
	history = model.fit(X_train, y_train,
	 							 validation_split=0.1, epochs=5000, batch_size=2048,
	 							 callbacks=[tensorboard], verbose=1)

	# ----------- save model and weights ----------- #
	model_json = model.to_json()
	with open("generated_files/{}.json".format(modelname), "w") as json_file:
	    json_file.write(model_json)

	model.save_weights("generated_files/{}.h5".format(modelname))
	print("Saved model to disk")

if __name__ == '__main__':
	main()
