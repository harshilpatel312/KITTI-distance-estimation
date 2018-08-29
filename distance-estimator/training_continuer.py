import pandas as pd
import time
import argparse

from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

argparser = argparse.ArgumentParser(description='Continue training a model')
argparser.add_argument('-m', '--modelname',
                       help='model name (.json)')
argparser.add_argument('-w', '--weights',
                       help='weights filename (.h5)')

args = argparser.parse_args()

# parse arguments
MODEL = args.modelname
WEIGHTS = args.weights

df_train = pd.read_csv('data/train.csv')
df_test = pd.read_csv('data/test.csv')

X_train = df_train[['xmin', 'ymin', 'xmax', 'ymax']].values
y_train = df_train[['zloc']].values

# standardized data
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
y_train = scalar.fit_transform(y_train)

# load json and create model
json_file = open('generated_files/{}.json'.format(MODEL), 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("generated_files/{}.h5".format(WEIGHTS))
print("Loaded model from disk")

# compile and train model
parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='mean_squared_error', optimizer='adam')

modelname = "model@{}".format(int(time.time()))
tensorboard = TensorBoard(log_dir="logs/{}".format(modelname))

parallel_model.fit(X_train, y_train, validation_split=0.1, epochs=5000, batch_size=2048, callbacks=[tensorboard], verbose=1)

# ----------- save model and weights ----------- #
model_json = model.to_json()
with open("generated_files/{}.json".format(modelname), "w") as json_file:
    json_file.write(model_json)

model.save_weights("generated_files/{}.h5".format(modelname))
print("Saved model to disk")
