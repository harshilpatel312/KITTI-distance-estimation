from comet_ml import Experiment

from keras.models import model_from_json
import pandas as pd
from sklearn.preprocessing import StandardScaler

experiment = Experiment(api_key="nSwZg5gxVt0RWFAHrZjvYeKcn")

df_test = pd.read_csv('data/test.csv')
df_result = df_test
df_result['zloc_pred'] = -100000

X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
y_test = df_test[['zloc']].values

# standardized data
scalar = StandardScaler()
X_test = scalar.fit_transform(X_test)
y_test = scalar.fit_transform(y_test)

# load json and create model
json_file = open('generated_files/model@1535477330.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json( loaded_model_json )

# load weights into new model
loaded_model.load_weights("generated_files/model@1535477330.h5")
print("Loaded model from disk")

# evaluate loaded model on test data
loaded_model.compile(loss='mean_squared_error', optimizer='adam')
y_pred = loaded_model.predict(X_test)
y_pred = scalar.inverse_transform(y_pred)
y_test = scalar.inverse_transform(y_test)

for idx, row in df_result.iterrows():
    df_result.at[idx, 'zloc_pred'] = y_pred[idx]

df_result.to_csv('data/predictions.csv', index=False)
