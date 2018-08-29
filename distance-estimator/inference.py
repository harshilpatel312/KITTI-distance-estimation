import pandas as pd
import argparse

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler

argparser = argparse.ArgumentParser(description='Get predictions of test set')
argparser.add_argument('-m', '--modelname',
                       help='model name (.json)')
argparser.add_argument('-w', '--weights',
                       help='weights filename (.h5)')

args = argparser.parse_args()

# parse arguments
MODEL = args.modelname
WEIGHTS = args.weights

def main():
    # get data
    df_test = pd.read_csv('data/test.csv')
    X_test = df_test[['xmin', 'ymin', 'xmax', 'ymax']].values
    y_test = df_test[['zloc']].values

    # standardized data
    scalar = StandardScaler()
    X_test = scalar.fit_transform(X_test)
    y_test = scalar.fit_transform(y_test)

    # load json and create model
    json_file = open('generated_files/{}.json'.format(MODEL), 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json( loaded_model_json )

    # load weights into new model
    loaded_model.load_weights("generated_files/{}.h5".format(WEIGHTS))
    print("Loaded model from disk")

    # evaluate loaded model on test data
    loaded_model.compile(loss='mean_squared_error', optimizer='adam')
    y_pred = loaded_model.predict(X_test)

    # scale up predictions to original values
    y_pred = scalar.inverse_transform(y_pred)
    y_test = scalar.inverse_transform(y_test)

    # save predictions
    df_result = df_test
    df_result['zloc_pred'] = -100000

    for idx, row in df_result.iterrows():
        df_result.at[idx, 'zloc_pred'] = y_pred[idx]

    df_result.to_csv('data/predictions.csv', index=False)

if __name__ == '__main__':
    main()
