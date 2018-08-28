import pickle
import matplotlib.pyplot as plt
import argparse

argparser = argparse.ArgumentParser(description='Set training parameters')
argparser.add_argument('-f', '--filename',
                       help='name of filename to load history')

args = argparser.parse_args()

# parse arguments
NAME = args.filename

history = pickle.load(open(NAME, 'rb'))

# summarize history for accuracy
try:
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
except:
    pass

# summarize history for loss
try:
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
except:
    pass
