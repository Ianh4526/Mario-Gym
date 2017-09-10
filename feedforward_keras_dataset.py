import time
import numpy as np
from matplotlib import pyplot as plt
from keras.utils import np_utils
import keras.callbacks as cb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop
from keras.datasets import mnist
from numpy import genfromtxt


class LossHistory(cb.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        batch_loss = logs.get('loss')
        self.losses.append(batch_loss)


def load_data():


    print 'Loading data...1'
    #with open("MDL.csv") as f:
    #mario_level_data = np.loadtxt('MDL.csv', delimiter=',', dtype=np.float32)
    #mario_level_data = csv.reader(f,delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    
    #mario_level_data = pd.read_csv('MDL.csv', delimiter=',', engine= 'python' )
    mario_level_data = np.genfromtxt('MDL.csv', delimiter=',',max_rows=400)
    print 'Loading data...2'
    data_y = mario_level_data[:,[172032,172033,172034,172035,172036,172037]]
    print 'Loading data...3'
    data_x = mario_level_data[:,:172031]#344063 colums usecols=[1,3]
    print 'Loading data...4'

    split_number = len(data_x)/2
    (X_train, X_test) = data_x[:split_number,:], data_x[split_number:,:]
    (y_train, y_test) = data_y[:split_number,:], data_y[split_number:,:]

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255
    print(mario_level_data.shape)
    print 'Data loaded.'
    return [X_train, X_test, y_train, y_test]


def init_model():
    start_time = time.time()
    print 'Compiling Model ... '
    model = Sequential()
    model.add(Dense(500, input_dim=172031))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(300))
    model.add(Activation('relu'))
    model.add(Dropout(0.4))
    model.add(Dense(6))
    model.add(Activation('softmax'))

    rms = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=rms, metrics=['accuracy'])
    print 'Model compield in {0} seconds'.format(time.time() - start_time)
    return model


def run_network(data=None, model=None, epochs=500, batch=256):
    try:
        start_time = time.time()
        if data is None:
            X_train, X_test, y_train, y_test = load_data()
        else:
            X_train, X_test, y_train, y_test = data

        if model is None:
            model = init_model()

        history = LossHistory()

        print 'Training model...'
        model.fit(X_train, y_train, nb_epoch=epochs, batch_size=batch,
                  callbacks=[history],
                  validation_data=(X_test, y_test), verbose=2)

        print "Training duration : {0}".format(time.time() - start_time)
        score = model.evaluate(X_test, y_test, batch_size=16)

        print "Network's test score [loss, accuracy]: {0}".format(score)

# serialize model to JSON
        model_json = model.to_json()
        with open("GymMario.json", "w") as json_file:
        	json_file.write(model_json)
        # serialize weights to HDF5
        model.save_weights("GymMario.h5")
        print("Saved model to disk")

        return model, history.losses
       


    except KeyboardInterrupt:
        print ' KeyboardInterrupt'
        return model, history.losses


def plot_losses(losses):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(losses)
    ax.set_title('Loss per batch')
    fig.show()



run_network()

