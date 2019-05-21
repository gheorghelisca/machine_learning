import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers

import matplotlib.pyplot as plt

training_file_name = None
groundtruth_file_name = None

tm = None

def read_data():
    '''TODO(lisca):
    '''
    training_data = []
    evaluation_data = []
    groundtruth_data = []    
    with open(training_file_name) as f:
        training_data_flag = True
        for line in f:
            line_array = np.fromstring(line, dtype=float, sep=',')
            if training_data_flag == True:
                if np.count_nonzero(line_array) > 0:
                    training_data.append(line_array)
                else:
                    training_data_flag = False
            else:
                evaluation_data.append(line_array)

    with open(groundtruth_file_name) as f:
        for line in f:
            line_array = np.fromstring(line, dtype=float, sep=',')
            groundtruth_data.append(line_array[0])

    return np.array(training_data), np.array(evaluation_data), np.array(groundtruth_data)

def train_model(training_data):
    '''TODO(lisca):
    '''
    x_train = training_data[:,0:1]
    y_train = training_data[:,1]

    # create a keras model
    model = Sequential()
    model.add(Dense(32, input_shape=(1,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1,))
    
    model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error', metrics=['mse', 'mae'])

#    sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
#    model.compile(optimizer=sgd, loss='mean_squared_error', metrics=['acc'])

    model.fit(x_train, y_train, epochs=100)

    return model


def evaluate_model(model, evaluation_data, groundtruth_data):
    '''TODO(lisca):
    '''
    evaluation = model.evaluate(evaluation_data, groundtruth_data)

    mse = model.history.history['mean_squared_error']
    mae = model.history.history['mean_absolute_error']

    print mse
    print mae

    predicted_data = model.predict(evaluation_data)

    # plot
    plt.plot(evaluation_data, groundtruth_data, 'bs', evaluation_data, predicted_data, 'rs')
    plt.xlabel('Regression Results')
    plt.savefig('./plots/regression.png')
    plt.show()
    

def run():
    '''TODO(lisca):
    '''
    training_data, evaluation_data, groundtruth_data = read_data()
    
    trained_model = train_model(training_data)
    evaluation = evaluate_model(trained_model, evaluation_data, groundtruth_data)

if __name__ == '__main__':
    training_file_name = sys.argv[1]
    groundtruth_file_name = sys.argv[2]
    run()
