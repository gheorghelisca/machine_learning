import sys
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

import matplotlib.pyplot as plt

training_file_name = None
groundtruth_file_name = None

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
    x_train = training_data[:,0:2]
    y_train = training_data[:,2]

    # replace the class label -1 with 0
    y_train = np.where(y_train==-1, 0, y_train)
#    print y_train
#    y_train = to_categorical(y_train)
#    print y_train
    

    # create a keras model
    model = Sequential()
    model.add(Dense(32, input_shape=(2,), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(32, activation='relu'))    
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.001), metrics = ['accuracy'])

    # start training
    # model.fit((x_train, y_train, batch_size=32, epochs=100, validation_data=(X_val, Y_val))
    model.fit(x_train, y_train, epochs=100)
    
    return model


def evaluate_model(model, evaluation_data, groundtruth_data):
    '''TODO(lisca):
    '''
    groundtruth_data = np.where(groundtruth_data==-1, 0, groundtruth_data)

    evaluation = model.evaluate(evaluation_data, groundtruth_data)
    print evaluation

    predicted_data = model.predict(evaluation_data)
    print predicted_data

    # plot
    plt.plot(evaluation_data, groundtruth_data, 'bs', evaluation_data, predicted_data, 'rs')
    plt.xlabel('Cassification Results')
    plt.savefig('./plots/classification.png')
    plt.show()
    

def run():
    '''TODO(lisca):
    '''
    training_data, evaluation_data, groundtruth_data = read_data()
    
    trained_model = train_model(training_data)
    evaluate_model(trained_model, evaluation_data, groundtruth_data)

if __name__ == '__main__':
    training_file_name = sys.argv[1]
    groundtruth_file_name = sys.argv[2]
    run()
