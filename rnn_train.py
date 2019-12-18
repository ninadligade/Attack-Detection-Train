import time

from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.layers import Dropout
from tensorflow.python.keras.layers import GRU
from tensorflow.python.keras.layers import LSTM

from Ancillary_Functions import load_from_file, generate_samples
from TimeHistory import TimeHistory


class RNN:

    def create_rnn_model(self, n_timesteps, n_features, n_outputs, model_type):
        model = Sequential()
        if model_type == "GRU":
            model.add(GRU(100, input_shape=(n_timesteps, n_features)))
        else:
            model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
        model.add(Dropout(0.5))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(n_outputs, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.summary()

        return model

    def train_model(self, input_folder):
        X_train, y_train, times_train = load_from_file(input_folder, "x3-train.csv", "y3-train.csv", "t3-train.csv")
        X_val, y_val, times_val = load_from_file(input_folder, "x3-validation.csv", "y3-validation.csv",
                                                 "t3-validation.csv")
        X_test, y_test, times_test = load_from_file(input_folder, "x3-test.csv", "y3-test.csv", "t3-test.csv")
        trainX, trainy = generate_samples(X_train, y_train)
        testX, testy = generate_samples(X_test, y_test)
        valX, valY = generate_samples(X_val, y_val)

        verbose, epochs, batch_size = 1, 1, 100
        n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]

        model = self.create_rnn_model(n_timesteps, n_features, n_outputs)

        time_callback = TimeHistory()
        model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=0, validation_data=(valX, valY),
                  callbacks=[time_callback])

        this_time = time.time()
        model.predict_classes(testX[:10000])
        print("{0:.20f}".format(time.time() - this_time))
        model.save("rnn-keras-model.h5")
        this_time = time.time()
        print("{0:.20f}".format(time.time() - this_time))
        _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
        print(accuracy)
        return accuracy

