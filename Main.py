from tensorflow.python.keras.models import load_model

from Ancillary_Functions import load_from_file, generate_samples
from rnn_train import RNN

MAX_LENGTH = 11
SAVED_MODEL_NAME="rnn-keras-model.h5"


def use_and_analyse_input_model(input_dir):
    X_test, y_test, times_test = load_from_file(input_dir, "x3-test.csv", "y3-test.csv", "t3-test.csv")
    testX, testy = generate_samples(X_test, y_test)

    model = load_model(SAVED_MODEL_NAME)

    predicted_classes = model.predict_classes(testX, verbose=1)

    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for i in range(len(predicted_classes)):
        point_prediction = predicted_classes[i]
        point_reality = y_test[i + MAX_LENGTH - 1]

        if point_reality == 1 and point_prediction == 1:
            true_positives += 1
        elif point_reality == 0 and point_prediction == 0:
            true_negatives += 1
        elif point_reality == 1 and point_prediction == 0:
            false_negatives += 1
        else:
            false_positives += 1

    print("Number of false positives in data: " + str(false_positives))
    print("Number of false negatives in data: " + str(false_negatives))
    print("Number of true positives predicted: " + str(true_positives))
    print("Number of true negatives predicted: " + str(true_negatives))


if __name__ == '__main__':
    rnn = RNN()
    rnn.train_model("w3-cpu")
    use_and_analyse_input_model("w3-cpu")