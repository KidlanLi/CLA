# from keras.preprocessing.sequence import pad_sequences for older versions of tf and keras
from keras.utils import pad_sequences # for the latest versions of tf and keras
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, Bidirectional, Conv1D
import sys
import numpy as np

from hw08_neural_networks import get_data

VOCAB_SIZE = 10000
MAX_LEN = 100
BATCH_SIZE = 32
EPOCHS = 10

def build_and_evaluate_model(x_train, y_train, x_dev, y_dev):
    '''Builds, trains and evaluates a keras LSTM model.'''
    # TODO: REMOVE for exercise 3.
    x_train = pad_sequences(x_train, maxlen=MAX_LEN, value=0.0)  # TODO: Exercise 3.1
    x_dev = pad_sequences(x_dev, maxlen=MAX_LEN, value=0.0)  # TODO: Exercise 3.1
    """y_train = np.array(y_train)  # TODO: Exercise 3.1
    y_dev = np.array(y_dev)  # TODO: Exercise 3.1
    model = Sequential()"""

    y_train = np.asarray(y_train)
    y_dev = np.asarray(y_dev)
    model = Sequential(layers=[Embedding(VOCAB_SIZE, 50), Bidirectional(LSTM(25)), Dense(1, activation='sigmoid')])
    # TODO: Ex. 3.2 - 3.4
    # model.add(Embedding(input_dim=VOCAB_SIZE, output_dim=50))
    # model.add(Bidirectional(LSTM(25)))
    # model.add(Dense(1, activation='sigmoid'))
    # Add layers.
    # model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Compile model.
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_dev, y_dev))
    # Fit to data.
    # ODOT
    score, acc = model.evaluate(x_dev, y_dev)
    return score, acc, model


def main(argv):
    print('Loading data...')
    x_train, y_train, x_dev, y_dev, word2id = get_data.nltk_data(vocab_size=VOCAB_SIZE)
    print(len(x_train), 'training samples')
    print(len(x_dev), 'development samples')
    score, acc, _ = build_and_evaluate_model(x_train, y_train, x_dev, y_dev)
    print('\ndev score:', score)
    print('dev accuracy:', acc)

if __name__ == "__main__":
    main(sys.argv[1:])
