import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from Evaluation import evaluation


def Model_RBiLSTM(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [50, 5]
    out, model = LSTM_Bi_train(train_data, train_target, test_data, test_target, sol)
    pred = np.asarray(out)

    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0

    Eval = evaluation(pred, test_target)
    return Eval, pred


def LSTM_Bi_train(train_data, train_target, test_data, test_target, batch, sol=None):
    if sol is None:
        sol = [0, 0, 4, 50]
    Optimizers = ['adam', 'SGD', 'RMSProp', 'AdaDelta', 'Adagrad']
    Act = ['linear', 'sigmoid', 'relu', 'tanh']
    model = Sequential()
    model.add(Embedding(128, input_length=(1, test_data.shape[1])))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(1))
    model.add(Dense(1, activation=int(Act[sol[0]])))
    model.compile(loss='binary_crossentropy', optimizer=int(Optimizers[sol[1]]), metrics=['accuracy'])
    model.fit(train_data, train_target,
              batch_size=batch,
              epochs=int(sol[3]),
              validation_data=[test_data, test_target])
    testPredict = model.predict(test_data)

    return testPredict, model
