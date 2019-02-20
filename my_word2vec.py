

import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping


def word2vec(vocab_size, embeded_dim, max_len):
    model = Sequential()
    model.add(Embedding(vocab_size, embeded_dim, input_length=max_len, embeddings_initializer=uniform(seed=20190219)))
    model.add(Flatten())
    model.add(Dense(units=vocab_size, use_bias=True, kernel_initializer=glorot_uniform(seed=20190219)))
    model.add(Activation("softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="RMSprop", metrics=["categorical_accuracy"])
    return model

def training4model(model, x_train, y_train, batch_size, epoch_num, pretrained_params=None):
    callback = EarlyStopping(monitor='categorical_accuracy', patience=1, verbose=1)
    # print(y_train.shape)
    if pretrained_params != None:
        model.load_weights(pretrained_params)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch_num, verbose=1, shuffle=True, callbacks=[callback], validation_split=0.0)
    return trained_model
