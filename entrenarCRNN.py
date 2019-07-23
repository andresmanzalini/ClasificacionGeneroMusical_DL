import os
import keras
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.layers import Activation, Bidirectional, Input, Dense, concatenate, Conv1D, Conv2D, GRU, Lambda, MaxPooling1D, MaxPooling2D, LSTM, Flatten, Reshape, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop

from keras import regularizers

import multiprocessing as mp

import pandas as pd

import numpy as np

import utils 


AUDIO_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_small'
DATA_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_metadata'

LONG_SPECTO = 640
BINS = 128

BATCH_SIZE = 16

dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4, 
               'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8  }

list_gen = list(dict_genres.values())
np_gen = np.array(list_gen)
CANT_GENEROS = np_gen.shape[0]

df_train, df_valid, df_test = utils.procesarMetadata()

#cargo archivos numpy con memoria virtual en formato .dat 
x_train = np.memmap('x_train.dat', dtype='float32', mode='r', shape=(len(df_train), LONG_SPECTO, BINS))  
y_train = np.memmap('y_train.dat', dtype='float32', mode='r', shape=(len(df_train), CANT_GENEROS))  

x_valid = np.memmap('x_valid.dat', dtype='float32', mode='r', shape=(len(df_valid), LONG_SPECTO, BINS))  
y_valid = np.memmap('y_valid.dat', dtype='float32', mode='r', shape=(len(df_valid), CANT_GENEROS))  

x_test = np.memmap('x_test.dat', dtype='float32', mode='r', shape=(len(df_test), LONG_SPECTO, BINS))
y_test = np.memmap('y_test.dat', dtype='float32', mode='r', shape=(len(df_test), CANT_GENEROS))


import functools

top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

top3_acc.__name__ = 'top3_acc'


def dibujar_metricas(h):
    #def show_summary_stats(history):
    # List all data in history
    print(h.history.keys())

    # Summarize history for accuracy
    plt.plot(h.history['acc'])
    plt.plot(h.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('CRNN_accuracy.jpg')
    plt.show()

    # Summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('CRNN_loss.jpg')

    plt.show()
        

def definir_Modelo():
    keras.backend.clear_session()

    i = Input(shape=(LONG_SPECTO,BINS,))

    conv1 = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(i) 
    b1 = BatchNormalization(momentum=0.9)(conv1) #porque lo dice el paper
    p1 = MaxPooling1D(2)(b1)
    d1 = Dropout(0.1)(p1) #tambien lo dice el paper


    conv2 = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(d1) 
    b2 = BatchNormalization(momentum=0.9)(conv2)
    p2 = MaxPooling1D(2)(b2)
    d2 = Dropout(0.1)(p2)


    conv3 = Conv1D(filters=64, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.001))(d2) 
    b3 = BatchNormalization(momentum=0.9)(conv3)
    p3 = MaxPooling1D(2)(b3)
    d3 = Dropout(0.1)(p3)


    ## LSTM Layer
    lstm = LSTM(96, return_sequences=False)(d3)
    d4 = Dropout(0.1)(lstm)
        
    d = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(d4)
    d5 = Dropout(0.1)(d)
        

    out = Dense(CANT_GENEROS, activation='softmax')(d5)

    model = Model(inputs=i, outputs=out)    
    
    return model


def compilar_entrenar(model):

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy',top3_acc])
        
    print(model.summary())

    h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, 
                #steps_per_epoch=(npData_t.shape[0]//BATCH_SIZE),
                #validation_steps=(npData_v.shape[0]//BATCH_SIZE),
                epochs=40, verbose=1, validation_data=(x_valid,y_valid), 
                shuffle=True)

    print(model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE, 
                   verbose=1))

    return model, h


def matriz_confusion(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    mat = confusion_matrix(y_true, y_pred)
    sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False,
                xticklabels=dict_genres.keys(),
                yticklabels=dict_genres.keys())
    plt.xlabel('true label')
    plt.ylabel('predicted label')
    plt.savefig('CRNN_matrizConfusion.jpg')


def evaluar_modelo(model):
    from sklearn.metrics import classification_report

    dibujar_metricas(h)
    
    decoded = np.empty(y_test.shape[0])

    for i in range(y_test.shape[0]):
        datum = y_test[i]
        #print('index: %d' % i)
        print('encoded datum: %s' % datum)
        decoded[i] = np.argmax(datum)
        print('decoded datum: %s' % decoded)
        #print()
        y_decoded = np.array(decoded)
        print(y_decoded)
        #print(y_true)
        y_test_pred = model.predict(x_test)
        print(y_test_pred)
        y_test_pred = np.argmax(y_test_pred, axis=1)
        print(y_test_pred)
        labels = [0,1,2,3,4,5,6,7]
        target_names = dict_genres.keys()

    #print(y_decoded.shape, y_pred.shape)

    matriz_confusion(y_test, y_test_pred)


if __name__ == "__main__":
    modelo = definir_Modelo()
    model, h = compilar_entrenar(modelo)
    #model.save('genreClasification_CRNN_40epochs.h5')
    evaluar_modelo(model, h)
