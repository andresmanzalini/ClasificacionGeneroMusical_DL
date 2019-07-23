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

from keras import backend as K
from keras import regularizers
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau



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
x_train = np.memmap('x_train.dat', dtype='float32', mode='r', shape=(len(df_train), LONG_SPECTO, BINS, 1))  
y_train = np.memmap('y_train.dat', dtype='float32', mode='r', shape=(len(df_train), CANT_GENEROS))  

x_valid = np.memmap('x_valid.dat', dtype='float32', mode='r', shape=(len(df_valid), LONG_SPECTO, BINS, 1))  
y_valid = np.memmap('y_valid.dat', dtype='float32', mode='r', shape=(len(df_valid), CANT_GENEROS))  

x_test = np.memmap('x_test.dat', dtype='float32', mode='r', shape=(len(df_test), LONG_SPECTO, BINS, 1))
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
    plt.savefig('parallel_CNN-RNN_accuracy.jpg')
    plt.show()

    # Summarize history for loss
    plt.plot(h.history['loss'])
    plt.plot(h.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('parallel_CNN-RNN_loss.jpg')

    plt.show()
        

def definir_Modelo():
    keras.backend.clear_session()

    i = Input(shape=(LONG_SPECTO,BINS,1))


    # Bloque Convolucional

    c1 = Conv2D(16, kernel_size=(3,1), strides=1, padding='valid', activation='relu')(i)
    p1 = MaxPooling2D((2,2), strides=(2,2))(c1)

    c2 = Conv2D(32, kernel_size=(3,1), strides=1, padding='valid', activation='relu')(p1)
    p2 = MaxPooling2D((2,2), strides=(2,2))(c2) 

    c3 = Conv2D(64, kernel_size=(3,1), strides=1, padding='valid', activation='relu')(p2)
    p3 = MaxPooling2D((2,2), strides=(2,2))(c3)

    c4 = Conv2D(128, kernel_size=(3,1), strides=1, padding='valid', activation='relu')(p3)
    p4 = MaxPooling2D((4,4), strides=(4,4))(c4) 

    c5 = Conv2D(64, kernel_size=(3,1), strides=1, padding='valid', activation='relu')(p4)
    p5 = MaxPooling2D((4,4), strides=(4,4))(c5)

    f = Flatten()(p5) # salida 256. BIEN


    # Bloque recurrente - Bidireccional GRU 

    pool_LSTM = MaxPooling2D((1,2), strides=(1,2))(i)

    squeezed = Lambda(lambda x: K.squeeze(x, axis=-1))(pool_LSTM) #embedding

    lstm = Bidirectional(GRU(64))(squeezed)  #segun paper deberia ser 128, pero se pasa...
    #salida deberia ser 256D segun paper

    #Concat output
    concat = concatenate([f,lstm], axis=-1) 

    d = Dense(128, activation='relu')(concat)#, kernel_regularizer=regularizers.l2(0.01))(concat)
    #dr = Dropout(0.3)(d) #al trabajar con frecuencias y tener poca precision, este dropout baja las canciones con proba debajo de .3

    out = Dense(CANT_GENEROS, activation='softmax')(d) 

    model = Model(outputs=out,inputs=i)

    model.summary()

    return model



def compilar_entrenar(model):
    import functools
    
    top3_acc = functools.partial(keras.metrics.top_k_categorical_accuracy, k=3)

    top3_acc.__name__ = 'top3_acc'

    model.compile(loss='categorical_crossentropy', 
                optimizer=RMSprop(lr=0.001), #probar con opt=Adam 
                metrics=['accuracy',top3_acc])#'top_k_categorical_accuracy']) #buscar mas metricas


    #checkpoint_callback = ModelCheckpoint('./weights.best.h5', monitor='val_acc', verbose=1,
    #                                        save_best_only=True, mode='max')
        
    #reducelr_callback = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, min_delta=0.01)

    #callbacks_list = [checkpoint_callback, reducelr_callback]

    opt = Adam(lr=0.001)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['accuracy',top3_acc])
        
    h = model.fit(x_train, y_train, batch_size=BATCH_SIZE, 
                epochs=40, verbose=1, validation_data=(x_valid,y_valid), 
                shuffle=True)

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
    plt.savefig('parallel_CNN-RNN_matrizConfusion.jpg')



def evaluar_modelo(model, h):
    from sklearn.metrics import classification_report

    print(model.evaluate(x=x_test, y=y_test, batch_size=BATCH_SIZE))

    dibujar_metricas(h)

    decoded = np.empty(y_test.shape[0])

    for i in range(y_test.shape[0]):
        datum = y_test[i]
        #print('index: %d' % i)
        print('encoded datum: %s' % datum)
        decoded[i] = np.argmax(datum)
        print('decoded datum: %s' % decoded)
    
    y_decoded = np.array(decoded)
    print(y_decoded)
    #print(y_true)
    y_test_pred = model.predict(x_test)
    print(y_test_pred)
    y_test_pred = np.argmax(y_test_pred, axis=1)
    print(y_test_pred)
    #labels = [0,1,2,3,4,5,6,7]
    #target_names = dict_genres.keys()

    matriz_confusion(y_test, y_test_pred)


if __name__ == "__main__":
    modelo = definir_Modelo()
    model, h = compilar_entrenar(modelo)
    #model.save('genreClasification_parallel_CNN-RNN_40epochs.h5')
    evaluar_modelo(model, h)

