#aca hago un script que mapee los archivos a memoria

import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import keras

import utils 


AUDIO_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_small'
DATA_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_metadata'

LONG_SPECTO = 640 
BINS = 128 

dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4, 
               'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8  }

list_gen = list(dict_genres.values())
np_gen = np.array(list_gen)
CANT_GENEROS = np_gen.shape[0]

df_train, df_valid, df_test = utils.procesarMetadata()

### Generador MMF ###

def generar_MMF(df_train, df_valid, df_test):
    #Creo el arreglo numpy en un archivo mapeado a memoria
    npData_x_train = np.memmap('x_train.dat', dtype='float32', mode='w+', shape=(len(df_train), LONG_SPECTO, BINS))
    npData_y_train = np.memmap('y_train.dat', dtype='float32', mode='w+', shape=(len(df_train), CANT_GENEROS))

    npData_x_valid = np.memmap('x_valid.dat', dtype='float32', mode='w+', shape=(len(df_valid), LONG_SPECTO, BINS))
    npData_y_valid = np.memmap('y_valid.dat', dtype='float32', mode='w+', shape=(len(df_valid), CANT_GENEROS))

    npData_x_test = np.memmap('x_test.dat', dtype='float32', mode='w+', shape=(len(df_test), LONG_SPECTO, BINS))
    npData_y_test = np.memmap('y_test.dat', dtype='float32', mode='w+', shape=(len(df_test), CANT_GENEROS))


    #Proceso los datos
    for i, track_id in tqdm(enumerate(df_train.index), total=len(df_train)): 
        npData_x_train[i, :, :] = utils.create_spectrogram(track_id)[:LONG_SPECTO, :]
        genero = df_train[('track','genre_top')].loc[track_id]
        int_genero = dict_genres.get(genero)
        y_categ = keras.utils.np_utils.to_categorical(int_genero-1, CANT_GENEROS) #con -1 normalizo de 0 a 7
        npData_y_train[i, ] = y_categ    
        
    for i, track_id in tqdm(enumerate(df_valid.index), total=len(df_valid)): 
        npData_x_valid[i, :, :] = utils.create_spectrogram(track_id)[:LONG_SPECTO, :]
        genero = df_valid[('track','genre_top')].loc[track_id]
        int_genero = dict_genres.get(genero)
        y_categ = keras.utils.np_utils.to_categorical(int_genero-1, CANT_GENEROS)
        npData_y_valid[i, ] = y_categ
        
    for i, track_id in tqdm(enumerate(df_test.index), total=len(df_test)): 
        npData_x_test[i, :, :] = utils.create_spectrogram(track_id)[:LONG_SPECTO, :]
        genero = df_test[('track','genre_top')].loc[track_id]
        int_genero = dict_genres.get(genero)
        y_categ = keras.utils.np_utils.to_categorical(int_genero-1, CANT_GENEROS)
        npData_y_test[i, ] = y_categ
        

    del npData_x_train
    del npData_y_train
    del npData_x_valid
    del npData_y_valid
    del npData_x_test
    del npData_y_test



### MAIN ###

if __name__ == "__main__":
    tids = utils.get_tids_from_directory(AUDIO_DIR)
    print('Cantidad total de tracks: ',len(tids))
    #corto tracks menores al tamano minimo
    tracks_correctos = utils.filtrar_por_size(tids)
    #selecciono/corto segun tipo de entrenamiento con pandas 
    df_train, df_valid, df_test = utils.procesarMetadata()
    generar_MMF(df_train, df_valid, df_test)

