import os
import numpy as np
import librosa
import librosa.display
import pandas as pd
import matplotlib.pyplot as plt


AUDIO_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_small'
DATA_DIR = '/Users/andresmanzalini/Documents/Datasets/FMA/fma_metadata'


def get_tids_from_directory(audio_dir):
    tids = []
    for _, dirnames, files in os.walk(audio_dir):
        if dirnames == []:
            tids.extend(int(file[:-4]) for file in files if file !='.DS_Store')
    return tids


def get_audio_path(audio_dir, track_id):
    tid_str = '{:06d}'.format(track_id)
    return os.path.join(audio_dir, tid_str[:3], tid_str + '.mp3')


def filtrar_por_size(tids,min_size=350000): #corto a partir de 350 para abajo
    tids_correctos = []
    for file in tids:
        fpath = get_audio_path(AUDIO_DIR, file)
        tam = os.path.getsize(fpath)
        #print('file: ', filepath)
        #print('size: ',tam)
        if tam > min_size:
            tids_correctos.append(file)
        else:
            print('INCORRECTO ', file)
    tids_ok = np.array(tids_correctos, dtype='int64')
    print('Tracks correctos ', tids_ok.size)
    return tids_ok


### Metadata con Pandas ###

def procesarMetadata():
    filepath = DATA_DIR+'/tracks.csv'
    tracks = pd.read_csv(filepath, index_col=0, header=[0, 1])

    #selecciono columnas del dataframe 
    cols = [('set', 'split'),('set', 'subset'),('track', 'genre_top')]

    #filtro por columnas
    df_small = tracks[cols]
    df_small = df_small[df_small[('set', 'subset')] == 'small'] 

    tids = get_tids_from_directory(AUDIO_DIR)

    #filtro por los tracks que superan cierto tamano
    tracks_correctos = filtrar_por_size(tids)
    df_filtrado = df_small[df_small.index.isin(tracks_correctos)]

    df_filtrado[('track', 'genre_top')].unique()

    dict_genres = {'Electronic':1, 'Experimental':2, 'Folk':3, 'Hip-Hop':4, 
                   'Instrumental':5,'International':6, 'Pop' :7, 'Rock': 8  }
    print(dict_genres)
    df_train = df_filtrado[df_filtrado[('set', 'split')] == 'training']
    df_valid = df_filtrado[df_filtrado[('set', 'split')] == 'validation']
    df_test = df_filtrado[df_filtrado[('set', 'split')] == 'test']

    return df_train, df_valid, df_test


def create_spectrogram(track_id):
    try:
        filename = get_audio_path(AUDIO_DIR, track_id)
        #print('file ', track_id)
        y, sr = librosa.load(filename)
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        return spect.T
    except:
        print('ERROR al procesar ', track_id)
        pass


def plot_spect(track_id):
    spect = create_spectrogram(track_id)
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spect.T, y_axis='mel', fmax=8000, x_axis='time')
    plt.colorbar(format='%+2.0f dB')
    plt.show()