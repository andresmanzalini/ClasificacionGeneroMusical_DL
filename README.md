# Clasificacion de Género Musical con Deep Learning

Este proyecto consiste en el procesamiento y transformacion de canciones en formato .mp3 a espectrogramas (la representacion del sonido en frecuencias), lo que permite entrenar distintas redes neuronales y predecir el género musical de la canciones en formato .mp3 


Para entrenar se utiliza el repositorio medium de FMA. de 24G de canciones, con un total de 12000 canciones en formato .mp3


Las arquitecturas de las redes neuronales deben adaptarse a la estructura de datos original del problema: el espectrograma. 
Los fragmentos de canciones se representan en espectrogramas de dimensiones (LONG_SPECTO x BINS), donde:
LONG_SPECTO es la duración del track;
BINS son arreglos que contienen las frecuencias que componen las ondas. Es decir, por cada unidad de tiempo, contiene la informacion en frecuencias (Hertz). 
La dimension de BINS representa la calidad de audio.


El repo FMA contiene un baseline para usar como punto de partida.
Las arquitecturas del repositorio FMA no presentan buenos resultados. La mejora está determinada por la utilizacion de distintas estructuras recurrentes RNN: Bloques de aprendizaje LSTM o GRU. 
Este espacio extra de almacenamiento tiene mucha utilidad ya que puede guardar distintos patrones cortos que presentan las canciones.



En este proyecto se analizan 2 tipos de arqyutecturas: Una Arquitectura Secuencial y una Arquitectura Paralela.

El modelo secuencial CRNN con LSTM de 64 espacios de almacenamiento mejora los resultados de FMA. Camufla los valores de predicción finales con Dropout y Regulación, y parece entrenar muy bien pero los valores predichos distan de los reales.
 
El modelo paralelo CNN-RNN con GRU(Gated Recurrent Unit) Bidireccional de 32 bloques de aprendizaje es el que da mejores resultados. Los resultados de prediccion son cercanos a los de entrenamiento, con algunos resultados extremos. 


Hasta el momento, los mejores resultados los presenta el modelo secuencial CRNN , con precision de 0.82 por genero


<br />


### Preprocesamiento - Análisis de espectrogramas

```py
conda create -n envAudio python
```
```py
pip install -r requerimientos.txt
```


Un espectrograma representa las frecuencias (en Hertz) que componen ondas de sonido a lo largo del tiempo.

![spectro](/imagenes/spectrograma.jpg)

El procesamiento de las ondas de sonido (formato .wav) se realiza mediante la transformada de fourier a espectrogramas. 
En este caso, el dataset FMA tiene archivos .mp3, más comprimidos que .wav


Distintos parámetros como la frecuencia de muestreo (sampling rate 44100 Hertz o 22.5kHz) y la calidad del audio (quantization, arreglos de 128 o 256 frecuencias) permiten adaptar el espectrograma al modelo, según sus requisitos. 
El parámetro correspondiente a ventana deslizante (hop length) se utiliza para procesar de a partes el espectrograma.


Para clasificar los espectrogramas segun su género musical se usaran valores standard de sampling rate y calidad de audio.

Los parametros importantes para el procesamiento de la data son frecuencia de muestreo de 44100Hertz, quantization de 128 bins y hop_length=1024 (ventana de 15s, divido el track de 30s en 2 partes).


<br />


### Generador como Archivos mapeados a memoria

El procesamiento de grandes cantidades de datos necesita un mecanismo especial.
Este mecanismo, conocido como generador, crea batches de conjuntos de datos para entrenar el modelo. Entrena el moedlo procesa todo el dataset de a batches.

En este caso es necesario procesar 24G de informacion con una RAM de 4G

Para eso se decidio usar un generador como Archivo Mapeado a Memoria.

Este tipo de generador hace todo el preprocesamiento de los espectrogramas a arreglos numpy de dimensión LONG_SPECTO x BINS, y guarda una parte de esa informacion en un archivo .dat para efectuar un mapeo a memoria virtual.


Es más eficiente que guardarlo de forma estática en disco, ya que al estar en memoria compartida exprime al máximo el paralelismo interno de la pc y minimiza el espacio de almacenamiento. 

Esta técnica maximiza el uso de la RAM y devuelve buenos resultados desde la perspectiva *carga_computacional-tiempo_ejecucion*, a costa de una posible saturacion de memoria sin los parametros adecuados.


<br />


## Modelos

### CRNN - Convolutional Recurrent Neural Network 

Modelo secuencial. 
Implementa LSTM. 
Tiene buena respuesta computacional. 
Es una gran mejora respecto a los modelos sin celdas de almacenamiento extra.

![arqui_secuencial](/imagenes/arquitectura_secuencial.jpg)


Los resultados del modelo son aceptables, con precision por genero de un .82 pero con un error alto de 0.6


![CRNN_acc](/imagenes/CRNN_acc-val_acc.jpg)

![CRNN_loss](/imagenes/CRNN_loss-val_loss.jpg)


La matriz de confusion

![CRNN_matconf](/imagenes/CRNN_matconfusion.jpg)



##### Referencia

https://arxiv.org/pdf/1712.08370.pdf


<br />
<br />


### CNN-RNN - Parallel Convolutional-Recurrent Neural Netowrk

Modelo paralelo.
Combina la estructura de datos LSTM Bidireccional en un GRU (Gated Recurrent Unit), que almacena mas informacion que LSTM y filtra la informacion a guardar mediante sus puertas (Gates). 
Utiliza un modelo convolucional 2D en paralelo con estos espacios de almacenamiento (GRU) 

![arqui_paralela](/imagenes/arquitectura_paralela.jpg)


Arroja resultados mas consistentes a la hora de clasificar canciones segun su genero musical.

La precision es buena e incremental, de un .82
El problema es que se estanca el error, en un .6

![CNN-RNN_acc](/imagenes/CNN-RNN_acc-val_acc.jpg)

![CNN-RNN_loss](/imagenes/CNN-RNN_loss-val_loss.jpg)


##### Referencia

https://arxiv.org/pdf/1609.04243.pdf


<br />
<br />


### Conclusiones

Con el dataset medium mejoran los datos, pero sigue habiendo underfiting por la falta de data
Para tanta informacion sobre frecuencias por aprender, una cantidad de 10000 canciones divididas en 8 subgeneros aumenta la probabilidad de error.


El objetivo final es generar una secuencia de ondas de sonido con lo aprendido por el algoritmo.
Es decir, crear un fragmento de musica con lo aprendido por el algoritmo.


Los LSTM aprenden y decodifican informacion pero no ‘crean’ nuevos datos...
Las arquitecturas de redes neuronales generativas (GAN’s) serian lo mas eficiente para este objetivo.
Agregando una cancion en el discriminante y otra cancion totalmente diferente por el generador podria llegar a generar algo interesante...


