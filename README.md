# Clasificacion de generos musicales con Deep Learning

Procesamiento y transformacion a espectrogramas de las ondas de sonido de las canciones (archivos en formato .mp3), para entrenar las distintas redes neuronales y poder asi predecir el género musical. 

Las arquitecturas deben adaptarse a la estructura de datos original del problema. 
Las canciones se representan en espectrogramas de dimensiones (LONG_SPECTO x BINS), donde LONG_SPECTO es la duración del track y BINS son arreglos que contienen las frecuencias que componen las ondas. En otras palabras, la calidad de audio.


Las arquitecturas del repositorio FMA no presentan buenos resultados... son planas y convolucionales.
La mejora está determinada por la utilizacion de distintas estructuras recurrentes RNN: Bloques de aprendizaje LSTM o GRU. 

Este espacio extra de almacenamiento tiene mucha utilidad ya que puede guardar distintos patrones cortos que presentan las canciones.


El modelo secuencial CRNN con LSTM de 64 espacios de almacenamiento mejora los resultados de FMA, pero con valores extremos. Camufla los valores de predicción finales con Dropout y Regulación, y parece entrenar muy bien pero los valores predichos distan de los reales.
 
El modelo paralelo CNN-RNN con GRU(Gated Recurrent Unit) Bidireccional de 32 bloques de aprendizaje es el que da mejores resultados porque los resultados predichos son cercanos a los de entrenamiento, con algunos resultados extremos. con una precisión por subgénero de un ~.32, y una predicción por género de un ~.68



### Preprocesamiento - Analisis de espectrogramas

```py
conda create -n envAudio python
```
```py
pip install -r requerimientos.txt
```

El procesamiento de las ondas de sonido (formato .wav) se realiza mediante la transformada de fourier a espectrogramas. En este caso, el dataset FMA tiene archivos .mp3, más comprimidos que .wav
Un espectrograma representa las frecuencias (en Hertz) que componen ondas de sonido a lo largo del tiempo.

![spectro](/imagenes/spectrograma.jpg)

Distintos parámetros como la frecuencia de muestreo (sampling rate 44100 Hertz o 22.5kHz) y la calidad del audio (quantization, arreglos de 128 o 256 frecuencias) permiten adaptar el espectrograma al modelo, según sus requisitos. El parámetro correspondiente a ventana deslizante (hop length) se utiliza para procesar de a partes el espectrograma.
Para analizar el género musical, con valores standard de sampling rate y calidad de audio podemos lograr el objetivo de clasificar las canciones según sus espectrogramas.

En este caso la frecuencia de muestreo es 44100Hertz, quantization=128 bins y hop_length=1024 (ventana de 15s, divido el track de 30s en 2 partes).



### Generador como Archivos mapeados a memoria

Hace todo el preprocesamiento de los espectrogramas a arreglos numpy de dimensión LONG_SPECTO x BINS y guarda una parte de esa informacion en un archivo .dat para efectuar un mapeo a memoria virtual.

Es más eficiente que guardarlo de forma estática en disco, ya que al estar en memoria compartida exprime al máximo el paralelismo interno de la pc y minimiza el espacio de almacenamiento. 

Esta técnica maximiza el uso de la RAM y devuelve buenos resultados desde la perspectiva *carga_computacional-tiempo_ejecucion*, a costa de una posible saturacion de memoria sin los parametros adecuados.



### CRNN - Convolutional Recurrent Neural Network 

Modelo Conv1D que implementa LSTM. Tiene buena respuesta computacional. Es una gran mejora respecto a los modelos sin celdas de almacenamiento extra.
Una desventaja es que esta muy camuflado con Dropout y Regularizacion en todas las capas. Eso hace que los valores de aprendizaje reales disten de los predichos. 	

Los resultados del modelo parecen muy buenos, con una precision por subgenero de .423, por genero de un .735 y con un error de un 1.92 

Pero podemos ver el humo en el grafico..

![CRNN_acc](/imagenes/CRNN_acc-val_acc.jpg)

![CRNN_loss](/imagenes/CRNN_loss-val_loss.jpg)


https://arxiv.org/pdf/1712.08370.pdf



### CNN-RNN - Parallel Convolutional-Recurrent Neural Netowrk

Este modelo combina la estructura de datos LSTM Bidireccional en un GRU (Gated Recurrent Unit), que almacena mas informacion que LSTM y filtra la informacion a guardar mediante sus puertas (Gates). 
Un modelo convolucional 2D en paralelo con estos espacios de almacenamiento arrojan los resultados mas consistentes a la hora de clasificar canciones segun su genero musical.

Este ultimo modelo achica el error y ofrece predicciones similares a las de entrenamiento.
loss=1.856, acc=0.3212, top3=0.6737

![CNN-RNN_acc](/imagenes/CNN-RNN_acc-val_acc.jpg)

![CNN-RNN_loss](/imagenes/CNN-RNN_loss-val_loss.jpg)


https://arxiv.org/pdf/1609.04243.pdf



Hay underfitting por la falta de data y recursos computacionales. 


El objetivo final es generar una secuencia de ondas de sonido con lo aprendido por el algoritmo.
Los LSTM aprenden y decodifican informacion pero no ‘crean’ nuevos datos...

Las arquitecturas de redes neuronales generativas (GAN’s) serian lo mas eficiente para este objetivo.
Agregando una cancion en el discriminante y otra cancion totalmente diferente por el generador podria llegar a generar algo interesante...


