# Clasificacion de generos musicales con Deep Learning

Procesamientovy transformacion a espectrogramas de las ondas de sonido de las canciones (archivos en formato .mp3), para entrenar las distintas redes neuronales y poder asi predecir el género musical. 

Las arquitecturas deben adaptarse a la estructura de datos original del problema. Las canciones se representan en espectrogramas de dimensiones (LONG_SPECTO x BINS), donde LONG_SPECTO es la duración del track y BINS son arreglos que contienen las frecuencias que componen las ondas por cada LONG_SPECTO. En otras palabras, la calidad de audio.
Por eso analizarlo en una estructura con mas dimensiones es mejor que hacerlo en arreglos planos.

Las arquitecturas del repositorio FMA no presentan buenos resultados... son planas y convolucionales
La mejora está determinada por la aplicaion de distintas estructuras recurrentes RNN: Bloques de aprendizaje LSTM o GRU. 
Este espacio extra de almacenamiento tiene mucha utilidad ya que puede guardar distintos patrones cortos que presentan las canciones.

El modelo secuencial CRNN con LSTM de X espacios de almacenamiento mejora los resultados de FMA, pero con muchos valores extremos. Camufla los valores de predicción finales con Dropout y Regularización, y parece entrenar muy bien pero los valores predichos distan de los reales.
 
El modelo paralelo CNN-RNN con GRU(Gated Recurrent Unit) Bidireccional de W bloques de aprendizaje es el que da mejores resultados, con una precisión por subgénero de un ~.33, y una predicción por género de un ~.70


La música al ser intrínsecamente matemática, genera patrones de distintos tipos. Relacionado a la armonía podemos encontrarlo en las escalas; en el ritmo se presenta en los compases; la melodía en algunos casos se sale un poco de la estructura, pero en las canciones ‘estructuradas’ es muy probable encontrar un patrón.



### Preprocesamiento - Analisis de espectrogramas

conda create -n envAudio python

pip install -r requerimientos.txt


El procesamiento de las ondas de sonido (formato .wav) se realiza mediante la transformada de fourier a espectrogramas. En este caso, el dataset FMA tiene archivos .mp3, más comprimidos que .wav
Un espectrograma representa las frecuencias (en Hertz) que componen ondas de sonido a lo largo del tiempo.
Distintos parámetros como la frecuencia de muestreo (sampling rate 44100 Hertz o 22.5kHz) y la calidad del audio (quantization, arreglos de 128 o 256 frecuencias) permiten adaptar el espectrograma al modelo, según sus requisitos. El parámetro correspondiente a ventana deslizante (hop length) se utiliza para procesar de a partes el espectrograma.
Para analizar el género musical, con valores standard de sampling rate y calidad de audio podemos lograr el objetivo de clasificar las canciones según sus espectrogramas.
En este caso la frecuencia de muestreo es 44100Hertz, quantization=128 bins y hop_length=1024 (ventana de 15s, divido el track de 30s en 2 partes).



### Generador como Archivos mapeados a memoria

Hace todo el preprocesamiento de los espectrogramas a arreglos numpy de dimensión LONG_SPECTO x BINS y guarda la primera parte de esa informacion en un archivo .dat para efectuar un mapeo de memoria.

Es más eficiente que guardarlo de forma estática en disco, ya que al estar en memoria compartida exprime al máximo el paralelismo interno de la pc y minimiza el espacio de almacenamiento. 

Esta técnica maximiza el uso de la RAM y devuelve buenos resultados desde la perspectiva *carga_computacional-tiempo_ejecucion*, a costa de una posible saturacion de memoria sin los parametros adecuados.



### CRNN - Convolutional Recurrent Neural Network 

Modelo Conv1D que implementa LSTM. Tiene mejor respuesta computacional que el paralelo. Es una gran mejora respecto a los modelos sin celdas de almacenamiento extra.
Este modelo arroja resultados aceptables, de un
Una desventaja es que esta muy camuflado con Dropout y Regularizacion en todas las capas. Eso hace que los valores de aprendizaje reales disten de los predichos. 	

resultados del primer modelo

imagenes y matriz confusion

https://arxiv.org/pdf/1712.08370.pdf



### CNN-RNN - Parallel Convolutional-Recurrent Neural Netowrk

Este modelo combina la estructura de datos LSTM Bidireccional en un GRU (Gated Recurrent Unit), que almacena mas informacion que LSTM y filtra la informacion a guardar mediante sus puertas (Gates). 
La paralelizacion de un modelo convolucional 2D y en paralelo estos espacios de almacenamiento arrojan los mejores resultados a la hora de clasificar canciones segun su genero musical.
Estos ultimos modelos minimizan el error y ofrece predicciones con 50% precision

fotos de resultados y matriz confusion

https://arxiv.org/pdf/1609.04243.pdf



Como en todo, hay underfitting porque no tengo suficiente data ni recursos computacionales. 


El objetivo final es generar una secuencia de ondas de sonido con lo aprendido por el algoritmo.
Los LSTM aprenden y decodifican informacion pero no ‘crean’ nuevos datos.
Las arquitecturas de redes neuronales generativas (GAN’s) serian lo mas eficiente para este objetivo.
Agregando una cancion en el discriminante y otra cancion totalmente diferente por el generador podria llegar a generar algo interesante.


