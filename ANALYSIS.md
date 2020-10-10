# Analysis

En este archivo se detallan los pasos lógicos y de implementación realizados para la consecución del proyecto. 

## Objeto

Dado un dataset de documentos pertenecientes a 7 categorías (*exploration*, *headhunters*, *intelligence*, *logistics*, *politics*, *transportation* y *weapons*) se pretende crear un modelo de clasificación que, para un nuevo documento, determine a qué categoría pertenece. 

## Solución

Se propone la creación de un modelo de clasificación basado en Transformers. Los Transformers ([Vaswani et al., 2017, Attention is all you need](https://arxiv.org/abs/1706.03762)) son una arquitectura basada en modelos de atención que han alcanzado el estado del arte en la práctica totalidad de las tareas asociadas al Procesamiento del Lenguaje Natural (NLP). Una de las librerías en auge que ofrece Transformers es [Hugging Face](https://huggingface.co/), la cual ofrece multitud de modelos, pipelines, etc. Para este proyecto se hace uso de un [DistilBERT for Sequence Classification](https://huggingface.co/transformers/model_doc/distilbert.html#distilbertforsequenceclassification), modelo que puede ser usado para clasificar secuencias de texto en distintas categorías.

El DistilBERT que se descarga en primera instancia es un modelo de lenguaje preentrenado. Estos modelos han sido entrenados en grandes corpus, ofreciendo una *traducción matemática* (*word embeddings*) del lenguaje natural capaz de capturar el significado semántico subyacente a las palabras, dando esto una mayor utilidad que vectorizaciones anteriores (e.g. one-hot-encoding). 

Cada modelo cuenta con un tokenizador, que transforma una secuencia de texto de entrada para ser procesada por el modelo. Tras generar un conjunto de datos a partir de las categorías dadas, se realiza un fine-tuning del modelo preentrenado (técnica conocida como [transfer learning](https://en.wikipedia.org/wiki/Transfer_learning)) para que se adapte al nuevo conjunto de datos. Se divide el dataset en conjuntos de entrenamiento y validación para que la herramienta [Trainer](https://huggingface.co/transformers/main_classes/trainer.html) vaya ajustando el aprendizaje del modelo durante el entrenamiento. Se pueden observar distintas métricas desde Tensorboard mediante:

```tensorboard --logdir=./logs```  

Como se puede observar, la función de coste baja significativamente tras unas pocas épocas (<10)
![loss](reports/loss.png)

Tras el entrenamiento, se guarda el modelo en cualquiera de los dos backends disponibles (PyTorch y/o TensorFlow) y el tokenizador para que puedan ser [compartidos](https://huggingface.co/transformers/model_sharing.html).

## Próximos pasos

Con más tiempo, se pueden realizar múltiples acciones para mejorar el desempeño del modelo. Se debe hacer un estudio más profundo de las categorías del dataset. Una fase de análisis de datos puede dar información valiosa a la hora de abordar el problema. Se puede efectuar una limpieza de los datos: la lematización y la eliminación de *stopwords* han sido estudiadas durante mucho tiempo y pueden dar un mayor rendimiento al modelo final. También se puede considerar un análisis más detallado de los hiperparámetros empleados para comparar distintos modelos.  

Además, se puede desarrollar un script de evaluación en el que, sobre una partición de datos desconocida por el modelo (*test set*) se evalúe su comportamiento, analizando métricas como *Accuracy*, *Precision*, *Recall* o *F1-Score*, *Matriz de confusión*, etc. para determinar la correcta o no generalización del modelo. En la medida de lo posible, se buscará que el conjunto de datos de test se parezca al máximo posible a la distribución de datos del entorno real sobre el que se explotará el modelo. A largo término, y comparando la actuación del modelo en los sets de entrenamiento, validación y test, se identificarán problemas de sesgo, varianza (*overfitting* y *underfitting*) para actuar en consecuencia. Si el problema lo requiere, se puede reentrenar periódicamente el modelo con la llegada de nuevos datos.