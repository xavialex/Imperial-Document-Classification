# Imperial Document Classification

## Usage

Las grandes dependencias del proyecto son Scikit-Learn, HuggingFace y TensorFlow. Se recomienda instalarlas en un entorno virtual de Python.

## Training

Ejecutar:

```python train.py dataset```

Donde *dataset* será la ubicación del dataset proporcionado para la prueba, organizado en 7 directorios, uno por clase.

## Inference

Ejecutar:

```python classify.py model archivo1 archivo2 archivo3 ...```

Donde el primer argumento (*model*) es la ruta a la carpeta que contenga el modelo y el tokenizador y los siguientes son las rutas a los archivos a analizar. Se imprimirá por consola un listado como el que sigue:

```code
archivo1 categoría1
archivo2 categoría2
archivo3 categoría1
```