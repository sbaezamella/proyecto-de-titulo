# Mecanismo extractor de objectos en aprendizaje profundo por refuerzo aplicado en videojuego de Atari

Este código fuente corresponde a la propuesta para el Proyecto de Título

## Requerimientos

- Python 3
- virtualenv package

`$ pip install virtualenv`

## Instalación

### Creación del entorno virtual con `virtualenv`

`$ virtualenv venv`

### Activación del entorno virtual

`$ source venv/bin/activate`

### Instalación de paquetes en el entorno virtual, leer del `requirements.txt`

`$ pip install -r requirements.txt`

### Cargar los ROMS de Atari con `atari_py`

`$ python -m atari_py.import_roms ./Roms`

### Editar el archivo de configuración `config.ini` para ajustar el juego de Atari o parámetros para el training y testing

En `Environment`

```text
[Environment]
game = space_invaders
folder_name = space-invaders
n_objects = 6
```

`game` corresponde al nombre del juego, puede ser `space_invaders`, `ms_pacman` o `breakout`.

`folder_name` es donde estarán las imágenes del juego dentro de `./images`.

`n_objects` es la cantidad de objetos que se detectarán en la proximidad del agente.

En `Training`

```text
[Training]
learning_rate = 3e-4
gamma = 0.99
gae_lambda = 0.97
clip_range = 0.2
target_kl = 0.01
tensorboard_log = tensorboard/custom_spaceinvaders
verbose = 1
total_timesteps = 10000
saved_model_path = custom_spaceinvaders
```

`learning_rate` la tasa de aprendizaje para PPO.

`gamma` el factor de descuento del PPO.

`gae_lambda` el lambda, parámetro para reducir la varianza en el training.

`clip_range` cuánto clipping queremos.

`target_kl` divergencia KL límite.

`tensorboard_log` ruta a donde se guardará el tensorboard log.

`verbose` 1 para obtener logs.

`total_timesteps` cantidad máxima de pasos para training.

`saved_model_path` ruta del modelo entrenado a guardarse en formato `zip`.

### Training

`$ python train.py`

### Testing

`$ python test.py`
