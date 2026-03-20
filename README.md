# Detección de Frambuesas por Visión por Computadora

Detecta y clasifica frambuesas **maduras** e **inmaduras** en una imagen usando OpenCV y matplotlib. No requiere modelo de deep learning.

## Presentacion del proyecto

[![Canva](https://img.shields.io/badge/Ver%20en-Canva-blueviolet?logo=canva)](https://www.canva.com/design/DAHEaX5SDq8/kJl_glwjw6bSAkRAkvZQ4A/edit?utm_content=DAHEaX5SDq8&utm_campaign=designshare&utm_medium=link2&utm_source=sharebutton)

## Resultados

| Archivo | Contenido |
|---|---|
| `resultado.png` | Imagen original con detecciones anotadas |
| `pipeline_compartido.png` | Pasos de preprocesamiento (Laplaciano, nitidez) |
| `pipeline_maduras.png` | Pipeline de filtrado para berries maduras |
| `pipeline_inmaduras.png` | Pipeline de filtrado para berries inmaduras |

## Pipeline

```
Imagen original
    └── Escala de grises
        └── Laplaciano → |abs| → normalizar → Máscara de nitidez (bokeh filter)
                ├── MADURAS   : máscara HSV rojo/carmesí → AND nitidez → Opening → Closing → HoughCircles
                └── INMADURAS : máscara HSV verde/salmón → AND nitidez → Opening → Closing → Contornos + Hull
```

## Requisitos

```
opencv-python
numpy
matplotlib
```

```bash
pip install opencv-python numpy matplotlib
```

## Uso

Coloca tu imagen como `frambuesas.jpg` en el mismo directorio y ejecuta:

```bash
python frambuesas_detector.py
```

La consola imprime las coordenadas y radio de cada berry detectada:

```
Madura   1: x=312 y=198 r=67
Inmadura 1: x=120 y=88  r=54
```

## Parámetros ajustables

| Variable | Descripción |
|---|---|
| `ESCALA` | Factor de reducción de imagen (1 = original, 0.5 = mitad) |
| `LOW_MADURA_*` / `HIGH_MADURA_*` | Rangos HSV para berries maduras |
| `LOW_INMADURA_*` / `HIGH_INMADURA_*` | Rangos HSV para berries inmaduras |
| `minRadius` / `maxRadius` en HoughCircles | Tamaño esperado de las berries en píxeles |
| `fill_ratio >= 0.55` | Umbral de validación para círculos de Hough |
| `hull_circ < 0.78` | Umbral de circularidad para descartar hojas |
