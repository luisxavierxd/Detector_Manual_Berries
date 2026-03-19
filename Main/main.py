# Librerias
import matplotlib
matplotlib.use('Agg')
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# LECTURA Y CONVERSION DE IMAGEN
# ─────────────────────────────────────────────────────────────────────────────

# Leer imagen original (OpenCV la carga en BGR por defecto)
img = cv2.imread("frambuesas.jpg")

# Convertir a RGB para visualizar correctamente con matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# ─────────────────────────────────────────────────────────────────────────────
# ESCALADO DE IMAGEN
# HoughCircles es muy lento con radios grandes en imagenes de alta resolucion
# porque su acumulador interno crece de forma no lineal con el radio maximo.
# La solucion es reducir la imagen antes de procesar para que los radios
# originales (535-103 px) sigan siendo validos, y escalar las coordenadas
# de vuelta al final para dibujar sobre la imagen original.
# ─────────────────────────────────────────────────────────────────────────────

# Factor de escala: ajustar segun resolucion de la imagen
# 0.5 = mitad del tamaño original
ESCALA = 1

h, w = img.shape[:2]

# Crear versiones escaladas de todas las imagenes necesarias
img_scaled      = cv2.resize(img,     (int(w * ESCALA), int(h * ESCALA)))
img_rgb_scaled  = cv2.resize(img_rgb, (int(w * ESCALA), int(h * ESCALA)))

# Convertir a HSV y gris sobre la imagen escalada
img_hsv_scaled  = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2HSV)
img_gray_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

# ─────────────────────────────────────────────────────────────────────────────
# MASCARA DE NITIDEZ (SHARPNESS MASK)
# Sirve para eliminar el fondo desenfocado (efecto bokeh)
# Las berries estan enfocadas, el fondo no
# El Laplaciano mide el gradiente de la imagen: zonas enfocadas
# tienen gradiente alto, el fondo bokeh tiene gradiente casi cero
# ─────────────────────────────────────────────────────────────────────────────

# Calcular el Laplaciano de la imagen gris escalada
# cv2.CV_64F permite valores negativos (se usa valor absoluto despues)
laplaciano = cv2.Laplacian(img_gray_scaled, cv2.CV_64F)

# Valor absoluto para quedarnos solo con la magnitud del gradiente
sharpness_map = np.abs(laplaciano).astype(np.float32)

# Suavizar el mapa de nitidez para que las zonas enfocadas
# formen regiones continuas en lugar de puntos dispersos
sharpness_blur = cv2.GaussianBlur(sharpness_map, (41, 41), 0)

# Normalizar a rango 0-255 para poder umbralizar
sharpness_norm = cv2.normalize(sharpness_blur, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Umbralizar: pixeles con nitidez > 12 = zona enfocada (blanco)
#             pixeles con nitidez <= 12 = fondo bokeh (negro)
_, mascara_nitidez = cv2.threshold(sharpness_norm, 12, 255, cv2.THRESH_BINARY)

# ─────────────────────────────────────────────────────────────────────────────
# RANGOS DE COLOR HSV
# En OpenCV: H=[0,180], S=[0,255], V=[0,255]
# El ROJO es especial: esta partido en dos rangos (H=0-14 y H=155-180)
# porque en el circulo de color el rojo esta en los extremos (0° y 360°)
# ─────────────────────────────────────────────────────────────────────────────

## Maduras (rojo/carmesi)
# Rango A: rojo puro (H bajo, saturacion alta)
low_madura_a  = np.array([0,   150, 20])
high_madura_a = np.array([14,  255, 255])

# Rango B: carmesi/magenta (H alto, wraparound del rojo)
low_madura_b  = np.array([155, 60,  20])
high_madura_b = np.array([180, 255, 255])

# Rango C: naranja oscuro muy saturado (berries maduras en sombra)
# V bajo (<150) para no capturar naranjas brillantes que son inmaduras
low_madura_c  = np.array([14,  220, 20])
high_madura_c = np.array([28,  255, 150])

## Inmaduras
# Verde-amarillo de berry: H=20-32
# IMPORTANTE: las hojas tienen H=33-39, quedan fuera de este rango
# Esa diferencia de ~5 unidades en H es lo que separa berry de hoja
low_inmadura_verde  = np.array([20, 90,  70])
high_inmadura_verde = np.array([32, 255, 255])

# Salmon/naranja brillante: berries semi-maduras (H=7-22, V alto)
low_inmadura_salmon  = np.array([7,  100, 140])
high_inmadura_salmon = np.array([22, 255, 255])

# ─────────────────────────────────────────────────────────────────────────────
# MASCARAS DE COLOR
# cv2.inRange devuelve 255 donde el pixel esta dentro del rango, 0 fuera
# Todas las mascaras se calculan sobre la imagen escalada
# ─────────────────────────────────────────────────────────────────────────────

## Mascaras individuales de maduras
mascara_madura_a = cv2.inRange(img_hsv_scaled, low_madura_a, high_madura_a)
mascara_madura_b = cv2.inRange(img_hsv_scaled, low_madura_b, high_madura_b)
mascara_madura_c = cv2.inRange(img_hsv_scaled, low_madura_c, high_madura_c)

# Combinar los tres rangos con OR (cualquiera de los tres es madura)
mascara_madura_color = cv2.bitwise_or(mascara_madura_a, mascara_madura_b)
mascara_madura_color = cv2.bitwise_or(mascara_madura_color, mascara_madura_c)

# Aplicar mascara de nitidez: solo conservar pixeles enfocados
mascara_madura = cv2.bitwise_and(mascara_madura_color, mascara_nitidez)

## Mascaras individuales de inmaduras
mascara_inmadura_verde  = cv2.inRange(img_hsv_scaled, low_inmadura_verde,  high_inmadura_verde)
mascara_inmadura_salmon = cv2.inRange(img_hsv_scaled, low_inmadura_salmon, high_inmadura_salmon)

# Combinar verde y salmon con OR
mascara_inmadura_color = cv2.bitwise_or(mascara_inmadura_verde, mascara_inmadura_salmon)

# Aplicar mascara de nitidez
mascara_inmadura_color = cv2.bitwise_and(mascara_inmadura_color, mascara_nitidez)

# Quitar de la mascara inmadura cualquier pixel que ya este en madura
# (evita que zonas rojizas aparezcan en ambas mascaras)
mascara_inmadura = cv2.bitwise_and(mascara_inmadura_color, cv2.bitwise_not(mascara_madura))

# ─────────────────────────────────────────────────────────────────────────────
# MORFOLOGIA
# Limpia el ruido y suaviza las mascaras antes de detectar
# Opening  = erosion + dilatacion: elimina puntos aislados de ruido
# Closing  = dilatacion + erosion: rellena huecos dentro de las berries
# ─────────────────────────────────────────────────────────────────────────────

kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)
kernel_3 = np.ones((3, 3), np.uint8)

# Morfologia para mascaras de maduras
mascara_madura_limpia = cv2.morphologyEx(mascara_madura, cv2.MORPH_OPEN,  kernel_5, iterations=1)
mascara_madura_limpia = cv2.morphologyEx(mascara_madura_limpia, cv2.MORPH_CLOSE, kernel_3, iterations=2)

# Morfologia para mascaras de inmaduras
# Se usa kernel_7 en closing porque los drupelets dejan mas huecos
mascara_inmadura_limpia = cv2.morphologyEx(mascara_inmadura, cv2.MORPH_OPEN,  kernel_5, iterations=1)
mascara_inmadura_limpia = cv2.morphologyEx(mascara_inmadura_limpia, cv2.MORPH_CLOSE, kernel_7, iterations=2)

# ─────────────────────────────────────────────────────────────────────────────
# DETECCION DE BERRIES MADURAS — HoughCircles sobre imagen gris escalada
#
# Se corre HoughCircles sobre la imagen gris escalada (no sobre la mascara)
# porque los drupelets de la frambuesa crean huecos en la mascara que
# rompen la deteccion por contornos. El gradiente circular existe en la
# imagen gris aunque la mascara tenga huecos.
#
# Luego se valida cada circulo contra la mascara de color usando
# fill_ratio = pixeles de mascara dentro del circulo / area del circulo
# Si el circulo no cubre suficiente color de berry, se descarta
# ─────────────────────────────────────────────────────────────────────────────

# Suavizar imagen gris escalada para reducir falsos bordes de drupelets
img_gray_blur = cv2.GaussianBlur(img_gray_scaled, (5, 5), 0)

# HoughCircles: busca circulos por acumulacion de gradientes
# dp=1.3       : resolucion del acumulador (1=igual a imagen, >1=mas rapido)
# minDist=90   : distancia minima entre centros de circulos detectados
# param1=65    : umbral superior del detector de bordes Canny interno
# param2=32    : umbral del acumulador (mas alto = menos circulos, mas precisos)
# minRadius=35 : radio minimo en pixeles (descarta drupelets sueltos)
# maxRadius=103: radio maximo en pixeles (descarta clusters enteros)
circulos = cv2.HoughCircles(img_gray_blur, cv2.HOUGH_GRADIENT,
                             dp=1.3, minDist=90,
                             param1=65, param2=32,
                             minRadius=35, maxRadius=103)

# Lista donde se guardan las detecciones validas
berries_maduras = []

if circulos is not None:
    for (cx, cy, r) in circulos[0]:
        cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))

        # Crear mascara circular para este candidato
        mascara_circulo = np.zeros_like(mascara_madura_limpia)
        cv2.circle(mascara_circulo, (cx, cy), r, 255, -1)

        # Contar pixeles de color madura dentro del circulo
        solapamiento = cv2.bitwise_and(mascara_madura_limpia, mascara_circulo)
        area_circulo = np.pi * r * r
        fill_ratio   = np.sum(solapamiento > 0) / area_circulo

        # Aceptar solo si al menos 42% del circulo es color de berry madura
        if fill_ratio >= 0.55:
            berries_maduras.append({"centro": (cx, cy), "radio": r, "fill": round(fill_ratio, 2)})

# ─────────────────────────────────────────────────────────────────────────────
# DETECCION DE BERRIES INMADURAS — Contornos + Hull Circularity
#
# Para inmaduras se usan contornos en lugar de Hough porque los clusters
# de berries verdes forman blobs mas compactos (sin tantos huecos).
#
# Filtro de forma: hull_circularity del convex hull
#   hull_circ = 4*pi*Area_hull / Perimetro_hull^2
#   Valor en (0,1]: 1 = circulo perfecto
#
# Se usa el hull (envolvente convexa) en lugar del contorno directo
# porque los drupelets hacen el contorno muy irregular (circ baja),
# pero el hull del conjunto es casi circular (circ alta)
#
# Hojas: hull_circ < 0.78 (formas alargadas o irregulares)
# Berries: hull_circ >= 0.78
# ─────────────────────────────────────────────────────────────────────────────

contornos, _ = cv2.findContours(mascara_inmadura_limpia, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

berries_inmaduras = []

for contorno in contornos:
    area = cv2.contourArea(contorno)

    # Descartar blobs demasiado pequenos (ruido, fragmentos de hoja)
    if area < 2000:
        continue

    # Calcular convex hull del contorno
    hull       = cv2.convexHull(contorno)
    area_hull  = cv2.contourArea(hull)
    perim_hull = cv2.arcLength(hull, True)

    # Calcular circularidad del hull (no del contorno jagged)
    if perim_hull == 0:
        continue
    hull_circ = (4 * np.pi * area_hull) / (perim_hull ** 2)

    # Descartar si la forma no es suficientemente circular
    # 0.78 separa berries (0.82-0.92) de hojas y fragmentos (<0.75)
    if hull_circ < 0.78:
        continue

    # Circulo minimo que encierra el contorno (para dibujar)
    (cx, cy), r = cv2.minEnclosingCircle(contorno)

    berries_inmaduras.append({
        "centro":    (int(cx), int(cy)),
        "radio":     int(r),
        "hull_circ": round(hull_circ, 2),
        "contorno":  contorno
    })

# Imprimir resultados en consola
print(f"Berries maduras detectadas:   {len(berries_maduras)}")
print(f"Berries inmaduras detectadas: {len(berries_inmaduras)}")

# ─────────────────────────────────────────────────────────────────────────────
# DIBUJAR RESULTADOS SOBRE LA IMAGEN ORIGINAL (SIN ESCALAR)
# Las coordenadas detectadas en la imagen escalada se multiplican por
# 1/ESCALA para volver al espacio de la imagen original
# ─────────────────────────────────────────────────────────────────────────────

resultado = img_rgb.copy()

# Dibujar circulos rojos para maduras
for berry in berries_maduras:
    # Escalar coordenadas de vuelta a la resolucion original
    cx = int(berry["centro"][0] / ESCALA)
    cy = int(berry["centro"][1] / ESCALA)
    r  = int(berry["radio"]     / ESCALA)

    cv2.circle(resultado, (cx, cy), r, (220, 30, 30), 3)
    cv2.circle(resultado, (cx, cy), 5, (220, 30, 30), -1)
    cv2.putText(resultado, "Madura",
                (cx - 28, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 30, 30), 2)

# Dibujar circulos verdes para inmaduras + contorno exacto
for berry in berries_inmaduras:
    # Escalar coordenadas de vuelta a la resolucion original
    cx = int(berry["centro"][0] / ESCALA)
    cy = int(berry["centro"][1] / ESCALA)
    r  = int(berry["radio"]     / ESCALA)

    # Escalar el contorno multiplicando todos sus puntos por 1/ESCALA
    contorno_escalado = (berry["contorno"] / ESCALA).astype(np.int32)

    cv2.circle(resultado, (cx, cy), r, (30, 200, 50), 3)
    cv2.drawContours(resultado, [contorno_escalado], -1, (80, 230, 80), 2)
    cv2.circle(resultado, (cx, cy), 5, (30, 200, 50), -1)
    cv2.putText(resultado, "Inmadura",
                (cx - 38, cy - r - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 50), 2)

# ─────────────────────────────────────────────────────────────────────────────
# HIGHLIGHTS (estilo del codigo original)
# Imagen en gris + color aislado encima para resaltar las detecciones
# Se escalan las mascaras de vuelta al tamaño original para el highlight
# ─────────────────────────────────────────────────────────────────────────────

# Escalar mascaras limpias de vuelta al tamaño original
mascara_madura_orig   = cv2.resize(mascara_madura_limpia,   (w, h), interpolation=cv2.INTER_NEAREST)
mascara_inmadura_orig = cv2.resize(mascara_inmadura_limpia, (w, h), interpolation=cv2.INTER_NEAREST)

# Convertir gris original a 3 canales para mezclarlo con imagen color
img_gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_3c       = cv2.cvtColor(img_gray_original, cv2.COLOR_GRAY2RGB)

# Aislar solo los pixeles de cada mascara sobre la imagen color original
aislado_madura   = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara_madura_orig)
aislado_inmadura = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara_inmadura_orig)

# Mezclar gris con color aislado (misma tecnica del codigo original)
highlight_madura   = cv2.addWeighted(img_gray_3c, 0.5, aislado_madura,   1.4, 0)
highlight_inmadura = cv2.addWeighted(img_gray_3c, 0.5, aislado_inmadura, 1.4, 0)

# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZACION
# ─────────────────────────────────────────────────────────────────────────────

imagenes = [img_rgb,
            mascara_madura_orig,
            resultado,
            mascara_inmadura_orig,
            highlight_madura,
            highlight_inmadura]

titulos  = ["Original",
            "Mascara madura",
            f"Detecciones: {len(berries_maduras)} maduras / {len(berries_inmaduras)} inmaduras",
            "Mascara inmadura",
            "Highlight madura",
            "Highlight inmadura"]

cmaps    = [None, "Reds", None, "Greens", None, None]

plt.figure(figsize=(16, 10))

for i, (imagen, titulo, cmap) in enumerate(zip(imagenes, titulos, cmaps)):
    plt.subplot(2, 3, i + 1)
    plt.title(titulo)
    plt.imshow(imagen, cmap=cmap)
    plt.axis("off")

plt.tight_layout()
plt.savefig("resultado.png", dpi=130, bbox_inches="tight")
plt.close()