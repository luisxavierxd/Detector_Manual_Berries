"""
Detección de frambuesas maduras e inmaduras mediante visión por computadora.

Técnicas empleadas:
  - Filtrado HSV con múltiples rangos de color.
  - Máscara de nitidez por Laplaciano (eliminación de fondo bokeh).
  - Morfología matemática (opening / closing).
  - Transformada de Hough para berries maduras.
  - Análisis de contornos + circularidad del hull convexo para inmaduras.

Salidas:
  - resultado.png  : imagen con detecciones anotadas.
  - pipeline.png   : figura vertical del pipeline paso a paso.
  - Consola        : tabla de coordenadas y tamaños de cada berry.
"""

# ---------------------------------------------------------------------------
# Librerías
# ---------------------------------------------------------------------------
import matplotlib                      # Debe importarse antes de pyplot.
matplotlib.use("Agg")                  # Backend sin GUI (escritura a archivo).

import cv2                              # noqa: E402
import numpy as np                        # noqa: E402
import matplotlib.pyplot as plt           # noqa: E402


# ===========================================================================
# 1. LECTURA Y CONVERSIÓN DE IMAGEN
# ===========================================================================

# OpenCV carga en BGR; se convierte a RGB para visualización con matplotlib.
img = cv2.imread("frambuesas.jpg")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ===========================================================================
# 2. ESCALADO DE IMAGEN
#
# HoughCircles es lento con radios grandes en imágenes de alta resolución
# porque el acumulador interno crece de forma no lineal con el radio máximo.
# Se reduce la imagen antes de procesar y se escalan las coordenadas al
# final para dibujar sobre la imagen original.
# ===========================================================================

ESCALA = 1  # Factor de escala: 0.5 = mitad del tamaño original.

h, w = img.shape[:2]
img_scaled = cv2.resize(img, (int(w * ESCALA), int(h * ESCALA)))
img_rgb_scaled = cv2.resize(img_rgb, (int(w * ESCALA), int(h * ESCALA)))

img_hsv_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2HSV)
img_gray_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)


# ===========================================================================
# 3. MÁSCARA DE NITIDEZ (SHARPNESS MASK)
#
# El Laplaciano mide el gradiente de la imagen: las zonas enfocadas tienen
# gradiente alto; el fondo bokeh tiene gradiente casi cero.  Se usa para
# eliminar el fondo desenfocado y conservar solo las berries en foco.
# ===========================================================================

# CV_64F permite valores negativos; se toma valor absoluto después.
laplaciano = cv2.Laplacian(img_gray_scaled, cv2.CV_64F)
sharpness_map = np.abs(laplaciano).astype(np.float32)

# Suavizar para que las zonas enfocadas formen regiones continuas.
sharpness_blur = cv2.GaussianBlur(sharpness_map, (41, 41), 0)

# Normalizar a rango 0-255 para poder umbralizar.
sharpness_norm = cv2.normalize(
    sharpness_blur, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Umbral: > 12 = zona enfocada (blanco); <= 12 = fondo bokeh (negro).
_, mascara_nitidez = cv2.threshold(
    sharpness_norm, 12, 255, cv2.THRESH_BINARY
)


# ===========================================================================
# 4. RANGOS DE COLOR HSV
#
# En OpenCV: H ∈ [0, 180], S ∈ [0, 255], V ∈ [0, 255].
# El rojo está partido en dos rangos (H=0-14 y H=155-180) porque ocupa
# ambos extremos del círculo de color (0° y 360°).
# ===========================================================================

# ── Maduras (rojo / carmesí) ─────────────────────────────────────────────
LOW_MADURA_A = np.array([0, 150, 20])    # Rojo puro: H bajo, S alta.
HIGH_MADURA_A = np.array([14, 255, 255])

LOW_MADURA_B = np.array([155, 60, 20])   # Carmesí / magenta (wraparound).
HIGH_MADURA_B = np.array([180, 255, 255])

# Naranja oscuro muy saturado (maduras en sombra).
# V < 150 evita capturar naranjas brillantes propios de inmaduras.
LOW_MADURA_C = np.array([14, 220, 20])
HIGH_MADURA_C = np.array([28, 255, 150])

# ── Inmaduras ────────────────────────────────────────────────────────────
# Verde-amarillo de berry: H = 20-32.
# Las hojas tienen H = 33-39 (~5 unidades de margen de separación).
LOW_INMADURA_VERDE = np.array([20, 90, 70])
HIGH_INMADURA_VERDE = np.array([32, 255, 255])

# Salmón / naranja brillante: berries semi-maduras (H = 7-22, V alto).
LOW_INMADURA_SALMON = np.array([7, 100, 140])
HIGH_INMADURA_SALMON = np.array([22, 255, 255])


# ===========================================================================
# 5. MÁSCARAS DE COLOR
#
# cv2.inRange devuelve 255 donde el píxel está dentro del rango, 0 fuera.
# ===========================================================================

# Máscaras individuales para berries maduras.
mask_madura_a = cv2.inRange(img_hsv_scaled, LOW_MADURA_A, HIGH_MADURA_A)
mask_madura_b = cv2.inRange(img_hsv_scaled, LOW_MADURA_B, HIGH_MADURA_B)
mask_madura_c = cv2.inRange(img_hsv_scaled, LOW_MADURA_C, HIGH_MADURA_C)

# Unión de los tres rangos de maduras con OR.
mascara_madura_color = cv2.bitwise_or(mask_madura_a, mask_madura_b)
mascara_madura_color = cv2.bitwise_or(mascara_madura_color, mask_madura_c)

# Retener solo píxeles enfocados.
mascara_madura = cv2.bitwise_and(mascara_madura_color, mascara_nitidez)

# Máscaras individuales para berries inmaduras.
mask_inmadura_verde = cv2.inRange(
    img_hsv_scaled, LOW_INMADURA_VERDE, HIGH_INMADURA_VERDE
)
mask_inmadura_salmon = cv2.inRange(
    img_hsv_scaled, LOW_INMADURA_SALMON, HIGH_INMADURA_SALMON
)

# Combinar verde y salmón con OR (antes de aplicar nitidez).
mascara_inmadura_color_pre = cv2.bitwise_or(
    mask_inmadura_verde, mask_inmadura_salmon
)

# Aplicar máscara de nitidez.
mascara_inmadura_color = cv2.bitwise_and(
    mascara_inmadura_color_pre, mascara_nitidez
)

# Eliminar de la máscara inmadura los píxeles ya clasificados como maduras.
mascara_inmadura = cv2.bitwise_and(
    mascara_inmadura_color, cv2.bitwise_not(mascara_madura)
)


# ===========================================================================
# 6. MORFOLOGÍA
#
# Opening  (erosión → dilatación): elimina puntos aislados de ruido.
# Closing  (dilatación → erosión): rellena huecos dentro de las berries.
# ===========================================================================

kernel_3 = np.ones((3, 3), np.uint8)
kernel_5 = np.ones((5, 5), np.uint8)
kernel_7 = np.ones((7, 7), np.uint8)

# Maduras: opening con kernel 5×5 + closing con kernel 3×3.
mascara_madura_open = cv2.morphologyEx(
    mascara_madura, cv2.MORPH_OPEN, kernel_5, iterations=1
)
mascara_madura_limpia = cv2.morphologyEx(
    mascara_madura_open, cv2.MORPH_CLOSE, kernel_3, iterations=2
)

# Inmaduras: kernel_7 en closing porque los drupelets dejan más huecos.
mascara_inmadura_open = cv2.morphologyEx(
    mascara_inmadura, cv2.MORPH_OPEN, kernel_5, iterations=1
)
mascara_inmadura_limpia = cv2.morphologyEx(
    mascara_inmadura_open, cv2.MORPH_CLOSE, kernel_7, iterations=2
)


# ===========================================================================
# 7. DETECCIÓN DE BERRIES MADURAS — HoughCircles
#
# Se aplica sobre la imagen gris (no la máscara) porque los drupelets
# crean huecos que rompen la detección por contornos.  El gradiente
# circular existe en la imagen gris aunque la máscara tenga huecos.
#
# Parámetros:
#   dp=1.3    : resolución del acumulador (>1 = más rápido).
#   minDist=90: distancia mínima entre centros detectados.
#   param1=65 : umbral superior del detector de bordes Canny interno.
#   param2=32 : umbral del acumulador (más alto = menos círculos, más precisos).
#   minRadius : descarta drupelets sueltos.
#   maxRadius : descarta clusters enteros.
#
# Validación: fill_ratio = píxeles de máscara / área del círculo.
# Se descarta si fill_ratio < 0.55.
# ===========================================================================

img_gray_blur = cv2.GaussianBlur(img_gray_scaled, (5, 5), 0)

circulos = cv2.HoughCircles(
    img_gray_blur,
    cv2.HOUGH_GRADIENT,
    dp=1.3,
    minDist=90,
    param1=65,
    param2=32,
    minRadius=35,
    maxRadius=103,
)

berries_maduras = []

if circulos is not None:
    for cx, cy, r in circulos[0]:
        cx, cy, r = int(round(cx)), int(round(cy)), int(round(r))

        # Máscara circular para este candidato.
        mascara_circulo = np.zeros_like(mascara_madura_limpia)
        cv2.circle(mascara_circulo, (cx, cy), r, 255, -1)

        # Proporción del círculo cubierta por color de berry madura.
        solapamiento = cv2.bitwise_and(mascara_madura_limpia, mascara_circulo)
        area_circulo = np.pi * r * r
        fill_ratio = np.sum(solapamiento > 0) / area_circulo

        if fill_ratio >= 0.55:
            berries_maduras.append({
                "centro": (cx, cy),
                "radio": r,
                "fill": round(fill_ratio, 2),
            })


# ===========================================================================
# 8. DETECCIÓN DE BERRIES INMADURAS — Contornos + Hull Circularity
#
# Se usan contornos porque los clusters verdes forman blobs más compactos.
#
# hull_circularity = 4π · Área_hull / Perímetro_hull²  ∈ (0, 1]
#   1.0 = círculo perfecto.
# Se usa el hull convexo porque los drupelets hacen el contorno muy
# irregular, pero el hull del conjunto es casi circular.
#   Berries: hull_circ ≥ 0.78   (rango típico: 0.82 – 0.92)
#   Hojas  : hull_circ < 0.78   (formas alargadas o irregulares)
# ===========================================================================

contornos, _ = cv2.findContours(
    mascara_inmadura_limpia,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE,
)

berries_inmaduras = []

for contorno in contornos:
    area = cv2.contourArea(contorno)
    if area < 2000:
        continue  # Descartar ruido y fragmentos de hoja.

    hull = cv2.convexHull(contorno)
    area_hull = cv2.contourArea(hull)
    perim_hull = cv2.arcLength(hull, True)

    if perim_hull == 0:
        continue

    hull_circ = (4 * np.pi * area_hull) / (perim_hull ** 2)
    if hull_circ < 0.78:
        continue

    (cx, cy), r = cv2.minEnclosingCircle(contorno)
    berries_inmaduras.append({
        "centro": (int(cx), int(cy)),
        "radio": int(r),
        "hull_circ": round(hull_circ, 2),
        "contorno": contorno,
    })


# ===========================================================================
# 9. RESULTADOS EN CONSOLA
# ===========================================================================

etiquetas = (
    ["Madura  "] * len(berries_maduras)
    + ["Inmadura"] * len(berries_inmaduras)
)
numeros = list(range(1, len(berries_maduras) + 1)) + list(range(1, len(berries_inmaduras) + 1))
berries = berries_maduras + berries_inmaduras

for etiqueta, num, b in zip(etiquetas, numeros, berries):
    print(f"{etiqueta} {num}: x={b['centro'][0]} y={b['centro'][1]} r={b['radio']}")


# ===========================================================================
# 10. DIBUJAR DETECCIONES SOBRE LA IMAGEN ORIGINAL
#
# Las coordenadas detectadas en la imagen escalada se dividen por ESCALA
# para volver al espacio de la imagen original.
# ===========================================================================

resultado = img_rgb.copy()

# Círculos rojos para berries maduras.
for berry in berries_maduras:
    cx = int(berry["centro"][0] / ESCALA)
    cy = int(berry["centro"][1] / ESCALA)
    r = int(berry["radio"] / ESCALA)

    cv2.circle(resultado, (cx, cy), r, (220, 30, 30), 3)
    cv2.circle(resultado, (cx, cy), 5, (220, 30, 30), -1)
    cv2.putText(
        resultado, "Madura",
        (cx - 28, cy - r - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 30, 30), 2,
    )

# Círculos verdes + contorno exacto para berries inmaduras.
for berry in berries_inmaduras:
    cx = int(berry["centro"][0] / ESCALA)
    cy = int(berry["centro"][1] / ESCALA)
    r = int(berry["radio"] / ESCALA)

    contorno_escalado = (berry["contorno"] / ESCALA).astype(np.int32)

    cv2.circle(resultado, (cx, cy), r, (30, 200, 50), 3)
    cv2.drawContours(resultado, [contorno_escalado], -1, (80, 230, 80), 2)
    cv2.circle(resultado, (cx, cy), 5, (30, 200, 50), -1)
    cv2.putText(
        resultado, "Inmadura",
        (cx - 38, cy - r - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 200, 50), 2,
    )


# ===========================================================================
# 11. HIGHLIGHTS
#
# Imagen en escala de grises con el color de cada clase superpuesto para
# resaltar visualmente las detecciones (estilo del código original).
# ===========================================================================

# Escalar máscaras limpias de vuelta al tamaño original.
mascara_madura_orig = cv2.resize(
    mascara_madura_limpia, (w, h), interpolation=cv2.INTER_NEAREST
)
mascara_inmadura_orig = cv2.resize(
    mascara_inmadura_limpia, (w, h), interpolation=cv2.INTER_NEAREST
)

# Imagen gris original en 3 canales para mezclarla con la imagen a color.
img_gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray_3c = cv2.cvtColor(img_gray_original, cv2.COLOR_GRAY2RGB)

# Aislar solo los píxeles de cada clase sobre la imagen original a color.
aislado_madura = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara_madura_orig)
aislado_inmadura = cv2.bitwise_and(img_rgb, img_rgb, mask=mascara_inmadura_orig)

# Mezclar gris + color aislado.
highlight_madura = cv2.addWeighted(img_gray_3c, 0.5, aislado_madura, 1.4, 0)
highlight_inmadura = cv2.addWeighted(img_gray_3c, 0.5, aislado_inmadura, 1.4, 0)


# ===========================================================================
# 12. FIGURA PRINCIPAL — Resumen de resultados
# ===========================================================================

imagenes_resumen = [
    img_rgb,
    mascara_madura_orig,
    resultado,
    mascara_inmadura_orig,
    highlight_madura,
    highlight_inmadura,
]

titulos_resumen = [
    "Original",
    "Máscara madura",
    f"Detecciones: {len(berries_maduras)} maduras / {len(berries_inmaduras)} inmaduras",
    "Máscara inmadura",
    "Highlight madura",
    "Highlight inmadura",
]

cmaps_resumen = [None, "Reds", None, "Greens", None, None]

fig_resumen, axes = plt.subplots(2, 3, figsize=(16, 10))

for ax, imagen, titulo, cmap in zip(
    axes.flat, imagenes_resumen, titulos_resumen, cmaps_resumen
):
    ax.imshow(imagen, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

fig_resumen.tight_layout()
fig_resumen.savefig("resultado.png", dpi=130, bbox_inches="tight")
plt.close(fig_resumen)


# ===========================================================================
# 13. FIGURAS DE PIPELINE — Proceso de filtrado paso a paso
#
# Tres figuras independientes guardadas como PNG separados:
#   pipeline_compartido.png : pasos comunes (gris, Laplaciano, nitidez).
#   pipeline_maduras.png    : rama de berries maduras.
#   pipeline_inmaduras.png  : rama de berries inmaduras.
# ===========================================================================

# Laplaciano crudo (CV_64F con negativos) normalizado a uint8 para display.
laplaciano_viz_raw = cv2.normalize(
    laplaciano, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# |Laplaciano| (sharpness_map es float32) normalizado a uint8 para display.
laplaciano_viz_abs = cv2.normalize(
    sharpness_map, None, 0, 255, cv2.NORM_MINMAX
).astype(np.uint8)

# Detecciones Hough sobre imagen gris escalada (maduras).
viz_hough = cv2.cvtColor(img_gray_scaled, cv2.COLOR_GRAY2RGB)
for b in berries_maduras:
    cv2.circle(viz_hough, b["centro"], b["radio"], (220, 30, 30), 2)
    cv2.circle(viz_hough, b["centro"], 4, (220, 30, 30), -1)

# Detecciones por contorno sobre imagen gris escalada (inmaduras).
viz_contornos = cv2.cvtColor(img_gray_scaled, cv2.COLOR_GRAY2RGB)
for b in berries_inmaduras:
    cv2.circle(viz_contornos, b["centro"], b["radio"], (30, 200, 50), 2)
    cv2.drawContours(viz_contornos, [b["contorno"]], -1, (80, 230, 80), 2)

# ── Figura 1: pasos compartidos ──────────────────────────────────────────

imgs1 = [img_rgb_scaled, img_gray_scaled,
         laplaciano_viz_raw, laplaciano_viz_abs, sharpness_norm,
         mascara_nitidez]
titulos1 = ["1 · Original", "2 · Escala de grises",
            "3 · Laplaciano (CV_64F)", "4 · |Laplaciano| (abs)",
            "5 · Normalizado 0-255 (suavizado)", "6 · Máscara nitidez (umbral=12)"]
cmaps1 = [None, "gray", "bwr", "hot", "hot", "gray"]

fig1, axes1 = plt.subplots(6, 1, figsize=(7, 21))
fig1.suptitle("Pipeline — pasos compartidos", fontsize=12, fontweight="bold")

for ax, img, titulo, cmap in zip(axes1, imgs1, titulos1, cmaps1):
    ax.imshow(img, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

fig1.tight_layout()
fig1.savefig("pipeline_compartido.png", dpi=110, bbox_inches="tight")
plt.close(fig1)

# ── Figura 2: rama maduras ───────────────────────────────────────────────

imgs2 = [mascara_madura_color, mascara_madura, mascara_madura_open,
         mascara_madura_limpia, viz_hough]
titulos2 = ["5 · Máscara color (OR a+b+c)", "6 · AND nitidez",
            "7 · Opening (5×5)", "8 · Closing (3×3) — final",
            f"9 · HoughCircles → {len(berries_maduras)} berries"]
cmaps2 = ["Reds", "Reds", "Reds", "Reds", None]

fig2, axes2 = plt.subplots(5, 1, figsize=(7, 17))
fig2.suptitle("Pipeline — rama MADURAS", fontsize=12, fontweight="bold",
              color="#cc1a1a")

for ax, img, titulo, cmap in zip(axes2, imgs2, titulos2, cmaps2):
    ax.imshow(img, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

fig2.tight_layout()
fig2.savefig("pipeline_maduras.png", dpi=110, bbox_inches="tight")
plt.close(fig2)

# ── Figura 3: rama inmaduras ─────────────────────────────────────────────

imgs3 = [mascara_inmadura_color_pre, mascara_inmadura, mascara_inmadura_open,
         mascara_inmadura_limpia, viz_contornos]
titulos3 = ["5 · Máscara color (OR verde+salmón)", "6 · AND nitidez — excl. maduras",
            "7 · Opening (5×5)", "8 · Closing (7×7) — final",
            f"9 · Contornos + Hull → {len(berries_inmaduras)} berries"]
cmaps3 = ["Greens", "Greens", "Greens", "Greens", None]

fig3, axes3 = plt.subplots(5, 1, figsize=(7, 17))
fig3.suptitle("Pipeline — rama INMADURAS", fontsize=12, fontweight="bold",
              color="#1a7a2e")

for ax, img, titulo, cmap in zip(axes3, imgs3, titulos3, cmaps3):
    ax.imshow(img, cmap=cmap)
    ax.set_title(titulo, fontsize=10)
    ax.axis("off")

fig3.tight_layout()
fig3.savefig("pipeline_inmaduras.png", dpi=110, bbox_inches="tight")
plt.close(fig3)

print("Archivos guardados: resultado.png, pipeline_compartido.png, "
      "pipeline_maduras.png, pipeline_inmaduras.png")