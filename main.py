import cv2
import numpy as np
import os


def mejorar_foto(ruta):

    imagen = cv2.imread(ruta)

    if imagen is None:
        print("No se encontró la foto")
        return

    print("Foto encontrada")

    # --------------------------------
    # 1. ANALIZAR BRILLO Y CONTRASTE
    # --------------------------------

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    brillo = np.mean(gris)
    contraste = np.std(gris)

    print("Brillo original:", brillo)
    print("Contraste original:", contraste)

    # --------------------------------
    # 2. ILUMINACION AUTOMATICA
    # --------------------------------

    if brillo < 60:
        gamma = 1.10
    elif brillo < 90:
        gamma = 1.06
    elif brillo < 120:
        gamma = 1.03
    elif brillo > 190:
        gamma = 0.96
    else:
        gamma = 1.0

    tabla = np.array([
        ((i / 255.0) ** (1 / gamma)) * 255
        for i in np.arange(0, 256)
    ]).astype("uint8")

    imagen = cv2.LUT(imagen, tabla)

    # --------------------------------
    # 3. ALTAS LUCES Y SOMBRAS
    # --------------------------------

    lab = cv2.cvtColor(imagen, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    # Analizamos las zonas oscuras y claras
    sombras = np.mean(l[l < 80])
    altas_luces = np.mean(l[l > 180])

    print("Sombras:", sombras)
    print("Altas luces:", altas_luces)

    # Levantar sombras
    if sombras < 60:
        l = np.where(
            l < 100,
            l + 8,
            l
        )

    # Bajar altas luces
    if altas_luces > 210:
        l = np.where(
            l > 180,
            l - 8,
            l
        )

    l = np.clip(l, 0, 255).astype(np.uint8)

    lab = cv2.merge((l, a, b))

    imagen = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    # --------------------------------
    # 4. SATURACION AUTOMATICA
    # --------------------------------

    hsv = cv2.cvtColor(imagen, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)

    saturacion = np.mean(s)

    print("Saturación original:", saturacion)

    if saturacion < 70:
        s = cv2.convertScaleAbs(s, alpha=1.20)
    elif saturacion < 110:
        s = cv2.convertScaleAbs(s, alpha=1.10)
    elif saturacion > 180:
        s = cv2.convertScaleAbs(s, alpha=0.90)

    hsv = cv2.merge((h, s, v))

    imagen = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # --------------------------------
    # 5. REDUCCION DE RUIDO
    # --------------------------------

    imagen = cv2.fastNlMeansDenoisingColored(
        imagen,
        None,
        5,
        5,
        7,
        21
    )

    # --------------------------------
    # 6. NITIDEZ
    # --------------------------------

    suavizada = cv2.GaussianBlur(
        imagen,
        (0, 0),
        3
    )

    imagen = cv2.addWeighted(
        imagen,
        1.4,
        suavizada,
        -0.4,
        0
    )

    # --------------------------------
    # 7. GUARDAR
    # --------------------------------

    os.makedirs("resultados", exist_ok=True)

    nombre = os.path.basename(ruta)

    salida = os.path.join(
        "resultados",
        "mejorada_" + nombre
    )

    cv2.imwrite(salida, imagen)

    print("Foto mejorada guardada en:")
    print(salida)

    print("--------------------------------")


# --------------------------------
# PROCESAR TODAS LAS FOTOS
# --------------------------------

extensiones = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tiff"
)

for nombre in os.listdir("fotos"):

    if nombre.lower().endswith(extensiones):

        ruta = os.path.join("fotos", nombre)

        mejorar_foto(ruta)