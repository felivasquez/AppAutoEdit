from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os
import zipfile

app = Flask(__name__)

UPLOAD_FOLDER = "fotos"
RESULT_FOLDER = "resultados"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)


def mejorar_foto(ruta, salida):

    imagen = cv2.imread(ruta)

    if imagen is None:
        return False

    # --------------------------------
    # 1. ANALIZAR BRILLO
    # --------------------------------

    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

    brillo = np.mean(gris)
    contraste = np.std(gris)

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

    sombras = np.mean(l[l < 80])
    altas_luces = np.mean(l[l > 180])

    if sombras < 60:
        l = np.where(
            l < 100,
            l + 8,
            l
        )

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
    # 4. SATURACION
    # --------------------------------

    hsv = cv2.cvtColor(
        imagen,
        cv2.COLOR_BGR2HSV
    )

    h, s, v = cv2.split(hsv)

    saturacion = np.mean(s)

    if saturacion < 70:
        s = cv2.convertScaleAbs(
            s,
            alpha=1.20
        )

    elif saturacion < 110:
        s = cv2.convertScaleAbs(
            s,
            alpha=1.10
        )

    elif saturacion > 180:
        s = cv2.convertScaleAbs(
            s,
            alpha=0.90
        )

    hsv = cv2.merge((h, s, v))

    imagen = cv2.cvtColor(
        hsv,
        cv2.COLOR_HSV2BGR
    )

    # --------------------------------
    # 5. REDUCCION DE RUIDO SUAVE
    # --------------------------------

    imagen = cv2.fastNlMeansDenoisingColored(
        imagen,
        None,
        3,
        3,
        7,
        15
    )

    # --------------------------------
    # 6. NITIDEZ SUAVE
    # --------------------------------

    suavizada = cv2.GaussianBlur(
        imagen,
        (0, 0),
        1.2
    )

    imagen = cv2.addWeighted(
        imagen,
        1.2,
        suavizada,
        -0.2,
        0
    )

    # --------------------------------
    # GUARDAR
    # --------------------------------

    cv2.imwrite(salida, imagen)

    return True


@app.route("/")
def inicio():
    return render_template("index.html")


@app.route("/editar", methods=["POST"])
def editar():

    archivos = request.files.getlist("fotos")

    resultados = []

    for archivo in archivos:

        if archivo.filename == "":
            continue

        nombre = archivo.filename

        entrada = os.path.join(
            UPLOAD_FOLDER,
            nombre
        )

        salida = os.path.join(
            RESULT_FOLDER,
            "mejorada_" + nombre
        )

        archivo.save(entrada)

        mejorar_foto(
            entrada,
            salida
        )

        resultados.append(salida)

    # --------------------------------
    # CREAR ZIP
    # --------------------------------

    zip_path = os.path.join(
        RESULT_FOLDER,
        "fotos_mejoradas.zip"
    )

    with zipfile.ZipFile(
        zip_path,
        "w",
        zipfile.ZIP_DEFLATED
    ) as zip_file:

        for archivo in resultados:

            zip_file.write(
                archivo,
                os.path.basename(archivo)
            )

    # --------------------------------
    # DEVOLVER ZIP
    # --------------------------------

    return send_file(
        zip_path,
        as_attachment=True,
        download_name="fotos_mejoradas.zip"
    )

if __name__ == "__main__":
    app.run(debug=True)