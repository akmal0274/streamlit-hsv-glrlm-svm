import streamlit as st
import joblib
import cv2
import numpy as np
from PIL import Image
from rembg import remove
from glrlm import GLRLM
import pandas as pd
import matplotlib.pyplot as plt

st.markdown(
    """
    <style>
        section[data-testid="stSidebar"] div.stButton button {
            width: 100%;
        }
        div[data-testid="stImage"] {
            display: block;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stAppViewContainer"] {
            background-color: #000000;
        }

        .text-prediction {
            font-size: 24px;
            font-weight: bold;
        }
    </style>
    """, unsafe_allow_html=True
)

def load_model(file_path):
  with open(file_path, 'rb') as file:
    return joblib.load(file)

def remove_background(input_image):
    try:
        # with open(input_image, "rb") as f:
        #     image_bytes = f.read()
        image_bytes = input_image.getvalue()
        result = remove(image_bytes)
        img_no_bg = cv2.imdecode(np.frombuffer(result, np.uint8), cv2.IMREAD_UNCHANGED)
        return img_no_bg
    except Exception as e:
        st.error(f"Error in background removal: {e}")
        return None

def resize_image(image, size=(128, 128)):
    if image is not None:
        resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        return resized_image
    else:
        print("Error: Cannot resize a None image.")
        return None

def rgb_to_hsv(image):
    
    readImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    r, g, b = np.mean(readImage, axis=(0, 1)) / 255.0
    cmax = max(r, g, b)
    cmin = min(r, g, b)
    v = cmax

    if cmax == 0:
        s = 0
    else:
        s = 1 - (cmin/cmax)

    if cmax == cmin:
        h = 0
    elif cmax == r:
        h = 60 * ((g - b) / (cmax - cmin) % 6) 
    elif cmax == g:
        h = 60 * (2 + (b - r) / (cmax - cmin))
    else:
        h = 60 * (4 + (r - g) / (cmax - cmin))
    if h < 0:
        h += 360
    h = int(h / 2)
    s = int(s * 255)
    v = int(v * 255)
    return (h, s, v)

def glrlm_features(image):
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    app = GLRLM()
    glrlm = app.get_features(gray_image)
    SRE = glrlm.SRE
    LRE = glrlm.LRE
    GLU = glrlm.GLU
    RLU = glrlm.RLU
    RPC = glrlm.RPC
    return SRE, LRE, GLU, RLU, RPC

def extract_feature(image):
    app = GLRLM()
    grayscale_image = app._GLRLM__check_and_convert_to_gray(image)
    normalized_image = app._GLRLM__normalizer.normalizationImage(grayscale_image, 0, 8)
    degree_obj = app._GLRLM__degree.create_matrix(normalized_image, 8)

    return degree_obj

def __SRE(degree_object):
    input_matrix = degree_object.Degrees
    matSRE = []
    for input_matrix in input_matrix:
        S = 0
        SRE = 0
        for x in range(input_matrix.shape[1]):
            for y in range(input_matrix.shape[0]):
                S += input_matrix[y][x]

        for x in range(input_matrix.shape[1]):
            Rj = 0
            for y in range(input_matrix.shape[0]):
                Rj += input_matrix[y][x]

            SRE += (Rj/S)/((x+1)**2)
            # print('( ',Rj,'/',S,' ) / ',(x+1)**2)
        SRE = round(SRE, 3)
        matSRE.append(SRE)
    # print('Perhitungan SRE')
    return matSRE


def __LRE(degree_object):
    input_matrix = degree_object.Degrees
    matLRE = []
    for input_matrix in input_matrix:
        S = 0
        LRE = 0
        for x in range(input_matrix.shape[1]):
            for y in range(input_matrix.shape[0]):
                S += input_matrix[y][x]

        for x in range(input_matrix.shape[1]):
            Rj = 0
            for y in range(input_matrix.shape[0]):
                Rj += input_matrix[y][x]

            LRE += (Rj * ((x + 1) ** 2)) / S
            # print('( ', Rj ,' * ',((x + 1) ** 2), ' ) /', S)
        LRE = round(LRE, 3)
        matLRE.append(LRE)
    # print('Perhitungan LRE')
    return matLRE


def __GLU(degree_object):
    input_matrix = degree_object.Degrees
    matGLU = []
    for input_matrix in input_matrix:
        S = 0
        GLU = 0
        for x in range(input_matrix.shape[1]):
            for y in range(input_matrix.shape[0]):
                S += input_matrix[y][x]

        for x in range(input_matrix.shape[1]):
            Rj = 0
            for y in range(input_matrix.shape[0]):
                Rj += input_matrix[y][x]

            GLU += ((x + 1) ** 2) / S
            # print('( ',((x + 1) ** 2), ' ) /', S)
        GLU = round(GLU, 3)
        matGLU.append(GLU)
    # print('Perhitungan GLU')
    return matGLU


def __RLU(degree_object):
    input_matrix = degree_object.Degrees
    matRLU = []
    for input_matrix in input_matrix:
        S = 0
        RLU = 0
        for x in range(input_matrix.shape[1]):
            for y in range(input_matrix.shape[0]):
                S += input_matrix[y][x]

        for x in range(input_matrix.shape[1]):
            Rj = 0
            for y in range(input_matrix.shape[0]):
                Rj += input_matrix[y][x]

            RLU += (Rj ** 2) / S
            # print('( ', (Rj ** 2), ' ) /', S)
        RLU = round(RLU, 3)
        matRLU.append(RLU)
    # print('Perhitungan RLU')
    return matRLU


def __RPC(degree_object):
    input_matrix = degree_object.Degrees
    matRPC = []
    for input_matrix in input_matrix:
        S = 0
        RPC = 0
        for x in range(input_matrix.shape[1]):
            for y in range(input_matrix.shape[0]):
                S += input_matrix[y][x]

        for x in range(input_matrix.shape[1]):
            Rj = 0
            for y in range(input_matrix.shape[0]):
                Rj += input_matrix[y][x]

            RPC += (Rj) / (input_matrix.shape[0]*input_matrix.shape[1])
            # print('( ', (Rj), ' ) /', input_matrix.shape[0]*input_matrix.shape[1])
        RPC = round(RPC, 3)
        matRPC.append(RPC)
    # print('Perhitungan RPC')
    return matRPC
  
svm_model = load_model('model_svm_linear_1_8020/svm_linear_1_8020_model.pkl')



if "page" not in st.session_state:
  st.session_state.page = "beranda"

st.sidebar.title("Menu")
if st.sidebar.button("Beranda"):
  st.session_state.page = "beranda"
if st.sidebar.button("Prediksi"):
  st.session_state.page = "prediksi"
if st.sidebar.button("Evaluasi"):
  st.session_state.page = "evaluasi"

if st.session_state.page == "beranda":
  st.title("KLASIFIKASI KUALITAS BIJI JAGUNG DENGAN METODE HSV DAN GLRLM MENGGUNAKAN SVM")

if st.session_state.page == "prediksi":
  st.title("KLASIFIKASI KUALITAS BIJI JAGUNG DENGAN METODE HSV DAN GLRLM MENGGUNAKAN SVM")
  uploaded_file = st.file_uploader("Upload gambar", type=["png"])
  st.text("PNG only!")
  if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # file_path = save_uploaded_file_to_directory(uploaded_file)

    col1,col2,col3=st.columns(3)
    with col2:
        st.image(image, caption="Gambar yang Diunggah")
    if st.button("Prediksi Kualitas Jagung"):
        image_no_bg = remove_background(uploaded_file)
        if image_no_bg is not None:
            resized_image = resize_image(image_no_bg)
            if resized_image is not None:
                resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
                hsv_image = cv2.cvtColor(resized_image_rgb, cv2.COLOR_RGB2HSV)
                grayscale_image = cv2.cvtColor(resized_image_rgb, cv2.COLOR_RGB2GRAY)
                degree_obj = extract_feature(resized_image)
                degree = [0,45,90,135]
                SRE = __SRE(degree_obj)
                LRE = __LRE(degree_obj)
                RLU = __RLU(degree_obj)
                GLU = __GLU(degree_obj)
                RPC = __RPC(degree_obj)

                img1,img2,img3=st.columns(3)

                with img1:
                    st.image(resized_image_rgb, caption="Gambar Removed Background dan Resized")

                with img2:
                    st.image(hsv_image, caption="Gambar HSV")

                with img3:
                    st.image(grayscale_image, caption="Gambar Grayscale")

                h, s, v = rgb_to_hsv(resized_image)
                sre, lre, glu, rlu, rpc = glrlm_features(resized_image)

                feature_names_HSV = ["Hue", "Saturation", "Value"]
                features_value_HSV = [h, s, v]
                features_value_HSV = np.array(features_value_HSV).reshape(1, -1)
                features_df_HSV = pd.DataFrame(features_value_HSV, columns=feature_names_HSV)
                st.table(features_df_HSV)

                feature_names_GLRLM = ["Degree", "SRE", "LRE", "GLN", "RLN", "RP"]
                features_df_GLRLM = pd.DataFrame(columns=feature_names_GLRLM)
                for i in range(len(degree)):
                    features_df_GLRLM = features_df_GLRLM._append(pd.Series({'Degree':degree[i],'SRE':SRE[i],'LRE':LRE[i],'GLN':GLU[i],'RLN':RLU[i],'RP':RPC[i]}), ignore_index=True)
                st.table(features_df_GLRLM)

                features = [h, s, v, sre, lre, glu, rlu, rpc]
                features = np.array(features).reshape(1, -1)
                prediction = svm_model.predict(features)
                print(prediction)
                st.markdown(f'<div class="text-prediction">Hasil Prediksi: <strong>{prediction[0]}</strong></div>', unsafe_allow_html=True)

if st.session_state.page == "evaluasi":
  st.title("EVALUASI MODEL")
  image_conf_matrix = Image.open('image/confusion_matrix.png')
  image_classification_report = Image.open('image/classification_report.png')
  st.image(image_conf_matrix, caption="Confusion Matrix")
  st.image(image_classification_report, caption="Classification Report")