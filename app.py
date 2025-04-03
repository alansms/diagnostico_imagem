import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import cv2
from io import BytesIO
import os
import io
import zipfile
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ================== CONFIGURAÇÕES DA PÁGINA ==================
st.set_page_config(
    page_title="Diagnóstico de Pneumonia por Radiografia",
    page_icon="logo_fiap.jpg",
    layout="wide"
)

# ================== CSS PERSONALIZADO (TEMA DARK COMPLETO) ==================
st.markdown("""
<style>
:root {
    --primary-bg: #1a1a1a;
    --secondary-bg: #2e2e2e;
    --primary-color: #ffffff;
    --accent-color: #ff0000;
    --font-family: 'Segoe UI', sans-serif;
}

/* Forçar cor do body e remover margens */
body {
    background-color: var(--primary-bg) !important;
    margin: 0;
    padding: 0;
}

/* Forçar cor do header (“Deploy”) */
[data-testid="stHeader"] {
    background-color: var(--primary-bg) !important;
}
[data-testid="stHeader"] * {
    color: var(--primary-color) !important;
}

/* Área principal */
.block-container {
    background-color: var(--primary-bg) !important;
    color: var(--primary-color) !important;
    font-family: var(--font-family) !important;
    padding-top: 2rem !important; /* Espaço no topo */
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: var(--primary-bg) !important;
    padding: 1rem !important;
}
[data-testid="stSidebar"] * {
    color: var(--primary-color) !important;
    font-family: var(--font-family) !important;
}

/* Terminal de logs */
[data-testid="stTextArea"] textarea {
    color: red !important;
    background-color: var(--primary-bg) !important;
    border: none;
}

/* File Uploader */
[data-testid="stFileUploadDropzone"] {
    max-width: 300px !important;
    margin: 0 auto !important;
    background-color: var(--secondary-bg) !important;
    border: 2px dashed #444 !important;
    border-radius: 12px !important;
    padding: 20px !important;
    font-size: 1rem;
    text-align: center;
    transition: background-color 0.3s ease;
}
[data-testid="stFileUploadDropzone"]:hover {
    background-color: #333 !important;
}

/* Botões genéricos */
.stButton > button {
    background-color: var(--accent-color);
    color: var(--primary-color);
    font-weight: 600;
    padding: 0.8rem 1.5rem;
    border: none;
    border-radius: 8px;
    margin-top: 0.5rem;
    cursor: pointer;
    transition: background-color 0.3s ease;
}
.stButton > button:hover {
    background-color: #cc0000;
}
.stButton > button:disabled {
    background-color: #888 !important;
}

/* Botão de download (st.download_button) */
[data-testid="stDownloadButton"] > button {
    background-color: #333 !important;
    color: #fff !important;
    border: 1px solid #555 !important;
    border-radius: 8px !important;
    padding: 0.6rem 1.2rem !important;
    font-weight: 600 !important;
    margin-top: 0.5rem;
    cursor: pointer;
}
[data-testid="stDownloadButton"] > button:hover {
    background-color: #444 !important;
}

/* Títulos */
h1 {
    margin-top: 1rem;
    color: var(--primary-color);
}

/* Responsividade para mobile */
@media only screen and (max-width: 600px) {
    .block-container {
        padding: 1rem !important;
    }
    [data-testid="stFileUploadDropzone"] {
        max-width: 90% !important;
        padding: 15px !important;
        font-size: 0.9rem;
    }
    h1 {
        font-size: 1.5rem;
    }
}
</style>
""", unsafe_allow_html=True)

# ================== SIDEBAR: LOGO, THRESHOLD E DEPURADOR ==================
st.sidebar.image("logo_fiap.jpg", use_container_width=True)

threshold = st.sidebar.slider(
    "Definir threshold para classificação", 0.0, 1.0, 0.5, 0.01
)

if "debug_logs" not in st.session_state:
    st.session_state["debug_logs"] = ""

def debug_log(msg: str):
    st.session_state["debug_logs"] += msg + "\n"

st.sidebar.header("Depurador (Terminal)")
st.sidebar.text_area("Logs", st.session_state["debug_logs"], height=300)
debug_log(f"Threshold definido: {threshold}")

# ================== CARREGAMENTO DO MODELO ==================
@st.cache_resource(show_spinner=False)
def load_model():
    debug_log("Carregando o modelo 'diagnostico_imagem.h5'...")
    model_ = tf.keras.models.load_model('diagnostico_imagem.h5')
    debug_log("Modelo carregado com sucesso!")
    return model_

model = load_model()

# ================== FUNÇÃO DE PRÉ-PROCESSAMENTO ==================
def preprocess_image(img, target_size=(180, 180)):
    debug_log("Iniciando o pré-processamento da imagem.")
    if img.mode != "RGB":
        img = img.convert("RGB")
        debug_log("Imagem convertida para RGB.")
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    debug_log("Imagem pré-processada e normalizada.")
    return img_array

# ================== CLASSE PARA CAPTURA DE CÂMERA ==================
class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

# ================== LAYOUT PRINCIPAL ==================
st.title("Diagnóstico de Pneumonia por Radiografia")
st.write("Escolha o modo de entrada para a imagem:")

modo = st.radio("Selecione o método de entrada:", ("Upload de Imagem", "Captura pela Câmera"))

if modo == "Upload de Imagem":
    st.write("Faça upload de uma imagem de radiografia para análise.")
    uploaded_file = st.file_uploader("Escolha uma imagem (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            debug_log("Arquivo de imagem recebido (upload). Iniciando leitura...")
            img = Image.open(uploaded_file)
            debug_log("Imagem carregada com sucesso via PIL (upload).")
            st.image(img, caption="Imagem Carregada", width=400)
            debug_log("Imagem exibida ao usuário (upload).")

            processed_img = preprocess_image(img)
            debug_log("Imagem pré-processada com sucesso (upload).")

            with st.spinner("Realizando a predição..."):
                debug_log("Enviando imagem (upload) para o modelo...")
                prediction = model.predict(processed_img)
                debug_log(f"Resultado bruto da predição (upload): {prediction}")

            probability = float(prediction[0][0])
            result = "PNEUMONIA" if probability > threshold else "NORMAL"
            st.write("**Resultado da análise:**", result)
            st.write("**Probabilidade:**", probability)
            debug_log(f"Classificação final (upload): {result} (Threshold: {threshold})")

        except Exception as e:
            st.error("Erro ao carregar ou processar a imagem (upload).")
            debug_log(f"Erro durante o processamento (upload): {str(e)}")

elif modo == "Captura pela Câmera":
    st.write("Clique em 'Iniciar Câmera' para ativar a câmera e em 'Capturar Imagem' para tirar uma foto. (Apenas a câmera traseira será utilizada)")
    # Atualização nas restrições: utiliza uma abordagem menos rígida para o parâmetro facingMode.
    media_constraints = {"video": {"facingMode": "environment"}, "audio": False}
    webrtc_ctx = webrtc_streamer(key="camera", video_transformer_factory=VideoTransformer,
                                 media_stream_constraints=media_constraints)

    if webrtc_ctx.video_transformer:
        if st.button("Capturar Imagem"):
            captured_frame = webrtc_ctx.video_transformer.frame
            if captured_frame is not None:
                # Converte BGR para RGB
                captured_rgb = cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB)
                st.image(captured_rgb, caption="Imagem Capturada", use_container_width=True)
                debug_log("Imagem capturada da câmera exibida.")

                # Converte para PIL e processa a imagem
                pil_img = Image.fromarray(captured_rgb)
                processed_img = preprocess_image(pil_img)

                with st.spinner("Realizando a predição..."):
                    debug_log("Enviando imagem (câmera) para o modelo...")
                    prediction = model.predict(processed_img)
                    debug_log(f"Resultado bruto da predição (câmera): {prediction}")

                probability = float(prediction[0][0])
                result = "PNEUMONIA" if probability > threshold else "NORMAL"
                st.write("**Resultado da análise:**", result)
                st.write("**Probabilidade:**", probability)
                debug_log(f"Classificação final (câmera): {result} (Threshold: {threshold})")
            else:
                st.warning("Nenhum frame capturado, tente novamente.")

# ================== SEÇÃO ADICIONAL: DOWNLOAD DE IMAGENS DE TESTE ==================
st.markdown("---")
st.write("**Imagens de Teste**")

images_dir = "extracted_images"  # Diretório com as imagens de teste
if os.path.isdir(images_dir):
    # Cria um arquivo ZIP na memória
    image_files = sorted(os.listdir(images_dir))
    with io.BytesIO() as buffer:
        with zipfile.ZipFile(buffer, "w") as zf:
            for image_name in image_files:
                file_path = os.path.join(images_dir, image_name)
                if os.path.isfile(file_path):
                    zf.write(file_path, arcname=image_name)
        zip_data = buffer.getvalue()

    st.download_button(
        label="Download Imagens (ZIP)",
        data=zip_data,
        file_name="imagens_para_teste.zip",
        mime="application/zip"
    )
else:
    st.warning(f"O diretório '{images_dir}' não foi encontrado.")
