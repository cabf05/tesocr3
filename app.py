import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import streamlit as st
import pytesseract
from PIL import Image
import numpy as np
import easyocr
import requests
from pdf2image import convert_from_bytes
import io
from doctr.io import DocumentFile
from doctr.models import ocr_predictor
import cv2
import base64

# Configuração inicial
st.set_page_config(page_title="Sistema OCR Multi-Ferramentas", layout="wide")

# Inicialização do session state
if 'ocr_tool' not in st.session_state:
    st.session_state.ocr_tool = None
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'page' not in st.session_state:
    st.session_state.page = 'config'

# Pré-carregar modelos
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['en', 'pt'], gpu=False)

@st.cache_resource
def load_doctr(model_type='accurate'):
    return ocr_predictor(
        det_arch='db_resnet50' if model_type == 'accurate' else 'db_mobilenet_v3_large',
        reco_arch='crnn_vgg16_bn' if model_type == 'accurate' else 'crnn_mobilenet_v3_small',
        pretrained=True
    )

# Página de configuração
def config_page():
    st.title("Configuração do Sistema OCR")
    st.markdown("### Selecione e configure sua ferramenta OCR")

    ocr_tool = st.selectbox(
        "Ferramenta OCR:",
        ["Tesseract OCR", "EasyOCR", "OCR.Space", "DocTR", "Taggun", "Free OCR API"],
        index=0
    )

    # Configurações específicas para cada ferramenta
    if ocr_tool == "Tesseract OCR":
        st.markdown("**Melhor para:** Documentos impressos de alta qualidade")
        col1, col2 = st.columns(2)
        with col1:
            languages = st.text_input("Idiomas (ex: por+eng)", "por+eng")
        with col2:
            psm = st.selectbox("Modo Segmentação", [3, 6, 11, 12], format_func=lambda x: f"PSM {x}")
        st.session_state.config = {"config": f"--psm {psm} -l {languages}"}

    elif ocr_tool == "EasyOCR":
        st.markdown("**Melhor para:** Imagens com fundo complexo")
        langs = st.multiselect("Idiomas", ["en", "pt"], default=["en", "pt"])
        st.session_state.config = {"langs": langs}

    elif ocr_tool == "OCR.Space":
        st.markdown("**Melhor para:** Uso via API")
        api_key = st.text_input("API Key", type="password")
        language = st.selectbox("Idioma", ["por", "eng"])
        st.session_state.config = {"api_key": api_key, "language": language, "engine": 1}

    elif ocr_tool == "DocTR":
        st.markdown("**Melhor para:** Layout complexo")
        doc_lang = st.selectbox("Idioma", ["en", "pt"])
        st.session_state.config = {"doc_lang": doc_lang, "model_type": "accurate"}

    elif ocr_tool == "Taggun":
        st.markdown("**Melhor para:** Documentos fiscais")
        api_key = st.text_input("API Key Taggun", type="password")
        language = st.selectbox("Idioma", ["por", "eng"])
        st.session_state.config = {"api_key": api_key, "language": language}

    elif ocr_tool == "Free OCR API":
        st.markdown("**Melhor para:** Uso rápido")
        language = st.selectbox("Idioma", ["por", "eng"])
        st.session_state.config = {"language": language}

    if st.button("Salvar Configuração"):
        st.session_state.ocr_tool = ocr_tool
        st.session_state.page = 'process'
        st.rerun()

# Página de processamento
def process_page():
    st.title("Extrair Texto de Documentos")
    
    uploaded_file = st.file_uploader("Carregue seu documento", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file:
        if st.button("Processar"):
            try:
                with st.spinner("Processando..."):
                    # Converter PDF para imagens
                    if uploaded_file.type == "application/pdf":
                        images = convert_from_bytes(uploaded_file.read())
                    else:
                        images = [Image.open(uploaded_file)]

                    full_text = ""
                    
                    # Tesseract
                    if st.session_state.ocr_tool == "Tesseract OCR":
                        for img in images:
                            text = pytesseract.image_to_string(img, config=st.session_state.config['config'])
                            full_text += text + "\n\n"
                    
                    # EasyOCR
                    elif st.session_state.ocr_tool == "EasyOCR":
                        reader = load_easyocr()
                        for img in images:
                            results = reader.readtext(np.array(img))
                            full_text += "\n".join([res[1] for res in results]) + "\n\n"
                    
                    # OCR.Space
                    elif st.session_state.ocr_tool == "OCR.Space":
                        for img in images:
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format='PNG')
                            response = requests.post(
                                'https://api.ocr.space/parse/image',
                                files={'file': img_bytes.getvalue()},
                                data={'apikey': st.session_state.config['api_key'], 'language': st.session_state.config['language']}
                            )
                            full_text += response.json()['ParsedResults'][0]['ParsedText'] + "\n\n"
                    
                    # DocTR
                    elif st.session_state.ocr_tool == "DocTR":
                        predictor = load_doctr()
                        for img in images:
                            doc = predictor([np.array(img)])
                            full_text += "\n".join([" ".join([w.value for w in line.words]) 
                                                for block in doc.pages[0].blocks 
                                                for line in block.lines]) + "\n\n"
                    
                    # Taggun
                    elif st.session_state.ocr_tool == "Taggun":
                        img_bytes = io.BytesIO()
                        images[0].save(img_bytes, format='PNG')
                        response = requests.post(
                            'https://api.taggun.io/api/receipt/v1/simple/file',
                            headers={'apikey': st.session_state.config['api_key']},
                            files={'file': img_bytes.getvalue()},
                            data={'language': st.session_state.config['language']}
                        )
                        full_text = response.json()['text']
                    
                    # Free OCR API
                    elif st.session_state.ocr_tool == "Free OCR API":
                        img_bytes = io.BytesIO()
                        images[0].save(img_bytes, format='PNG')
                        response = requests.post(
                            'https://api.ocr.space/parse/image',
                            files={'file': img_bytes},
                            data={'apikey': 'helloworld', 'language': st.session_state.config['language']}
                        )
                        full_text = response.json()['ParsedResults'][0]['ParsedText']

                    st.success("Texto extraído!")
                    st.text_area("Resultado", full_text, height=400)

            except Exception as e:
                st.error(f"Erro: {str(e)}")

    if st.button("Voltar"):
        st.session_state.page = 'config'
        st.rerun()

# Controle de navegação
if st.session_state.page == 'config':
    config_page()
else:
    process_page()
