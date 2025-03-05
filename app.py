import os
import streamlit as st
import pytesseract
import cv2
import numpy as np
import tempfile
import re
import logging
import requests
import base64
from pdf2image import convert_from_bytes
from PIL import Image
from doctr.models import ocr_predictor

# Configurações iniciais
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização do session state
if 'ocr_tool' not in st.session_state:
    st.session_state.ocr_tool = 'Tesseract'
if 'config' not in st.session_state:
    st.session_state.config = {}

# Funções de pré-processamento mantidas
def preprocess_image(image, binarization_threshold=31, denoise_strength=10):
    # ... (mantenha sua implementação atual) ...

# Funções de correção e validação mantidas
def correct_text_format(text):
    # ... (mantenha sua implementação atual) ...

def validate_extracted_text(text):
    # ... (mantenha sua implementação atual) ...

# Novas funções OCR
@st.cache_resource
def load_easyocr():
    import easyocr
    return easyocr.Reader(['en', 'pt'], gpu=False)

@st.cache_resource
def load_doctr(model_type='accurate'):
    return ocr_predictor(
        det_arch='db_resnet50' if model_type == 'accurate' else 'db_mobilenet_v3_large',
        reco_arch='crnn_vgg16_bn' if model_type == 'accurate' else 'crnn_mobilenet_v3_small',
        pretrained=True
    )

def ocr_tesseract(images, config):
    texts = []
    for img in images:
        processed = preprocess_image(np.array(img), 
                                   config['binarization_threshold'], 
                                   config['denoise_strength'])
        custom_config = f"--oem {config['oem']} --psm {config['psm']} -l por+eng"
        texts.append(pytesseract.image_to_string(processed, config=custom_config))
    return "\n".join(texts)

def ocr_easyocr(images, config):
    reader = load_easyocr()
    texts = []
    for img in images:
        results = reader.readtext(np.array(img))
        texts.append("\n".join([res[1] for res in results]))
    return "\n".join(texts)

def ocr_doctr(images, config):
    predictor = load_doctr(config['model_type'])
    texts = []
    for img in images:
        doc = predictor([np.array(img)])
        text = "\n".join([" ".join([w.value for w in line.words]) 
                        for block in doc.pages[0].blocks 
                        for line in block.lines])
        texts.append(text)
    return "\n".join(texts)

def ocr_space_api(images, config):
    texts = []
    for img in images:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': img_byte_arr.getvalue()},
            data={'apikey': config['api_key'], 'language': config['language']}
        )
        texts.append(response.json()['ParsedResults'][0]['ParsedText'])
    return "\n".join(texts)

# Interface
def config_page():
    st.title("Configuração do Sistema OCR")
    
    tools = {
        'Tesseract': {'icon': '🔍', 'desc': 'Melhor para documentos estruturados'},
        'EasyOCR': {'icon': '🚀', 'desc': 'Imagens com fundo complexo'},
        'DocTR': {'icon': '🧠', 'desc': 'Documentos modernos com layout complexo'},
        'OCR.Space': {'icon': '🌐', 'desc': 'API externa com alta precisão'}
    }
    
    cols = st.columns(4)
    for i, (tool, meta) in enumerate(tools.items()):
        with cols[i]:
            if st.button(f"{meta['icon']} {tool}", use_container_width=True):
                st.session_state.ocr_tool = tool
            st.caption(meta['desc'])
    
    st.divider()
    
    if st.session_state.ocr_tool == 'OCR.Space':
        st.session_state.config['api_key'] = st.text_input("API Key", type="password")
        st.session_state.config['language'] = st.selectbox("Idioma", ["por", "eng"])
    
    elif st.session_state.ocr_tool == 'DocTR':
        st.session_state.config['model_type'] = st.selectbox("Modelo", ["fast", "accurate"], index=1)
    
    else:
        with st.expander("Configurações Avançadas"):
            st.session_state.config.update({
                'binarization_threshold': st.slider("Limiar de Binarização", 10, 50, 31),
                'denoise_strength': st.slider("Remoção de Ruído", 5, 20, 10),
                'psm': st.slider("PSM", 3, 13, 6),
                'oem': st.slider("OEM", 1, 3, 3)
            })

def process_page():
    st.title("Processamento de Documentos")
    
    uploaded_file = st.file_uploader("Carregue seu documento", type=["pdf", "png", "jpg", "jpeg"])
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(tmp_file.read(), dpi=300)
            else:
                images = [Image.open(tmp_file.name)]
            
            try:
                if st.session_state.ocr_tool == 'Tesseract':
                    text = ocr_tesseract(images, st.session_state.config)
                elif st.session_state.ocr_tool == 'EasyOCR':
                    text = ocr_easyocr(images, st.session_state.config)
                elif st.session_state.ocr_tool == 'DocTR':
                    text = ocr_doctr(images, st.session_state.config)
                elif st.session_state.ocr_tool == 'OCR.Space':
                    text = ocr_space_api(images, st.session_state.config)
                
                text = correct_text_format(text)
                valid = validate_extracted_text(text)
                
                if valid:
                    st.success("Validação bem-sucedida!")
                else:
                    st.warning("Possíveis problemas na extração")
                
                st.download_button("Baixar Texto", text, file_name="texto_extraido.txt")
                st.text_area("Texto Extraído", text, height=400)
            
            except Exception as e:
                st.error(f"Erro: {str(e)}")
            finally:
                os.unlink(tmp_file.name)

# Controle de navegação
if 'page' not in st.session_state:
    st.session_state.page = 'config'

if st.session_state.page == 'config':
    config_page()
    if st.button("Iniciar Processamento"):
        st.session_state.page = 'process'
        st.rerun()
else:
    process_page()
    if st.button("Voltar para Configurações"):
        st.session_state.page = 'config'
        st.rerun()
