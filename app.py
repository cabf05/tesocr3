import os
import streamlit as st
import pytesseract
import cv2
import numpy as np
import tempfile
import re
import logging
import requests
import io
import traceback
from pdf2image import convert_from_bytes
from PIL import Image
from doctr.models import ocr_predictor

# Configurações iniciais
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Inicialização do session state
if 'page' not in st.session_state:
    st.session_state.page = 'config'
if 'ocr_tool' not in st.session_state:
    st.session_state.ocr_tool = None
if 'config' not in st.session_state:
    st.session_state.config = {}

def setup_ocr_tool():
    """Configura os parâmetros iniciais para cada ferramenta OCR"""
    tools = {
        'Tesseract': {
            'params': ['psm', 'oem', 'lang'],
            'defaults': {'psm': 6, 'oem': 3, 'lang': 'por+eng'}
        },
        'EasyOCR': {
            'params': ['langs'],
            'defaults': {'langs': ['pt', 'en']}
        },
        'DocTR': {
            'params': ['model_type'],
            'defaults': {'model_type': 'accurate'}
        },
        'OCR.Space': {
            'params': ['api_key', 'language'],
            'defaults': {'language': 'por'}
        },
        'Taggun': {
            'params': ['api_key', 'language'],
            'defaults': {'language': 'por'}
        }
    }
    return tools

def show_config_page():
    """Página de seleção e configuração da ferramenta OCR"""
    st.title("⚙️ Configuração do Sistema OCR")
    
    tools = setup_ocr_tool()
    selected_tool = st.selectbox(
        "Selecione a ferramenta OCR:",
        list(tools.keys()),
        index=0
    )
    
    st.session_state.ocr_tool = selected_tool
    st.session_state.config = tools[selected_tool]['defaults'].copy()
    
    # Configurações específicas
    with st.expander("🔧 Configurações Avançadas", expanded=True):
        if selected_tool == 'Tesseract':
            col1, col2 = st.columns(2)
            with col1:
                st.session_state.config['psm'] = st.slider("Modo Segmentação (PSM)", 3, 13, 6)
            with col2:
                st.session_state.config['oem'] = st.slider("Motor OCR (OEM)", 1, 3, 3)
            st.session_state.config['lang'] = st.text_input("Idiomas (ex: por+eng)", "por+eng")
        
        elif selected_tool == 'EasyOCR':
            st.session_state.config['langs'] = st.multiselect(
                "Idiomas",
                ['en', 'pt'],
                default=['pt', 'en']
            )
        
        elif selected_tool == 'DocTR':
            st.session_state.config['model_type'] = st.selectbox(
                "Tipo de Modelo",
                ["fast", "accurate"],
                index=1
            )
        
        elif selected_tool in ['OCR.Space', 'Taggun']:
            st.session_state.config['api_key'] = st.text_input(
                "Chave API",
                type="password",
                help="Obtenha em https://ocr.space/API ou https://taggun.io"
            )
            st.session_state.config['language'] = st.selectbox(
                "Idioma",
                ["por", "eng"]
            )
    
    if st.button("✅ Salvar Configurações"):
        st.session_state.page = 'process'
        st.rerun()

def process_file():
    """Página de processamento do arquivo"""
    st.title("📄 Processamento de Documentos")
    
    uploaded_file = st.file_uploader(
        "Carregue seu documento (PDF ou imagem)",
        type=["pdf", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        try:
            file_bytes = uploaded_file.read()
            
            if len(file_bytes) == 0:
                raise ValueError("Arquivo vazio ou corrompido")
            
            # Processar PDF
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(
                    file_bytes,
                    dpi=300,
                    poppler_path="/usr/bin"
                )
                st.info(f"PDF convertido em {len(images)} página(s)")
            else:
                images = [Image.open(io.BytesIO(file_bytes))]
            
            with st.spinner(f"Processando com {st.session_state.ocr_tool}..."):
                texto = process_ocr(images)
                texto_corrigido = correct_text_format(texto)
                
                if validate_extracted_text(texto_corrigido):
                    st.success("✅ Validação bem-sucedida!")
                else:
                    st.warning("⚠️ Possíveis problemas na extração")
                
                st.download_button(
                    "💾 Baixar Texto",
                    texto_corrigido,
                    file_name=f"texto_{st.session_state.ocr_tool}.txt"
                )
                st.text_area("📝 Texto Extraído", texto_corrigido, height=400)
        
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            logger.error(traceback.format_exc())
    
    if st.button("↩️ Voltar para Configurações"):
        st.session_state.page = 'config'
        st.rerun()

def process_ocr(images):
    """Executa o OCR de acordo com a ferramenta selecionada"""
    tool = st.session_state.ocr_tool
    config = st.session_state.config
    
    try:
        if tool == 'Tesseract':
            custom_config = f"--psm {config['psm']} --oem {config['oem']} -l {config['lang']}"
            return "\n".join([pytesseract.image_to_string(preprocess_image(np.array(img)), config=custom_config) for img in images])
        
        elif tool == 'EasyOCR':
            reader = easyocr.Reader(config['langs'], gpu=False)
            return "\n".join(["\n".join([res[1] for res in reader.readtext(np.array(img))]) for img in images])
        
        elif tool == 'DocTR':
            predictor = ocr_predictor(
                det_arch='db_resnet50' if config['model_type'] == 'accurate' else 'db_mobilenet_v3_large',
                reco_arch='crnn_vgg16_bn' if config['model_type'] == 'accurate' else 'crnn_mobilenet_v3_small',
                pretrained=True
            )
            return "\n".join(["\n".join([" ".join([w.value for w in line.words]) for block in predictor([np.array(img)]).pages[0].blocks for line in block.lines]) for img in images])
        
        elif tool == 'OCR.Space':
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
        
        elif tool == 'Taggun':
            img_byte_arr = io.BytesIO()
            images[0].save(img_byte_arr, format='PNG')
            response = requests.post(
                'https://api.taggun.io/api/receipt/v1/simple/file',
                headers={'apikey': config['api_key']},
                files={'file': img_byte_arr.getvalue()},
                data={'language': config['language']}
            )
            return response.json()['text']
    
    except Exception as e:
        raise RuntimeError(f"Falha no {tool}: {str(e)}")

# Funções auxiliares mantidas conforme necessidade
def preprocess_image(image, binarization_threshold=31, denoise_strength=10):
    # ... (manter implementação anterior) ...

def correct_text_format(text):
    # ... (manter implementação anterior) ...

def validate_extracted_text(text):
    # ... (manter implementação anterior) ...

# Controle de fluxo principal
if st.session_state.page == 'config':
    show_config_page()
else:
    process_file()
