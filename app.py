import os
import streamlit as st
import pytesseract
import cv2
import numpy as np
import re
import logging
import requests
import io
import traceback
import easyocr
from pdf2image import convert_from_bytes
from PIL import Image
from doctr.models import ocr_predictor

# ===========================================
# CONFIGURAÇÕES INICIAIS
# ===========================================
os.environ["TESSDATA_PREFIX"] = "/usr/share/tesseract-ocr/4.00/tessdata/"
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===========================================
# GERENCIAMENTO DE ESTADO
# ===========================================
if 'page' not in st.session_state:
    st.session_state.page = 'config'
if 'ocr_tool' not in st.session_state:
    st.session_state.ocr_tool = None
if 'config' not in st.session_state:
    st.session_state.config = {}

# ===========================================
# FUNÇÕES AUXILIARES
# ===========================================
@st.cache_resource
def load_easyocr(langs):
    return easyocr.Reader(langs, gpu=False)

def preprocess_image(image, binarization_threshold=31, denoise_strength=10):
    """Melhora a qualidade da imagem para OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_strength, 
                                          templateWindowSize=7, searchWindowSize=21)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, binarization_threshold, 2)
        return thresh
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        return None

def correct_text_format(text):
    """Corrige formatos comuns de texto em NFS-e."""
    corrections = {
        r'(\d{2})[\.]?(\d{3})[\.]?(\d{3})[/]?0001[-]?(\d{2})': r'\1.\2.\3/0001-\4',
        r'(\d{2})[\/.-](\d{2})[\/.-](\d{4})': r'\1/\2/\3',
        r'R\$ (\d+)[,.](\d{2})': r'R$\1,\2'
    }
    for pattern, replacement in corrections.items():
        text = re.sub(pattern, replacement, text)
    return text

def validate_extracted_text(text):
    """Valida se o texto contém informações chave."""
    required_patterns = [
        r'NOTA FISCAL DE SERVIÇOS ELETRÔNICA',
        r'CNPJ',
        r'Valor Total',
        r'Data e Hora de Emissão'
    ]
    for pattern in required_patterns:
        if not re.search(pattern, text, re.IGNORECASE):
            return False
    return True

def handle_ocrspace_api(img, config):
    """Processamento específico para OCR.Space com tratamento de erros"""
    try:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        response = requests.post(
            'https://api.ocr.space/parse/image',
            files={'file': ('document.png', img_byte_arr.getvalue(), 'image/png')},
            data={
                'apikey': config['api_key'],
                'language': config['language'],
                'filetype': 'PNG',
                'OCREngine': 2
            }
        )
        result = response.json()
        
        if response.status_code != 200 or result.get('IsErroredOnProcessing', False):
            error_msg = result.get('ErrorMessage', ['Erro desconhecido'])[0]
            raise Exception(f"Erro na API: {error_msg}")
            
        return result['ParsedResults'][0]['ParsedText']
    except Exception as e:
        raise RuntimeError(f"Falha no OCR.Space: {str(e)}")

# ===========================================
# PÁGINA DE CONFIGURAÇÃO
# ===========================================
def show_config_page():
    """Interface de configuração do OCR"""
    st.title("⚙️ Configuração do Sistema OCR")
    
    tools = {
        'Tesseract': {'icon': '🔍', 'desc': 'Documentos impressos de alta qualidade'},
        'EasyOCR': {'icon': '🎨', 'desc': 'Imagens com fundo complexo'},
        'DocTR': {'icon': '🧠', 'desc': 'Layouts complexos com deep learning'},
        'OCR.Space': {'icon': '🌐', 'desc': 'API externa para múltiplos formatos'},
        'Taggun': {'icon': '🧾', 'desc': 'Documentos fiscais/recibos'}
    }
    
    selected_tool = st.selectbox(
        "Selecione a ferramenta OCR:",
        list(tools.keys()),
        format_func=lambda x: f"{tools[x]['icon']} {x} - {tools[x]['desc']}"
    )
    
    with st.expander("🔧 Configurações Avançadas", expanded=True):
        if selected_tool == 'Tesseract':
            col1, col2 = st.columns(2)
            with col1:
                psm = st.slider("Modo Segmentação (PSM)", 3, 13, 6)
            with col2:
                oem = st.slider("Motor OCR (OEM)", 1, 3, 3)
            lang = st.text_input("Idiomas (ex: por+eng)", "por+eng")
            st.session_state.config = {'psm': psm, 'oem': oem, 'lang': lang}
        
        elif selected_tool == 'EasyOCR':
            langs = st.multiselect("Idiomas", ['en', 'pt'], default=['pt', 'en'])
            st.session_state.config = {'langs': langs}
            st.warning("A primeira execução pode demorar para baixar os modelos")
        
        elif selected_tool == 'DocTR':
            model_type = st.selectbox("Tipo de Modelo", ["fast", "accurate"], index=1)
            st.session_state.config = {'model_type': model_type}
        
        elif selected_tool == 'OCR.Space':
            api_key = st.text_input("Chave API", type="password")
            language = st.selectbox("Idioma", ["por", "eng"])
            st.session_state.config = {'api_key': api_key, 'language': language}
        
        elif selected_tool == 'Taggun':
            api_key = st.text_input("Chave API", type="password")
            language = st.selectbox("Idioma", ["por", "eng"])
            st.session_state.config = {'api_key': api_key, 'language': language}
    
    if st.button("✅ Salvar Configurações"):
        st.session_state.ocr_tool = selected_tool
        st.session_state.page = 'process'
        st.rerun()

# ===========================================
# PÁGINA DE PROCESSAMENTO
# ===========================================
def process_file():
    """Interface de processamento de documentos"""
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
            
            # Converter para imagens
            if uploaded_file.type == "application/pdf":
                images = convert_from_bytes(file_bytes, dpi=300, poppler_path="/usr/bin")
                st.info(f"PDF convertido em {len(images)} página(s)")
            else:
                images = [Image.open(io.BytesIO(file_bytes))]
            
            text = ""
            tool = st.session_state.ocr_tool
            config = st.session_state.config
            
            # Processar OCR
            if tool == 'Tesseract':
                custom_config = f"--psm {config['psm']} --oem {config['oem']} -l {config['lang']}"
                text = "\n".join([pytesseract.image_to_string(preprocess_image(np.array(img)), config=custom_config) for img in images])
            
            elif tool == 'EasyOCR':
                reader = load_easyocr(config['langs'])
                text = "\n".join(["\n".join([res[1] for res in reader.readtext(np.array(img))]) for img in images])
            
            elif tool == 'DocTR':
                predictor = ocr_predictor(
                    det_arch='db_resnet50' if config['model_type'] == 'accurate' else 'db_mobilenet_v3_large',
                    reco_arch='crnn_vgg16_bn' if config['model_type'] == 'accurate' else 'crnn_mobilenet_v3_small',
                    pretrained=True
                )
                text = "\n".join(["\n".join([" ".join([w.value for w in line.words]) for block in predictor([np.array(img)]).pages[0].blocks for line in block.lines]) for img in images])
            
            elif tool == 'OCR.Space':
                text = "\n".join([handle_ocrspace_api(img, config) for img in images])
            
            elif tool == 'Taggun':
                img_byte_arr = io.BytesIO()
                images[0].save(img_byte_arr, format='PNG')
                response = requests.post(
                    'https://api.taggun.io/api/receipt/v1/simple/file',
                    headers={'apikey': config['api_key']},
                    files={'file': img_byte_arr.getvalue()},
                    data={'language': config['language']}
                )
                response.raise_for_status()
                text = response.json().get('text', '')
            
            texto_corrigido = correct_text_format(text)
            
            if validate_extracted_text(texto_corrigido):
                st.success("✅ Validação bem-sucedida!")
            else:
                st.warning("⚠️ Possíveis problemas na extração")
            
            st.download_button("💾 Baixar Texto", texto_corrigido, file_name=f"texto_{tool}.txt")
            st.text_area("📝 Texto Extraído", texto_corrigido, height=400)
        
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
            logger.error(traceback.format_exc())
    
    if st.button("↩️ Voltar para Configurações"):
        st.session_state.page = 'config'
        st.rerun()

# ===========================================
# CONTROLE PRINCIPAL
# ===========================================
if st.session_state.page == 'config':
    show_config_page()
else:
    process_file()
