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
import io
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
if 'ocr_tool' not in st.session_state:
    st.session_state.ocr_tool = 'Tesseract'
if 'config' not in st.session_state:
    st.session_state.config = {}
if 'page' not in st.session_state:
    st.session_state.page = 'config'

def preprocess_image(image, binarization_threshold=31, denoise_strength=10):
    """Melhora a qualidade da imagem para OCR."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=denoise_strength, templateWindowSize=7, searchWindowSize=21)
        thresh = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY, binarization_threshold, 2)
        return thresh
    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        st.error(f"Erro no pré-processamento: {e}")
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

def ocr_processor(images, tool, config):
    """Processa imagens com a ferramenta OCR selecionada."""
    try:
        if tool == 'Tesseract':
            texts = []
            for img in images:
                processed = preprocess_image(np.array(img), config['binarization_threshold'], config['denoise_strength'])
                custom_config = f"--oem {config['oem']} --psm {config['psm']} -l por+eng"
                texts.append(pytesseract.image_to_string(processed, config=custom_config))
            return "\n".join(texts)
        
        elif tool == 'EasyOCR':
            reader = load_easyocr()
            texts = []
            for img in images:
                results = reader.readtext(np.array(img))
                texts.append("\n".join([res[1] for res in results]))
            return "\n".join(texts)
        
        elif tool == 'DocTR':
            predictor = load_doctr(config['model_type'])
            texts = []
            for img in images:
                doc = predictor([np.array(img)])
                text = "\n".join([" ".join([w.value for w in line.words]) 
                                for block in doc.pages[0].blocks 
                                for line in block.lines])
                texts.append(text)
            return "\n".join(texts)
        
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
        
        else:
            raise ValueError("Ferramenta OCR não suportada")
    
    except Exception as e:
        logger.error(f"Erro no OCR: {e}")
        st.error(f"Falha no processamento OCR: {e}")
        return ""

def config_page():
    """Página de configuração das ferramentas OCR."""
    st.title("🛠 Configuração do Sistema OCR")
    
    ferramentas = {
        'Tesseract': {'icon': '🔍', 'desc': 'Documentos impressos de alta qualidade'},
        'EasyOCR': {'icon': '🎨', 'desc': 'Imagens com fundo complexo'},
        'DocTR': {'icon': '🧠', 'desc': 'Layouts complexos com deep learning'},
        'OCR.Space': {'icon': '🌐', 'desc': 'API externa para múltiplos formatos'},
        'Taggun': {'icon': '🧾', 'desc': 'Documentos fiscais/recibos'}
    }
    
    cols = st.columns(5)
    for i, (tool, meta) in enumerate(ferramentas.items()):
        with cols[i]:
            if st.button(f"{meta['icon']} {tool}", use_container_width=True, 
                        help=meta['desc']):
                st.session_state.ocr_tool = tool
    
    st.divider()
    
    if st.session_state.ocr_tool in ['OCR.Space', 'Taggun']:
        st.session_state.config['api_key'] = st.text_input(
            "Chave API", 
            type="password",
            help="Obtenha em https://ocr.space/API ou https://taggun.io"
        )
    
    if st.session_state.ocr_tool == 'DocTR':
        st.session_state.config['model_type'] = st.selectbox(
            "Tipo de Modelo", 
            ["fast", "accurate"], 
            index=1
        )
    else:
        with st.expander("⚙️ Configurações Avançadas"):
            st.session_state.config.update({
                'binarization_threshold': st.slider("Limiar de Binarização", 10, 50, 31),
                'denoise_strength': st.slider("Remoção de Ruído", 5, 20, 10),
                'psm': st.slider("Modo Segmentação (PSM)", 3, 13, 6),
                'oem': st.slider("Motor OCR (OEM)", 1, 3, 3)
            })
    
    if st.button("✅ Salvar Configurações"):
        st.session_state.page = 'process'
        st.rerun()

def process_page():
    """Página de processamento de documentos."""
    st.title("📄 Processamento de Documentos")
    
    uploaded_file = st.file_uploader(
        "Carregue seu documento (PDF ou imagem)", 
        type=["pdf", "png", "jpg", "jpeg"]
    )
    
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            
            try:
                if uploaded_file.type == "application/pdf":
                    images = convert_from_bytes(tmp_file.read(), dpi=300)
                else:
                    images = [Image.open(tmp_file.name)]
                
                with st.spinner(f"Processando com {st.session_state.ocr_tool}..."):
                    texto_bruto = ocr_processor(images, st.session_state.ocr_tool, st.session_state.config)
                    texto_corrigido = correct_text_format(texto_bruto)
                
                col1, col2 = st.columns(2)
                with col1:
                    if validate_extracted_text(texto_corrigido):
                        st.success("✅ Validação bem-sucedida!")
                    else:
                        st.warning("⚠️ Possíveis problemas na extração")
                
                with col2:
                    st.download_button(
                        "💾 Baixar Texto",
                        texto_corrigido,
                        file_name=f"texto_extraido_{st.session_state.ocr_tool}.txt"
                    )
                
                st.text_area("📝 Texto Extraído", texto_corrigido, height=400)
            
            except Exception as e:
                st.error(f"❌ Erro crítico: {str(e)}")
            finally:
                os.unlink(tmp_file.name)
    
    if st.button("↩️ Voltar para Configurações"):
        st.session_state.page = 'config'
        st.rerun()

# Controle de navegação
if st.session_state.page == 'config':
    config_page()
else:
    process_page()
