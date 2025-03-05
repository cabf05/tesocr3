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
import os
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

# Página de configuração
def config_page():
    st.title("Configuração do Sistema OCR")
    st.markdown("### Selecione e configure sua ferramenta OCR")

    ocr_tool = st.selectbox(
        "Ferramenta OCR:",
        ["Tesseract OCR", "EasyOCR", "OCR.Space", "DocTR", "Taggun", "Free OCR API"],
        index=0,
        help="Selecione a ferramenta de OCR desejada"
    )

    # Configurações específicas para cada ferramenta
    if ocr_tool == "Tesseract OCR":
        st.markdown("**Melhor para:** Documentos impressos de alta qualidade com layout simples")
        col1, col2 = st.columns(2)
        with col1:
            languages = st.text_input("Idiomas (ex: por+eng)", "por+eng")
        with col2:
            psm = st.selectbox(
                "Modo de Segmentação",
                options=[3, 6, 11, 12],
                index=0,
                format_func=lambda x: f"PSM {x} - {'Auto' if x == 3 else 'Texto denso' if x == 6 else 'Sparse text' if x == 11 else 'Sparse text com OS'}"
            )
        st.session_state.config = {
            "languages": languages,
            "psm": psm,
            "config": f"--psm {psm} -l {languages}"
        }

    elif ocr_tool == "EasyOCR":
        st.markdown("**Melhor para:** Imagens com fundo complexo e textos em ângulos variados")
        langs = st.multiselect(
            "Idiomas",
            ["en", "pt"],
            default=["en", "pt"]
        )
        st.session_state.config = {
            "langs": langs,
            "gpu": False
        }

    elif ocr_tool == "OCR.Space":
        st.markdown("**Melhor para:** Uso via API com suporte a múltiplos formatos")
        api_key = st.text_input("API Key", type="password")
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Idioma", ["por", "eng"])
        with col2:
            engine = st.selectbox("Motor OCR", [1, 2, 3], help="1 - Normal, 2 - Melhor velocidade, 3 - Melhor precisão")
        st.session_state.config = {
            "api_key": api_key,
            "language": language,
            "engine": engine
        }

    elif ocr_tool == "DocTR":
        st.markdown("**Melhor para:** Documentos modernos com layout complexo usando deep learning")
        doc_lang = st.selectbox("Idioma", ["en", "pt"])
        st.session_state.config = {
            "doc_lang": doc_lang,
            "model_type": st.selectbox("Tipo de Modelo", ["fast", "accurate"], index=1)
        }

    elif ocr_tool == "Taggun":
        st.markdown("**Melhor para:** Documentos fiscais e recibos com estrutura complexa")
        api_key = st.text_input("API Key Taggun", type="password")
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Idioma", ["por", "eng"])
        with col2:
            is_receipt = st.checkbox("É um recibo?", value=True)
        st.session_state.config = {
            "api_key": api_key,
            "language": language,
            "is_receipt": is_receipt
        }

    elif ocr_tool == "Free OCR API":
        st.markdown("**Melhor para:** Uso rápido sem necessidade de chave API (limite de 250 requests/hora)")
        col1, col2 = st.columns(2)
        with col1:
            language = st.selectbox("Idioma", ["por", "eng"])
        with col2:
            overlay = st.checkbox("Mostrar overlay do texto", value=False)
        st.session_state.config = {
            "language": language,
            "overlay": overlay
        }

    if st.button("Salvar Configuração"):
        st.session_state.ocr_tool = ocr_tool
        st.session_state.page = 'process'
        st.experimental_rerun()

# Página de processamento
def process_page():
    st.title("Extrair Texto de Documentos")
    st.markdown(f"**Ferramenta selecionada:** {st.session_state.ocr_tool}")

    uploaded_file = st.file_uploader("Carregue seu documento (PDF ou imagem)", type=["png", "jpg", "jpeg", "pdf"])

    if uploaded_file is not None:
        if st.button("Processar Documento"):
            with st.spinner("Processando..."):
                try:
                    if uploaded_file.type == "application/pdf":
                        images = convert_from_bytes(uploaded_file.read())
                        st.info(f"PDF convertido em {len(images)} página(s)")
                    else:
                        images = [Image.open(uploaded_file)]

                    full_text = ""

                    # Tesseract OCR
                    if st.session_state.ocr_tool == "Tesseract OCR":
                        for img in images:
                            text = pytesseract.image_to_string(img, config=st.session_state.config['config'])
                            full_text += text + "\n\n"

                    # EasyOCR
                    elif st.session_state.ocr_tool == "EasyOCR":
                        reader = easyocr.Reader(st.session_state.config['langs'], gpu=False)
                        for img in images:
                            results = reader.readtext(np.array(img))
                            text = "\n".join([res[1] for res in results])
                            full_text += text + "\n\n"

                    # OCR.Space
                    elif st.session_state.ocr_tool == "OCR.Space":
                        if not st.session_state.config.get('api_key'):
                            st.error("API Key é obrigatória para OCR.Space!")
                            return

                        for img in images:
                            img_bytes = io.BytesIO()
                            img.save(img_bytes, format='PNG')
                            response = requests.post(
                                'https://api.ocr.space/parse/image',
                                files={'file': img_bytes.getvalue()},
                                data={
                                    'apikey': st.session_state.config['api_key'],
                                    'language': st.session_state.config['language'],
                                    'OCREngine': st.session_state.config['engine'],
                                    'isOverlayRequired': False
                                }
                            )
                            result = response.json()
                            if response.status_code == 200 and not result['IsErroredOnProcessing']:
                                full_text += result['ParsedResults'][0]['ParsedText'] + "\n\n"
                            else:
                                st.error(f"Erro na API: {result.get('ErrorMessage', 'Erro desconhecido')}")
                                return

                    # DocTR
                    elif st.session_state.ocr_tool == "DocTR":
                        predictor = ocr_predictor(
                            det_arch='db_resnet50' if st.session_state.config['model_type'] == 'accurate' else 'db_mobilenet_v3_large',
                            reco_arch='crnn_vgg16_bn' if st.session_state.config['model_type'] == 'accurate' else 'crnn_mobilenet_v3_small',
                            pretrained=True
                        )
                        for img in images:
                            doc = predictor([np.array(img)])
                            page_text = ""
                            for block in doc.pages[0].blocks:
                                for line in block.lines:
                                    page_text += " ".join([word.value for word in line.words]) + "\n"
                            full_text += page_text + "\n\n"

                    # Taggun
                    elif st.session_state.ocr_tool == "Taggun":
                        if not st.session_state.config.get('api_key'):
                            st.error("API Key é obrigatória para Taggun!")
                            return

                        img_bytes = io.BytesIO()
                        images[0].save(img_bytes, format='PNG')
                        base64_image = base64.b64encode(img_bytes.getvalue()).decode()

                        response = requests.post(
                            'https://api.taggun.io/api/receipt/v1/simple/file',
                            headers={'apikey': st.session_state.config['api_key']},
                            json={
                                'image': base64_image,
                                'language': st.session_state.config['language'],
                                'isReceipt': st.session_state.config['is_receipt']
                            }
                        )

                        if response.status_code == 200:
                            result = response.json()
                            full_text = result['text']
                        else:
                            st.error(f"Erro na API: {response.text}")
                            return

                    # Free OCR API
                    elif st.session_state.ocr_tool == "Free OCR API":
                        img_bytes = io.BytesIO()
                        images[0].save(img_bytes, format='PNG')
                        img_bytes.seek(0)

                        response = requests.post(
                            'https://api.ocr.space/parse/image',
                            files={'file': img_bytes},
                            data={
                                'apikey': 'helloworld',  # Chave pública gratuita
                                'language': st.session_state.config['language'],
                                'isOverlayRequired': st.session_state.config['overlay']
                            }
                        )

                        result = response.json()
                        if response.status_code == 200 and not result['IsErroredOnProcessing']:
                            full_text = result['ParsedResults'][0]['ParsedText']
                        else:
                            st.error(f"Erro na API: {result.get('ErrorMessage', 'Erro desconhecido')}")
                            return

                    st.success("Texto extraído com sucesso!")
                    st.text_area("Resultado", full_text, height=400)

                except Exception as e:
                    st.error(f"Erro no processamento: {str(e)}")

    if st.button("Voltar para Configurações"):
        st.session_state.page = 'config'
        st.experimental_rerun()

# Controle de navegação
if st.session_state.page == 'config':
    config_page()
else:
    process_page()
