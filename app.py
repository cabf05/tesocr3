import streamlit as st
import easyocr
import requests
from PIL import Image
import io
import pytesseract
from doctr.io import DocumentFile
from doctr.models import ocr_predictor

# Funções de extração de texto
def extract_with_easyocr(image, lang=['pt', 'en'], paragraph=True):
    """Extrai texto de uma imagem usando EasyOCR."""
    reader = easyocr.Reader(lang)
    result = reader.readtext(image, paragraph=paragraph)
    return "\n".join([text for _, text, _ in result])

def extract_with_ocr_space(image, api_key, lang='por'):
    """Extrai texto usando a API OCR.Space."""
    url = "https://api.ocr.space/parse/image"
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    payload = {"apikey": api_key, "language": lang, "isOverlayRequired": False}
    files = {"file": ("image.png", img_buffer.getvalue(), "image/png")}
    response = requests.post(url, files=files, data=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("ParsedResults", [{}])[0].get("ParsedText", "Erro na extração")
    return f"Erro na API OCR.Space: {response.status_code}"

def extract_with_taggun(image, api_key, lang='auto'):
    """Extrai texto usando a API Taggun, otimizada para recibos."""
    url = "https://api.taggun.io/api/receipt/v1/verbose/file"
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    headers = {"apikey": api_key}
    files = {"file": ("image.png", img_buffer.getvalue(), "image/png")}
    payload = {"language": lang}
    response = requests.post(url, headers=headers, files=files, data=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("text", {}).get("value", "Erro na extração")
    return f"Erro na API Taggun: {response.status_code}"

def extract_with_free_ocr(image, lang='por'):
    """Extrai texto usando a Free OCR API."""
    url = "https://free.ocr.space/OCRAPI"
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    payload = {"language": lang, "isOverlayRequired": False}
    files = {"filename": ("image.png", img_buffer.getvalue(), "image/png")}
    response = requests.post(url, files=files, data=payload)
    if response.status_code == 200:
        result = response.json()
        return result.get("ParsedResults", [{}])[0].get("ParsedText", "Erro na extração")
    return f"Erro na Free OCR API: {response.status_code}"

def extract_with_pytesseract(image, lang='por+eng', psm=3):
    """Extrai texto usando Pytesseract (Tesseract OCR)."""
    try:
        text = pytesseract.image_to_string(image, lang=lang, config=f'--psm {psm}')
        return text
    except Exception as e:
        return f"Erro ao usar Pytesseract: {str(e)}"

def extract_with_doctr(image, lang='pt'):
    """Extrai texto usando DocTR para documentos estruturados."""
    try:
        img_buffer = io.BytesIO()
        image.save(img_buffer, format="PNG")
        img_buffer.seek(0)
        doc = DocumentFile.from_images(img_buffer)
        predictor = ocr_predictor(pretrained=True)
        result = predictor(doc)
        text = "\n".join([word.value for page in result.pages for block in page.blocks for line in block.lines for word in line.words])
        return text
    except Exception as e:
        return f"Erro ao usar DocTR: {str(e)}"

# Configuração inicial da sessão
if "tool" not in st.session_state:
    st.session_state.tool = None
    st.session_state.api_keys = {"OCR.Space": "", "Taggun": ""}
    st.session_state.params = {}

# Navegação entre páginas
st.sidebar.title("Navegação")
page = st.sidebar.radio("Escolha a página", ["Configuração", "Uso"])

if page == "Configuração":
    st.title("Configuração da Ferramenta OCR")
    st.write("Escolha e configure a ferramenta de OCR que deseja usar.")

    # Seleção da ferramenta com ajuda
    tool = st.selectbox(
        "Escolha a ferramenta OCR",
        ["EasyOCR", "OCR.Space", "Taggun", "Free OCR API", "Pytesseract", "DocTR"],
        help="EasyOCR: Ideal para textos simples. OCR.Space: Boa para textos curtos. Taggun: Ótima para recibos. Free OCR API: Simples e gratuita. Pytesseract: Usa Tesseract OCR. DocTR: Para documentos estruturados."
    )

    # Configuração por ferramenta
    if tool == "EasyOCR":
        lang_options = st.multiselect(
            "Idiomas", 
            ["pt", "en"], 
            default=["pt", "en"], 
            help="Escolha os idiomas (Português: 'pt', Inglês: 'en'). Padrão: ambos."
        )
        paragraph = st.checkbox(
            "Agrupar em parágrafos", 
            value=True, 
            help="Melhora a formatação do texto extraído. Padrão: ativado."
        )
        st.session_state.params = {"lang": lang_options, "paragraph": paragraph}

    elif tool == "OCR.Space":
        api_key = st.text_input(
            "Chave da API OCR.Space", 
            value=st.session_state.api_keys["OCR.Space"], 
            type="password", 
            help="Obtenha em ocr.space. Obrigatória para mais de 25 requisições/dia."
        )
        lang = st.selectbox(
            "Idioma", 
            ["por", "eng"], 
            index=0, 
            help="Português ('por') ou Inglês ('eng'). Padrão: Português."
        )
        st.session_state.api_keys["OCR.Space"] = api_key
        st.session_state.params = {"lang": lang}

    elif tool == "Taggun":
        api_key = st.text_input(
            "Chave da API Taggun", 
            value=st.session_state.api_keys["Taggun"], 
            type="password", 
            help="Obtenha em taggun.io. Obrigatória para uso."
        )
        lang = st.selectbox(
            "Idioma", 
            ["auto", "pt", "en"], 
            index=0, 
            help="Automático ('auto') detecta o idioma, ou escolha Português ('pt') ou Inglês ('en'). Padrão: Automático."
        )
        st.session_state.api_keys["Taggun"] = api_key
        st.session_state.params = {"lang": lang}

    elif tool == "Free OCR API":
        lang = st.selectbox(
            "Idioma", 
            ["por", "eng"], 
            index=0, 
            help="Português ('por') ou Inglês ('eng'). Padrão: Português."
        )
        st.session_state.params = {"lang": lang}

    elif tool == "Pytesseract":
        lang = st.selectbox(
            "Idioma", 
            ["por+eng", "por", "eng"], 
            index=0, 
            help="Padrão: português + inglês ('por+eng')."
        )
        psm = st.slider(
            "PSM (Page Segmentation Mode)", 
            1, 13, 3, 
            help="3 é o padrão para segmentação automática."
        )
        st.session_state.params = {"lang": lang, "psm": psm}

    elif tool == "DocTR":
        lang = st.selectbox(
            "Idioma", 
            ["pt", "en"], 
            index=0, 
            help="Padrão: português ('pt')."
        )
        st.session_state.params = {"lang": lang}

    # Salvar configurações
    if st.button("Salvar Configuração"):
        st.session_state.tool = tool
        st.success(f"Ferramenta {tool} configurada com sucesso! Vá para a página 'Uso'.")

elif page == "Uso":
    st.title("Extrair Texto de Imagem")
    
    if not st.session_state.tool:
        st.error("Configure uma ferramenta na página 'Configuração' primeiro!")
    else:
        st.write(f"Ferramenta selecionada: {st.session_state.tool}")
        
        # Upload do arquivo
        uploaded_file = st.file_uploader("Faça upload de uma imagem", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Imagem carregada", use_column_width=True)
            
            if st.button("Extrair Texto"):
                with st.spinner("Extraindo texto..."):
                    try:
                        if st.session_state.tool == "EasyOCR":
                            text = extract_with_easyocr(image, **st.session_state.params)
                        elif st.session_state.tool == "OCR.Space":
                            text = extract_with_ocr_space(image, st.session_state.api_keys["OCR.Space"], **st.session_state.params)
                        elif st.session_state.tool == "Taggun":
                            if not st.session_state.api_keys["Taggun"]:
                                text = "Forneça uma chave de API válida para Taggun."
                            else:
                                text = extract_with_taggun(image, st.session_state.api_keys["Taggun"], **st.session_state.params)
                        elif st.session_state.tool == "Free OCR API":
                            text = extract_with_free_ocr(image, **st.session_state.params)
                        elif st.session_state.tool == "Pytesseract":
                            text = extract_with_pytesseract(image, **st.session_state.params)
                        elif st.session_state.tool == "DocTR":
                            text = extract_with_doctr(image, **st.session_state.params)
                        st.subheader("Texto Extraído")
                        st.text_area("Resultado", text, height=300)
                    except Exception as e:
                        st.error(f"Erro ao extrair texto: {str(e)}")
