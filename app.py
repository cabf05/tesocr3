import streamlit as st
import easyocr
import fitz  # pymupdf
import numpy as np
from PIL import Image

# Inicializa o leitor OCR
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['pt', 'en'])

reader = load_easyocr()

st.title("📄 OCR de PDFs com EasyOCR")

uploaded_file = st.file_uploader("Faça upload de um arquivo PDF", type=["pdf"])

if uploaded_file:
    st.write("📤 Processando o arquivo...")

    # Abre o PDF com pymupdf
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    
    extracted_text = ""

    for i, page in enumerate(doc):
        # Converte a página do PDF em imagem
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

        st.image(img, caption=f"Página {i+1}", use_column_width=True)

        # Converte para numpy array para o OCR
        img_array = np.array(img)

        # Executa OCR na imagem
        result = reader.readtext(img_array, detail=0)

        extracted_text += f"\n\n📜 **Página {i+1}:**\n" + "\n".join(result)

    # Exibe o texto extraído
    st.subheader("📑 Texto extraído:")
    st.text_area("Texto OCR", extracted_text, height=300)

    # Baixar texto extraído
    st.download_button("📥 Baixar texto extraído", extracted_text, file_name="texto_extraido.txt")
