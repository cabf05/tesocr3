import streamlit as st
import easyocr
import pdf2image
import numpy as np
from PIL import Image

# Inicializa o leitor OCR do EasyOCR
@st.cache_resource
def load_easyocr():
    return easyocr.Reader(['pt', 'en'])  # Suporte para português e inglês

reader = load_easyocr()

st.title("📄 OCR de PDFs com EasyOCR")

uploaded_file = st.file_uploader("Faça upload de um arquivo PDF", type=["pdf"])

if uploaded_file:
    st.write("📤 Processando o arquivo...")

    # Converte o PDF para imagens (uma imagem por página)
    images = pdf2image.convert_from_bytes(uploaded_file.read())

    st.write(f"📄 O PDF tem {len(images)} páginas.")

    extracted_text = ""
    
    for i, image in enumerate(images):
        st.image(image, caption=f"Página {i+1}", use_column_width=True)

        # Converte a imagem para numpy array (necessário para o EasyOCR)
        img_array = np.array(image)

        # Executa o OCR
        result = reader.readtext(img_array, detail=0)  # `detail=0` retorna apenas o texto

        # Junta os textos extraídos
        extracted_text += f"\n\n📜 **Página {i+1}:**\n" + "\n".join(result)

    # Exibe o texto extraído
    st.subheader("📑 Texto extraído:")
    st.text_area("Texto OCR", extracted_text, height=300)

    # Permite baixar o texto extraído
    st.download_button("📥 Baixar texto extraído", extracted_text, file_name="texto_extraido.txt")
