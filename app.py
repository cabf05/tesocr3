import streamlit as st
from ollama import generate  # Importa Ollama-OCR
from pdf2image import convert_from_bytes
from PIL import Image
import io

# Título da aplicação
st.title("OCR com Ollama no Streamlit Cloud")

# Upload do PDF
uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file:
    st.write("Arquivo recebido:", uploaded_file.name)

    # Converter PDF para imagens
    images = convert_from_bytes(uploaded_file.read())

    # Processar cada página do PDF
    extracted_text = ""
    for i, img in enumerate(images):
        st.image(img, caption=f"Página {i+1}", use_column_width=True)

        # Converte a imagem para bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format="PNG")
        img_bytes = img_bytes.getvalue()

        # Enviar para Ollama-OCR
        response = generate("ocr", img_bytes)  # Usa o modelo OCR do Ollama
        extracted_text += f"\n\nPágina {i+1}:\n{response}"

    # Exibir texto extraído
    st.subheader("Texto extraído:")
    st.text_area("Resultado OCR", extracted_text, height=300)
