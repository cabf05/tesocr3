import streamlit as st
from ollama import generate  # Importa Ollama-OCR
from pdf2image import convert_from_bytes
from PIL import Image
import io
import os

# Corrigir erro de Poppler no ambiente local (caso rode fora do Streamlit Cloud)
POPPLER_PATH = "/usr/bin/poppler"  # Caminho padrão no Linux
if not os.path.exists(POPPLER_PATH):
    POPPLER_PATH = None  # Streamlit Cloud já terá o pacote instalado via packages.txt

# Título da aplicação
st.title("OCR com Ollama no Streamlit Cloud")

# Upload do PDF
uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file:
    st.write("Arquivo recebido:", uploaded_file.name)

    try:
        # Converter PDF para imagens (agora com Poppler instalado corretamente)
        images = convert_from_bytes(uploaded_file.read(), poppler_path=POPPLER_PATH)

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

    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
