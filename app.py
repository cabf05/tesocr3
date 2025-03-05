import streamlit as st
from ollama import generate  # Importa Ollama-OCR
from pdf2image import convert_from_bytes
from PIL import Image
import io
import base64

# Título da aplicação
st.title("OCR com Ollama no Streamlit Cloud")

# Upload do PDF
uploaded_file = st.file_uploader("Envie um arquivo PDF", type="pdf")

if uploaded_file:
    st.write("Arquivo recebido:", uploaded_file.name)

    try:
        # Converter PDF para imagens
        images = convert_from_bytes(uploaded_file.read())

        # Processar cada página do PDF
        extracted_text = ""
        for i, img in enumerate(images):
            st.image(img, caption=f"Página {i+1}", use_column_width=True)

            # Converter imagem para Base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")
            img_base64 = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            # Enviar para Ollama OCR
            response = generate(model="ocr", prompt=img_base64)  
            extracted_text += f"\n\nPágina {i+1}:\n{response}"

        # Exibir texto extraído
        st.subheader("Texto extraído:")
        st.text_area("Resultado OCR", extracted_text, height=300)

    except Exception as e:
        st.error(f"Erro ao processar o PDF: {e}")
