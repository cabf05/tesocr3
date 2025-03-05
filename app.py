import streamlit as st
import requests
import os

# Configuração da chave da API
st.set_page_config(page_title="OCR Space API", layout="wide")

st.title("Configuração da Chave OCR")
ocr_api_key = st.text_input("Insira sua chave da API OCR Space:", type="password")

if ocr_api_key:
    st.session_state["ocr_api_key"] = ocr_api_key
    st.success("Chave API salva com sucesso!")

# Página para processar arquivo com OCR
def process_file():
    st.title("Upload e Processamento de Arquivo")
    uploaded_file = st.file_uploader("Faça upload do arquivo", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file is not None:
        if "ocr_api_key" not in st.session_state:
            st.error("Por favor, insira sua chave API na página principal.")
            return
        
        api_key = st.session_state["ocr_api_key"]
        files = {"file": uploaded_file.getvalue()}
        payload = {"apikey": api_key, "language": "eng"}
        
        response = requests.post("https://api.ocr.space/parse/image", files=files, data=payload)
        result = response.json()
        
        if response.status_code == 200 and result.get("ParsedResults"):
            st.subheader("Texto Extraído:")
            st.write(result["ParsedResults"][0]["ParsedText"])
        else:
            st.error("Erro ao processar o arquivo. Verifique sua chave API e tente novamente.")

if __name__ == "__main__":
    process_file()
