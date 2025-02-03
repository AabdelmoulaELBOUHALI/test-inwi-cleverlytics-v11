import base64
import streamlit as st
from pathlib import Path

def get_base64_of_bin_file(bin_file_path: str) -> str:
    file_bytes = Path(bin_file_path).read_bytes()
    return base64.b64encode(file_bytes).decode()

def main():
    st.set_page_config(page_title="INWI Chatbot - Accueil", layout="wide")

    # Read local image and convert to Base64
    img_base64 = get_base64_of_bin_file("./img/logo inwi celeverlytics.png")
    css_logo = f"""
    <style>
    [data-testid="stSidebarNav"]::before {{
        content: "";
        display: block;
        margin: 0 auto 20px auto;
        width: 80%;
        height: 100px;
        background-image: url("data:image/png;base64,{img_base64}");
        background-size: contain;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """

    st.markdown(css_logo, unsafe_allow_html=True)

    st.title("👋 Bienvenue sur le Chatbot INWI")
    st.markdown(
        """
        Ceci est la page principale.  
        Vous pouvez choisir le **Chatbot en Français** ou le **Chatbot en Arabe** en naviguant dans le menu de gauche (sous "Pages" ou "Select a page").
        """
    )
    st.write("Veuillez sélectionner la langue désirée dans la barre latérale.")

if __name__ == "__main__":
    main()
