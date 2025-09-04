# style_utils.py
import streamlit as st

def set_page_theme(bg_url: str):
    """
    Sets the background of a Streamlit page to an image from a URL.
    """
    style = f"""
        <style>
        .stApp {{
            background-image: url("{bg_url}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        h1, h2, h3 {{
            color: #00FFFF;
            text-shadow: 0 0 5px #00FFFF, 0 0 10px #00FFFF, 0 0 20px #FF0077;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)