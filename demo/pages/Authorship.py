# pages/2_📚_Authorship.py
import streamlit as st
from style_utils import set_page_theme

st.set_page_config(page_title="Authorship", page_icon="📚", layout="wide")
set_page_theme("https://i.pinimg.com/originals/97/8c/f7/978cf71ad42ee5695cfd4fc5d7d07a68.gif")

st.title("📚 Authorship & Sources")
st.markdown("---")

st.header("Authorship")
st.write("This project was created by **[Your Name Here]** with the assistance of Google's Gemini.")

st.header("Data Sources")
st.markdown("""
- **Light Curve Data:** Sourced from the [Mikulski Archive for Space Telescopes (MAST)](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html), from the TESS mission (SPOC pipeline).
- **Exoplanet Catalogs:** Ground-truth labels were retrieved from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
""")

st.header("Key Technologies")
st.markdown("""
- **`lightkurve` & `astroquery`:** For astronomical data acquisition and analysis.
- **`streamlit`:** For building this interactive web application.
- **`tensorflow` & `keras`:** For the deep learning model.
- **`numpy` & `pandas`:** For data manipulation.
""")