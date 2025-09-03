# pages/2_📚_Authorship.py
import streamlit as st

st.title("📚 Authorship and Sources")
st.markdown("---")

st.header("Authorship")
st.write("This project was created by **[Your Name Here]** with the assistance of Google's Gemini.")

st.header("Data Sources")
st.markdown("""
- **Light Curve Data:** Sourced from the [Mikulski Archive for Space Telescopes (MAST)](https://mast.stsci.edu/portal/Mashup/Clients/Mast/Portal.html), primarily from the TESS mission.
- **Exoplanet Catalogs:** Ground-truth labels were retrieved from the [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu/).
""")

st.header("Helpful Materials & Libraries")
st.markdown("""
- **`lightkurve`:** A Python package for Kepler and TESS data analysis. [Documentation](https://docs.lightkurve.org/)
- **`astroquery`:** A package for querying astronomical web services. [Documentation](https://astroquery.readthedocs.io/)
- **`streamlit`:** The framework used to build this web application. [Documentation](https://docs.streamlit.io/)
- **`tensorflow` / `keras`:** The deep learning framework used for the CNN model. [Documentation](https://www.tensorflow.org/)
""")