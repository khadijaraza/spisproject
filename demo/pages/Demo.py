# pages/1_🚀_Demo.py
import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import time

st.title("🚀 Interactive Demo")
st.write("Choose a pre-selected celestial object and see how the model classifies its light curve.")
st.markdown("---")

@st.cache_data
def load_example_data(tic_id):
    try:
        search = lk.search_lightcurve(target=tic_id, mission="TESS", author="TESS-SPOC")
        lc = search.download()
        return lc
    except Exception:
        st.error(f"Could not download data for {tic_id}.")
        return None

EXAMPLES = {
    "Clear Exoplanet (Pi Mensae c)": "TIC 261136679",
    "Eclipsing Binary Star (V-Shape)": "TIC 204214461",
    "Noisy Star": "TIC 321995146"
}

selection_name = st.selectbox("Choose an example to classify:", list(EXAMPLES.keys()))

if st.button(f"Analyze {selection_name}"):
    tic_id = EXAMPLES[selection_name]
    with st.spinner("Querying stellar database and running placeholder analysis..."):
        lc = load_example_data(tic_id)
        time.sleep(2)
        is_planet = "Planet" in selection_name
        score = np.random.uniform(0.6, 0.95) if is_planet else np.random.uniform(0.05, 0.4)
    
    st.subheader("Results")
    if lc:
        fig, ax = plt.subplots()
        lc.scatter(ax=ax, color='cyan', s=1)
        ax.set_title(f"Raw Light Curve for {tic_id}", color='#ADD8E6')
        ax.set_facecolor('black')
        fig.set_facecolor('black')
        # ... more plot styling ...
        st.pyplot(fig)

    if score > 0.5:
        st.success(f"**Prediction:** Planet Candidate (Confidence: {score:.1%})")
    else:
        st.info(f"**Prediction:** False Positive / Star (Confidence: {1-score:.1%})")
    st.progress(score)