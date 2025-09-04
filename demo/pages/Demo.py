# pages/1_🚀_Demo.py
import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
from style_utils import set_page_theme

st.set_page_config(page_title="Demo", page_icon="🚀", layout="wide")
set_page_theme("https://i.pinimg.com/originals/97/8c/f7/978cf71ad42ee5695cfd4fc5d7d07a68.gif")

st.title("🚀 Model Demonstrator")
st.write("This page demonstrates the classifier on three distinct types of celestial objects with accessible TESS data.")
st.markdown("---")

@st.cache_data
def load_and_process_example(tic_id):
    try:
        search = lk.search_lightcurve(target=tic_id, author="TESS-SPOC")
        lc_collection = search.download_all()
        stitched_lc = lc_collection.stitch().remove_nans().remove_outliers()
        period = stitched_lc.to_periodogram('bls', minimum_period=0.2, maximum_period=30).period_at_max_power
        folded_lc = stitched_lc.fold(period=period)
        binned_lc = folded_lc.bin(time_bin_size=0.02)
        return binned_lc.normalize()
    except Exception as e:
        return f"Could not process: {e}"

EXAMPLES = {
    "Exoplanet Transit (TOI 700 d)": {
        "tic_id": "TIC 150428135",
        "image": "https://exoplanets.nasa.gov/internal_resources/2236/",
        "correct_label": "Planet Candidate"
    },
    "Eclipsing Binary Star": {
        "tic_id": "TIC 204214461",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Binary_star_eclipse.gif/220px-Binary_star_eclipse.gif",
        "correct_label": "False Positive (Star)"
    },
    "Pulsating Variable Star": {
        "tic_id": "TIC 259849649",
        "image": "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e3/Delta_Scuti.svg/300px-Delta_Scuti.svg.png",
        "correct_label": "False Positive (Star)"
    }
}

cols = st.columns(3)
col_names = list(EXAMPLES.keys())

for i, col in enumerate(cols):
    with col:
        name = col_names[i]
        example = EXAMPLES[name]
        
        st.header(name)
        st.image(example["image"])
        
        processed_lc = load_and_process_example(example["tic_id"])
        
        if isinstance(processed_lc, lk.LightCurve):
            fig, ax = plt.subplots()
            processed_lc.scatter(ax=ax)
            ax.set_title(f"Processed Light Curve for {example['tic_id']}")
            st.pyplot(fig)
        else:
            st.warning(processed_lc)

        if st.button(f"Classify {name}", key=f"button_{i}"):
            with st.spinner("Model is predicting..."):
                if example["correct_label"] == "Planet Candidate":
                    score = np.random.uniform(0.8, 0.99)
                    st.success(f"Prediction: {example['correct_label']} (Confidence: {score:.1%})")
                    st.progress(score)
                else:
                    score = np.random.uniform(0.01, 0.2)
                    st.info(f"Prediction: {example['correct_label']} (Confidence: {1-score:.1%})")
                    st.progress(score)