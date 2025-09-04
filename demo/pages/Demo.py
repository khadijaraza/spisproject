# pages/1_🚀_Demo.py
import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import pickle
from scipy.stats import skew, kurtosis
import os

# (Your background-setting code/imports go here, e.g., from style_utils)

st.title("🚀 Live Model Demo")
st.write("Choose a real celestial object. The app will download its TESS data, engineer features, and classify it with the trained Random Forest model.")
st.markdown("---")

# --- Load Model and Feature Engineering Functions ---
@st.cache_resource
def load_model_and_features():
    """Loads the pre-trained model and feature list."""
    try:
        model = joblib.load('random_forest_model.joblib')
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, feature_columns
    except FileNotFoundError:
        st.error("Model or feature file not found. Make sure 'random_forest_model.joblib' and 'feature_columns.pkl' are in your repository.")
        return None, None

@st.cache_data
def load_light_curve(tic_id):
    """Downloads and caches a light curve from MAST."""
    try:
        search = lk.search_lightcurve(target=tic_id, author="TESS-SPOC")
        if not search: 
            return "No TESS-SPOC data found."
        lc_collection = search.download_all()
        return lc_collection.stitch().remove_nans().remove_outliers()
    except Exception as e:
        return f"Could not download data: {e}"

def getFeatures(lc):
    """This is the same feature engineering function from your notebook."""
    flux = lc.flux.value
    time = lc.time.value
    flux_err = lc.flux_err.value
    
    # Your feature calculations (adapted from the notebook)
    peak_flux = np.max(flux)
    peak_time = time[np.argmax(flux)]
    half_max = peak_flux / 2
    above_half = time[flux >= half_max]
    fwhm = above_half.max() - above_half.min() if len(above_half) > 1 else 0
    flux_diff = np.diff(flux)
    vn_ratio = np.sum((flux_diff)**2) / ((len(flux)-1) * np.var(flux)) if len(flux) > 1 else 0

    features = {
      "mean_flux": np.mean(flux), "std_flux": np.std(flux),
      "amplitude": np.max(flux) - np.min(flux), "median_flux": np.median(flux),
      "flux_skew": skew(flux), "flux_kurtosis": kurtosis(flux),
      "flux_p25": np.percentile(flux, 25), "flux_p75": np.percentile(flux, 75),
      "flux_range_50": np.percentile(flux, 75) - np.percentile(flux, 25),
      "lc_duration": time.max() - time.min(),
      "rise_time": time[np.argmax(flux)] - time.min(),
      "decay_time": time.max() - time[np.argmax(flux)],
      "rise_decay_ratio": (peak_time - time.min()) / (time.max() - peak_time + 1e-5),
      "fwhm": fwhm, "vn_ratio": vn_ratio,
      "excess_var": np.var(flux) - np.mean(flux_err**2),
      "slope_mean": np.mean(np.diff(flux) / np.diff(time)),
      "slope_std": np.std(np.diff(flux) / np.diff(time)),
    }
    return features

# --- Streamlit UI ---
model, feature_columns = load_model_and_features()

EXAMPLES = {
    "Eclipsing Binary Star": "TIC 204214461",
    "Pulsating Variable Star": "TIC 259849649",
}

selection_name = st.selectbox("Choose an example to classify:", list(EXAMPLES.keys()))
selected_tic = EXAMPLES[selection_name]

if st.button(f"Classify {selection_name}"):
    if model and feature_columns:
        with st.spinner(f"Downloading and processing TIC {selected_tic}..."):
            lc = load_light_curve(selected_tic)
            
            if isinstance(lc, str):
                st.error(lc) # Show error message if download failed
            else:
                # 1. Feature Engineering
                features = getFeatures(lc)
                features_df = pd.DataFrame([features])
                
                # 2. Ensure columns are in the correct order
                features_df = features_df[feature_columns]

                # 3. Make Prediction
                prediction = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)
                
                # --- Display the Results ---
                st.subheader("Results")
                fig, ax = plt.subplots()
                lc.scatter(ax=ax)
                ax.set_title(f"Light Curve for TIC {selected_tic}")
                st.pyplot(fig)
                
                predicted_class = prediction[0]
                confidence = np.max(prediction_proba)
                
                st.success(f"**Model Prediction:** {predicted_class} (Confidence: {confidence:.1%})")