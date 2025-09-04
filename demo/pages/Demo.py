# pages/1_🚀_Demo.py
import streamlit as st
import lightkurve as lk
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import pickle
from scipy.stats import skew, kurtosis

# (Your background-setting code/imports go here, e.g., from style_utils)

st.title("🚀 Live Model Demo")
st.write("Choose a real celestial object from the PLAsTiCC dataset. The app will load its light curve, engineer features, and classify it with the trained Random Forest model.")
st.markdown("---")

# --- Load Model, Data, and Feature Engineering Functions ---

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
def load_full_dataset():
    """Loads and caches the full (large) training_set CSV file."""
    try:
        # Use the full dataset as _full_lc_df to avoid scope issues
        _full_lc_df = pd.read_csv("training_set.csv")
        return _full_lc_df
    except FileNotFoundError:
        return None

def get_light_curve_from_df(object_id, full_df):
    """Extracts a light curve for a single object from the main dataframe."""
    obj_df = full_df[full_df['object_id'] == object_id]
    if obj_df.empty:
        return f"No light curve data found for object_id {object_id}."
    return lk.LightCurve(time=obj_df['mjd'], flux=obj_df['flux'], flux_err=obj_df['flux_err'])

def getFeatures(lc):
    """This is the same feature engineering function from your notebook."""
    flux = lc.flux.value
    time = lc.time.value
    flux_err = lc.flux_err.value
    
    peak_flux = np.max(flux)
    peak_time = time[np.argmax(flux)] if len(time) > 0 else 0
    half_max = peak_flux / 2
    above_half = time[flux >= half_max]
    fwhm = above_half.max() - above_half.min() if len(above_half) > 1 else 0
    flux_diff = np.diff(flux)
    vn_ratio = np.sum((flux_diff)**2) / ((len(flux)-1) * np.var(flux)) if len(flux) > 1 and np.var(flux) != 0 else 0

    features = {
      "mean_flux": np.mean(flux), "std_flux": np.std(flux),
      "amplitude": np.max(flux) - np.min(flux), "median_flux": np.median(flux),
      "flux_skew": skew(flux), "flux_kurtosis": kurtosis(flux),
      "flux_p25": np.percentile(flux, 25), "flux_p75": np.percentile(flux, 75),
      "flux_range_50": np.percentile(flux, 75) - np.percentile(flux, 25),
      "lc_duration": time.max() - time.min() if len(time) > 0 else 0,
      "rise_time": peak_time - time.min() if len(time) > 0 else 0,
      "decay_time": time.max() - peak_time if len(time) > 0 else 0,
      "rise_decay_ratio": (peak_time - time.min()) / (time.max() - peak_time + 1e-5) if len(time) > 0 and (time.max() - peak_time) != 0 else 0,
      "fwhm": fwhm, "vn_ratio": vn_ratio,
      "excess_var": np.var(flux) - np.mean(flux_err**2),
      "slope_mean": np.mean(np.diff(flux) / np.diff(time)) if len(time) > 1 else 0,
      "slope_std": np.std(np.diff(flux) / np.diff(time)) if len(time) > 1 else 0,
    }
    return features

# --- Streamlit UI ---
model, feature_columns = load_model_and_features()
full_lc_df = load_full_dataset()

EXAMPLES = {
    "Black Hole Event (Class 16)": {"object_id": 713, "image": "https://heasarc.gsfc.nasa.gov/docs/tess/images/Target-Pixel-Files_files/Target-Pixel-Files_54_0.png"},
    "Supernova Event (Class 90)": {"object_id": 615, "image": "https://heasarc.gsfc.nasa.gov/docs/tess/images/Target-Pixel-Files_files/Target-Pixel-Files_54_0.png"},
    "Pulsating Star (RR Lyrae, Class 92)": {"object_id": 1124, "image": "https://heasarc.gsfc.nasa.gov/docs/tess/images/Target-Pixel-Files_files/Target-Pixel-Files_54_0.png"}
}

if model is None or full_lc_df is None:
    st.stop() # Don't run the rest of the app if files are missing

cols = st.columns(3)
col_names = list(EXAMPLES.keys())

for i, col in enumerate(cols):
    with col:
        name = col_names[i]
        example = EXAMPLES[name]
        obj_id = example["object_id"]
        
        st.header(name)
        st.image(example["image"], use_column_width=True)
        
        lc = get_light_curve_from_df(obj_id, full_lc_df)
        
        if isinstance(lc, lk.LightCurve):
            fig, ax = plt.subplots()
            lc.scatter(ax=ax, s=2)
            ax.set_title(f"Light Curve for Object {obj_id}")
            st.pyplot(fig)
        else:
            st.warning(lc)

        if st.button(f"Classify {name}", key=f"button_{i}"):
            if isinstance(lc, lk.LightCurve):
                with st.spinner("Engineering features and predicting..."):
                    # 1. Feature Engineering
                    features = getFeatures(lc)
                    features_df = pd.DataFrame([features])
                    
                    # 2. Ensure columns are in the correct order for the model
                    features_df = features_df[feature_columns]

                    # 3. Make Prediction
                    prediction = model.predict(features_df)
                    prediction_proba = model.predict_proba(features_df)
                    
                    predicted_class = prediction[0]
                    confidence = np.max(prediction_proba)
                    
                    st.success(f"**Model Prediction:** {predicted_class} (Confidence: {confidence:.1%})")