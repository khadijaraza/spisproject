# app.py
import streamlit as st
from style_utils import set_page_theme

st.set_page_config(page_title="Exoplanet AI", page_icon="🌌", layout="wide")

# At the top of app.py, pages/1_🚀_Demo.py, etc.

from style_utils import set_page_theme

# A reliable, working GIF for your astral theme
image_address = "https://i.pinimg.com/originals/97/8c/f7/978cf71ad42ee5695cfd4fc5d7d07a68.gif"

set_page_theme(image_address)

# ... (the rest of your page code) ...

st.title("Classification of TESS Light Curves")


# In your app.py, after the section describing your CNN project:

st.markdown("---")
st.header("A Complementary Approach: Feature Engineering with a Random Forest Classifier")
st.write("""
This project explores a powerful alternative to the CNN model, leveraging a classical machine learning approach to classify celestial objects. Instead of relying on the raw visual pattern of the light curve, this method uses a **Random Forest** model trained on a set of descriptive, hand-engineered features extracted from the time-series data.
""")

st.subheader("The Dataset: A Simulated Universe")
st.write("""
The model was trained using the **PLAsTiCC 2018 Kaggle competition dataset**, a simulated dataset designed to mimic the cadence and noise properties of real astronomical surveys. This dataset is particularly challenging as it contains 15 different classes of celestial objects, many of which are rare, introducing the real-world problem of **class imbalance**. For this project, the task was refined to focus on a subset of these classes, primarily distinguishing between different types of stellar variability.
""")

st.subheader("Methodology: From Time-Series to Tabular Features")
st.write("""
While a CNN learns features automatically, a Random Forest's power depends on the quality of its input features. This project's core is a sophisticated **feature engineering** pipeline that transforms each light curve from a sequence of points into a rich set of statistical descriptors.
""")
st.markdown("""
By analyzing the properties of the time-series data, we can provide the model with crucial information, such as:
- **Statistical Moments:** Mean, median, standard deviation, skewness, and kurtosis of the brightness measurements.
- **Periodicity Features:** The dominant period of variation, amplitude, and significance of the signal.
- **Flux Ratios:** Ratios of brightness measurements across different photometric bands (colors).

This process converts the complex time-series problem into a structured, tabular classification task, which is ideal for a Random Forest. The model—an ensemble of decision trees—is then trained on these features. A key advantage of this approach is its **interpretability**; after training, we can directly inspect the model to see which specific features, like `amplitude` or `period`, were most important for its classification decisions.
""")

st.markdown("---")

st.markdown("### An End-to-End Data Pipeline for Exoplanet Candidate Vetting and CNN Classifiers")

st.subheader("Abstract")
st.write("""
The Transiting Exoplanet Survey Satellite (TESS) produces a vast stream of time-series photometric data, presenting a significant data analysis challenge. Within this dataset, signals indicative of exoplanets must be distinguished from astrophysical false positives, such as eclipsing binary stars. This project implements a complete pipeline to address this challenge, leveraging a Convolutional Neural Network (CNN) to perform automated classification based on the morphological features of light curve signals. We demonstrate a robust workflow encompassing data ingestion, parallelized preprocessing, and a deep learning model for candidate vetting.
""")

st.subheader("System Architecture & Methodology")
st.markdown("""
The project is structured as a scalable pipeline designed to process raw target lists into a model-ready dataset. Key stages include:

1.  **Data Ingestion & Validation:** Target lists are sourced from the NASA Exoplanet Archive and TESS Objects of Interest (TOI) catalogs. A robust validation routine handles "poison pill" scenarios—where single invalid Target IDs cause batch query failures—by checking each target individually before initiating large, efficient batch downloads.

2.  **Preprocessing:** A custom function finds the most probable period by analyzing each observation sector independently. The data is then stitched, flattened to remove stellar variability, and folded. The final light curve is binned to a uniform length to create a consistent input vector for the neural network.

3.  **Classification:** A 1D Convolutional Neural Network, built in TensorFlow/Keras, is trained on the preprocessed data to distinguish between the characteristically U-shaped transits of exoplanets and the V-shaped eclipses of stellar false positives.

Navigate to the **🚀 Demo** page to see a conceptual demonstration of the classifier.
""")