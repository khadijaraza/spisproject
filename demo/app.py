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

st.title("Classification of Astronomical Transients with Machine Learning")
st.markdown("---")


st.header("Introduction")
st.write("""
Over the past two decades, the volume of astronomical data collected has grown far beyond our expectations. Telescopes and facilities such as the Large Synoptic Survey Telescope (LSST) are expected to record tens of millions of new variable and transient events every night. Machine Learning has become essential for automatically classifying and interpreting time-series data such as light curves.
""")

st.header("Our Model")
st.write("""
Our machine-learning model analyzes light curves to determine the type of object or phenomenon producing them. It is trained on astronomical time-series data from the PLAsTiCC 2018 dataset depicting multiple measurements of a celestial object’s brightness over time.
""")
st.write("""
Our ML project explores a powerful alternative to the CNN model, leveraging a classical machine learning approach to classify celestial objects. Instead of relying on the raw visual pattern of the light curve, this method uses a **Random Forest** model trained on a set of descriptive statistical and astrophysical features extracted from the time-series data.
""")
st.markdown("""
While a CNN learns features automatically, a Random Forest's power depends on the quality of its input features. By analyzing the properties of the time-series data, we can provide the model with crucial information, such as:
* **Statistical Moments:** Mean, median, standard deviation, skewness, and kurtosis of the brightness measurements.
* **Periodicity Features:** The dominant period of variation, amplitude, and significance of the signal.
* **Flux Ratios:** Ratios of brightness measurements across different photometric bands, or colors.

This process converts the complex time-series problem into a structured tabular classification task, which is ideal for a Random Forest. The model is an ensemble of decision trees that asks a series of yes/no questions about the features, and then aggregates the results to make a robust prediction. By training on hundreds of decision trees, the Random Forest can reliably distinguish between broad categories such as supernova events, variable stars, and black hole-related events.
""")

st.subheader("Advantages of Random Forest")
st.write("""
One of the strongest advantages of using a Random Forest is directly inspecting the model to see which features such as “amplitude” or “period” were most important in classification decisions. These are also known as feature importances; if the model consistently relies on amplitude to separate variable stars from black hole events, that gives us confidence that the classifier is basing its decisions on meaningful properties. This ensures that the ML model isn't just a “black box”, but rather that it could provide insight into its own decision making process.
""")

st.header("The Dataset")
st.write("""
The model was trained on the PLAsTiCC 2018 Kaggle competition dataset, a simulated dataset designed to mimic the cadence and noise properties of real astronomical surveys such as the Large Synoptic Survey Telescope (LSST). PLAsTiCC is particularly challenging as it contains 15 different classes of celestial objects, many of which are rare. This introduces a common problem in ML known as class imbalance, where models can become biased towards predicting the majority classes in the presence of minority classes.
""")
st.write("""
To address class imbalance, particularly for rare astrophysical events, we used SMOTE to generate synthetic examples of underrepresented classes, ensuring the model can learn effectively across all categories. We also decided to focus on a subset of these classes by splitting the 15 classes into three broad categories: Supernovae events, variable stars, and black hole-related events. Although this does not account for the true diversity in astronomical phenomena, it differentiates between important key features and increases the accuracy of the model.
""")
st.success("""
**Result:** This strategy proved highly effective. By focusing on these three categories and training a Random Forest on engineered statistical and time-based features, our classifier achieved an **overall accuracy of 93%**. The performance is almost balanced across all classes, with supernovae and black hole events both classified with ~91% precision and recall, and variable stars achieving ~98%. With more incoming light curve data from a variety of celestial objects/phenomena, however, we hope to include rarer but scientifically valuable events as their own classes in the future.
""")

st.header("About Light Curves & Their Significance")
st.write("""
Light curves are short measurements of a celestial object’s brightness, or “flux”, as a function over time. Different objects have different periods of brightness, and the record of changes in brightness is a tool that can not only help astronomers identify a certain celestial object but also understand its inner workings.
""")
st.write("""
Here is how a certain phenomena may occur via a light curve: When a star explodes, it generates a bright ‘supernova’ signal that fades with time, but that signal does not ever repeat. Other events such as variables vary repeatedly in brightness, in a periodic way (RR Lyrae), episodically at the heart of galaxies (AGNs), and much more.
""")
st.write("""
Variation in these bright sources can provide important clues about themselves and their environment, as well as the evolution of the universe as a whole. Without light curve measurements from Type 1a supernovae, we wouldn't have retrieved the first evidence of accelerated expansion of the universe beyond theory. Variable star light curves provide clues that help us understand stellar evolution, the physics of explosions, and create accurate long-distance estimates between galaxies.
""")