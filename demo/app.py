# app.py
import streamlit as st

# --- Page Configuration and Custom CSS ---
st.set_page_config(
    page_title="Exoplanet AI",
    page_icon="🌌",
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

local_css("demo/style.css")

# --- Page Content ---
st.title("Welcome to the Exoplanet Classifier Project 🌌")
st.markdown("### Using Deep Learning to Distinguish Planets from Stars")
st.markdown("---")

st.header("The Challenge: A Universe of Data")
st.write("""
Astronomical surveys like NASA's TESS mission produce an incredible amount of data, monitoring the brightness of millions of stars over time. 
Within this data are faint, repeating dips in starlight that could signal a transiting exoplanet. However, many other phenomena, like eclipsing binary stars, can create similar-looking signals.
""")

st.header("Our Solution: The CNN 'Black Box'")
st.write("""
While the difference between a U-shaped exoplanet transit and a V-shaped stellar eclipse can sometimes be obvious, in the real world, most signals are buried in noise. A faint planetary transit might look V-shaped, and a grazing eclipse might have a slightly rounded bottom, making classification by eye difficult and subjective.

This is where the Convolutional Neural Network (CNN) excels. While its internal workings can seem like a "black box," its effectiveness comes from its ability to perform a highly sophisticated, quantitative analysis of the dip's shape, learning subtle features that are nearly impossible for a human to consistently identify.

Beyond the simple U-vs-V distinction, the CNN learns to classify based on a combination of more subtle morphological differences, such as:

The Curvature of the Dip: The model learns the precise mathematical curve of the dip's bottom, distinguishing between a perfectly flat transit, a slightly curved one due to limb darkening, and the sharp point of a stellar eclipse.

Symmetry: It analyzes the symmetry between the ingress (the dip's entry) and the egress (the exit). An asymmetrical dip can be a sign of starspots on the host star or other stellar phenomena, often indicating a false positive.

The "Shoulders" of the Dip: The sharpness of the transition from the flat, out-of-transit baseline to the steep slope of the dip is another key feature. This can hint at the properties of the star's atmosphere and the nature of the eclipsing object.

By analyzing these features across thousands of examples, the CNN builds a complex, mathematical intuition for what makes a signal a "planet," allowing it to find the faint, hidden patterns in the noise.
Navigate to the **🚀 Demo** page to see the model in action!
""")

st.markdown("---") # Add a separator

# --- "Data Processing" Section ---
st.header("Data Processing: Forging Order from Chaos")
st.write("""
A machine learning model is only as good as the data it's trained on. The raw data from space telescopes is messy, unlabeled, and inconsistent. A significant portion of this project was dedicated to overcoming these obstacles through a robust data processing pipeline.
""")

st.subheader("The Labeling Challenge")
st.write("""
The first major obstacle was that raw light curves are simply unlabeled time-series data. To train a supervised model, we needed "ground truth"—a reliable label for each star. This required scraping data from not one, but two separate, expert-curated databases:
""")
st.markdown("""
1.  **The NASA Exoplanet Archive:** For high-confidence confirmed exoplanets.
2.  **The TESS Objects of Interest (TOI) Catalog:** For planet candidates and, crucially, for known false positives like eclipsing binary stars.
""")

st.subheader("The Transformation Pipeline")
st.write("""
The second obstacle was that raw light curves are not uniform. To prepare this data for the CNN, which requires perfectly consistent inputs, we performed several key transformations:
""")
st.markdown("""
- **Period Finding:** Analyzed each star's observations to find the repeating period of the signal.
- **Stitching:** Combined all available data for a star to improve the signal's strength.
- **Flattening:** Removed the star's own variability (e.g., from starspots) to isolate the dip.
- **Folding & Centering:** Stacked all the dips on top of each other, centering the event at Phase 0.
- **Binning & Normalizing:** Resized every light curve to a uniform length for the CNN.
""")