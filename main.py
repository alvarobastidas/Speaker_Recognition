import streamlit as st
from show_training_model import show_training_model
from show_test_model import show_test_model
from show_create_dataset import create_dataset

# WEB PAGE TITLE
st.title("SPEAKER RECOGNITION")

# WEB PAGE MENU
page = st.sidebar.selectbox("Menu", ("Main", "Create Dataset", "Training Model", "Testing Model"))

# DISPLAY MAIN MENU
if page == "Training Model":
    st.write("""## TRAINING MODEL""")
    show_training_model()

elif page == "Testing Model":
    st.write("""## MODEL TESTING""")
    show_test_model()

elif page == "Create Dataset":
    st.write("""## CREATE DATASET""")
    create_dataset()
else:
    st.write("""# Northumbria University""")
    st.write("""## LD7083 : Computing and Digital Technologies Project""")
    st.write("""### Alvaro Bastidas""")
    st.write("""### Studen ID: 19036077""")
    st.image("Images/deep_learning.jpeg")



