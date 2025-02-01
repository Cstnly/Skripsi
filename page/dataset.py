import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import pandas as pd

def call_dataset():
    st.write("""
            <div style="font-size: 50px;font-weight: bold; color: black; text-align: center;">
                Project Dataset
            </div>
            """,unsafe_allow_html=True)
    with stylable_container(
        key="dataset-body",
        css_styles="""
            {
                background-color: #FFFFFF;
                text-align: center;
                padding: 20px;
            }
        """,
    ):  
        dataset_col1, dataset_col2 = st.columns(2)
        with dataset_col1:
            st.write("""
            <div style="color: red; margin-bottom: 10px;">
                Dataset Before Cleaning
            </div>
            """,unsafe_allow_html=True)

            dataset_before = pd.read_csv("dataset\cs-training.csv")
            st.write(dataset_before)
        
        with dataset_col2:
            st.write("""
            <div style="color: green; margin-bottom: 10px;">
                Dataset After Cleaning
            </div>
            """,unsafe_allow_html=True)

            dataset_after = pd.read_csv("dataset\cleaned_data.csv")
            st.write(dataset_after)
        
        with st.expander("What We Do"):
            st.write("oww")