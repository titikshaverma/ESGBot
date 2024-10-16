import streamlit as st
from PIL import Image
import os
print(os.getcwd())


class UiConfig:
    project_name = "ESG CHATBOT"

    def setup():
        im = Image.open("app/images/favicon.ico")

        st.set_page_config(page_title="Eco Insight APP 🌏", page_icon=im,layout="wide")

        t1, t2 = st.columns((0.25, 1))
        t1.title("")
        t1.image("app/images/logo.png", width=100)
        t2.title("Eco Insight APP 🌏")
        t2.markdown("**Eco Insight, a sophisticated chatbot enriched with LLM capabilities, is your dedicated resource for engaging conversations on sustainability reports following the BRSR framework. Unleashing the power of Mistral and LangChain, our bot delivers meticulous trend analysis, providing you with unparalleled insights. Navigate through sustainability data effortlessly and make informed decisions with Eco Insight's  professional approach to ESG intelligence.**")
