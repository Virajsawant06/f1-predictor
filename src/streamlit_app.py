import streamlit as st
import requests
import pandas as pd

API_URL = "http://localhost:8000"

st.set_page_config(page_title="F1 Race Predictor", page_icon="🏎️", layout="wide")

tab1, tab2 = st.tabs(["Single Driver", "Full Race Simulator"])