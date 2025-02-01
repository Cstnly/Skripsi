import streamlit as st
import pandas as pd
import base64
from streamlit_option_menu import option_menu
from supabase import create_client, Client
import os
from dotenv import load_dotenv
import bcrypt
from streamlit_extras.stylable_container import stylable_container
from page.home import call_home
from page.dataset import call_dataset
from page.train import call_train
import time
import requests

# Load environment variables
load_dotenv()

# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_PUBLIC_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

if 'id' not in st.session_state:
    st.session_state['id'] = 0

if 'authentication_status' not in st.session_state:
    st.session_state['authentication_status'] = False

def inject_css():
    with open('./assets/styles.css') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def fetch_user_password(username):
    response = supabase.table("users").select("password_hashed").eq("username", username).execute()
    if len(response.data) == 0:
        return None
    return response.data[0]["password_hashed"]

def fetch_user_phone_number(username):
    response = supabase.table("users").select("phone_number").eq("username", username).execute()
    if len(response.data) == 0:
        return None
    return response.data[0]["phone_number"]

def fetch_user_id(username):
    response = supabase.table("users").select("id").eq("username", username).execute()
    if len(response.data) == 0:
        return None
    return response.data[0]["id"]

@st.dialog("Forgot Password")
def forgot_pw():
    username = st.text_input("Username:")
    phone_number = st.text_input("Phone Number:")
    new_password = st.text_input("New Password:", type="password")

    if st.button("Submit"):
        if not new_password:
            st.error("Kolom Password yang baru tidak boleh kosong")
        elif len(new_password) < 5 or len (new_password) > 20:
            st.error("Kolom Password yang baru harus memiliki panjang antara 5-20 karakter")
        else:
            response_phone_number = fetch_user_phone_number(username)
            if response_phone_number == phone_number:
                hashed_new_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

                url = "http://localhost:5000/change_password"
                data = {
                    "username": username,
                    "encryptedPassword": hashed_new_password
                }
                
                try:
                    response = requests.post(url, json=data)

                    if response.status_code == 200:
                        st.success("Perubahan password telah berhasil, silahkan login kembali menggunakan password yang baru")
                        time.sleep(5)
                        st.session_state.forgot = False
                        st.rerun()
                    else:
                        st.error(f"Gagal merubah password. Server Error: {response.status_code}")
                        st.write(f"Server response: {response.text}")
                        time.sleep(5)
                        st.session_state.forgot = False
                        st.rerun()

                except Exception as e:
                    st.error(f"Error pada server: {e}")
            else:
                st.error("Username atau nomor telepon yang dimasukkan salah.")



def main():
    st.set_page_config(
        page_title="Credit Scoring Prediction",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    if not st.session_state['authentication_status']:
        
        with st.form(key="login_form"):
                st.title("Login")

                username = st.text_input("Username")
                password = st.text_input("Password", type="password")

                if 'username' not in st.session_state:
                    st.session_state['username'] = ""

                login_btn = st.form_submit_button("Login")
                if login_btn:
                    if not username and not password:
                        st.error("Kolom Username dan Password Tidak Boleh Kosong")
                    elif not username:
                        st.error("Kolom Username Tidak Boleh Kosong!")
                    elif not password:
                        st.error("Kolom Password Tidak Boleh Kosong!")
                    else:
                        hashed_password = fetch_user_password(username)
                        if hashed_password and bcrypt.checkpw(password.encode('utf-8'), hashed_password.encode('utf-8')):
                            st.session_state["authentication_status"] = True
                            st.session_state["username"] = username
                            st.rerun()
                        else:
                            st.session_state["authentication_status"] = False
                            st.error("Username/password yang anda masukkan salah.")
                
        forgot_btn = st.button("Forgot Password")
        if forgot_btn:
            st.session_state["forgot"] = True
    
    if st.session_state.get("forgot"):
        forgot_pw()
                        
    # If authenticated, show main app content
    if st.session_state["authentication_status"]:
        st.session_state['id'] = fetch_user_id(st.session_state['username'])
        inject_css()

        with st.sidebar:
            selected = option_menu(
                menu_title="Menu",
                options=["Beranda", "Prediksi"],
                icons=["house-fill", "lightbulb-fill"],
                default_index=0,
                orientation='vertical',
                styles = {
                    "container": {"padding" : "1!important", "background-color": "black", "color" : "white", "border-radius" : "5px"},
                    "icon": {"color": "white", "font-size": "18px"},
                    "nav-link": {"--hover-color": "#B23A2F", "color": "white", "border-radius" : "0px"},
                    "nav-link-selected": {"background-color": "red", "color" : "white", "border-radius" : "0px"},
                    "menu-title" : {"color" : "white"},
                }
            )

            with stylable_container(
                key="button_logout",
                css_styles="""
                    button {
                        height: 30px;
                        background-color: red;
                        color: black;
                        border-radius: 5px;
                        white-space: nowrap;
                    }
                """,
            ):
                logout_btn = st.button("Keluar", icon=":material/logout:")
                if logout_btn:
                    keys_to_remove = ["authentication_status", "username", "id", "forgot"]
                    for key in keys_to_remove:
                        if key in st.session_state:
                            del st.session_state[key]
                    st.rerun()


        if selected == "Beranda":
            call_home()
        elif selected == "Prediksi":
            call_train()

if __name__ == "__main__":
    main()
