import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import os
import pandas as pd
from supabase import create_client, Client


# Initialize Supabase client
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)


def fetch_prediction_history(id):
    try:
        response = supabase.table('customers').select('*').eq("id_cust", id).execute()
        if len(response.data) != 0:
            return response.data
        else:
            return None
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        return None
    
def display_confidence_bar(confidence):
    st.markdown(f"""
        <div style="
            display: flex; 
            align-items: center; 
            margin-right: 40px;
            margin-left: 40px;
            margin-bottom: 20px;
            gap: 10px;">
            <span style="font-weight: bold; margin-right: 10px;">Keyakinan:</span>
            <div style="width: 100%; height: 10px; border-radius: 10px; 
                background: linear-gradient(to right, #00c853 {confidence}%, #e0e0e0 {confidence}%);">
            </div>
            <div style="font-weight: bold;">{confidence}%</div>
        </div>
    """, unsafe_allow_html=True)

@st.dialog("Rincian Customer")
def user_details(id, username):
    df = fetch_prediction_history(id)
    df = pd.DataFrame(df)
    filtered_df = df.loc[df['nama_customer'] == username]

    if not filtered_df.empty:
        customer = filtered_df.iloc[0]

        monthly_income_idr = f"Rp {int(customer['MonthlyIncome'] * 16000):,}".replace(",", ".")
        status_pinjaman = "Layak" if customer['SeriousDlqin2yrs'] == 0 else "Tidak Layak"

        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
                **Nama Customer:** {customer["nama_customer"]}  
                **Tanggal Prediksi:** {customer["predict_date"]}  
                **Age:** {customer["age"]}  
                **Rasio Pemakaian Kredit:** {customer["RevolvingUtilizationOfUnsecuredLines"]}  
                **Rasio Utang:** {customer["DebtRatio"]}  
                **Penghasilan Bulanan:** {monthly_income_idr}  
            """)

        with col2:
            st.markdown(f"""
                **Jumlah Kartu Kredit:** {customer["NumberOfOpenCreditLinesAndLoans"]}  
                **Jumlah Pinjaman Properti:** {customer["NumberRealEstateLoansOrLines"]}  
                **Jumlah Tanggungan:** {customer["NumberOfDependents"]}  
                **Jumlah Terlambat Bayar (1-2 Bulan):** {customer["NumberOfTime30-59DaysPastDueNotWorse"]}  
                **Jumlah Terlambat Bayar (2-3 Bulan):** {customer["NumberOfTime60-89DaysPastDueNotWorse"]}  
                **Jumlah Terlambat Bayar (Lebih dari 3 Bulan):** {customer["NumberOfTimes90DaysLate"]}  
            """)
        
        st.markdown(f"""**Status Pinjaman:** {status_pinjaman}""")

    else:
        st.write("Nama Customer Tidak ditemukan di Database.")


def display_prediction_history(prediction_data):
    if prediction_data is not None and len(prediction_data) > 0:
        df = pd.DataFrame(prediction_data)

        col1, col2 = st.columns(2)

        for index, row in df.iterrows():
            target_col = col1 if index % 2 == 0 else col2
            with target_col:
                border_color = "#00c853" if row["SeriousDlqin2yrs"] == 0 else "#d32f2f"
                status_pinjaman = "Layak" if row["SeriousDlqin2yrs"] == 0 else "Tidak Layak"

                with stylable_container(
                    key=f"prediction-{index}",
                    css_styles=f"""
                        {{
                            background-color: #f9f9f9;
                            padding: 20px;
                            margin-bottom: 10px;
                            border: 2px solid {border_color};
                            border-radius: 8px;
                            text-align: center;
                            display: flex;
                            flex-direction: column;
                            align-items: center;
                        }}
                    """,
                ):
                    st.markdown(f"""
                        <div style="font-size: 24px; font-weight: bold; margin-bottom: 5px; text-align: center;">
                            {row["nama_customer"]}
                        </div>
                        <div style="font-size: 18px; font-weight: bold; color: {border_color}; margin-bottom: 10px; text-align: center;">
                            {status_pinjaman}
                        </div>
                        <div style="font-size: 16px; text-align: center;">
                            Tanggal Prediksi: {row["predict_date"]}
                        </div>
                    """, unsafe_allow_html=True)

                    display_confidence_bar(row["confidence"])

                    if st.button("Rincian", key=f"details_{row['nama_customer']}"):
                        user_details(st.session_state['id'], row['nama_customer'])

    else:
        st.write("No data available.")


def call_home():
    with stylable_container(
        key="home-title",
        css_styles="""
            {
                text-align: center;
                align-items: center;
                margin-bottom: 100px;
            }
        """,
    ):
        st.write("""
            <div style="
                display: block;
                width: 100%;
                padding: 20px 0;
                border: 2px solid rgba(0, 0, 0, 0.1);
                border-radius: 10px;
                background-color: #f9f9f9;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                font-size: 70px;
                font-weight: bold;
                text-align: center;
            ">
                Safelend
            </div>
        """, unsafe_allow_html=True)
        
    st.subheader("Prediction History")
    prediction_data = fetch_prediction_history(st.session_state['id'])
    display_prediction_history(prediction_data)

