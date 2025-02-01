import streamlit as st
from streamlit_extras.stylable_container import stylable_container
import json
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os
from supabase import create_client, Client
import time
from datetime import datetime

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_API_KEY")
supabase: Client = create_client(SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY)

if 'username_peminjam' not in st.session_state:
    st.session_state['username_peminjam'] = ""

def insert_data_to_database(df):
    try:
        flag = 0
        username_peminjam = df['username_peminjam'].item()
        response_check = supabase.table("customers").select("id_cust").eq("nama_customer", username_peminjam).execute()

        if len(response_check.data) > 0:
            flag = 1
            error_msg = (f"Username '{username_peminjam}' sudah terdaftar. Coba gunakan nama lain")
            return flag, error_msg
        
        response = supabase.table("customers").insert({
            "nama_customer": df['username_peminjam'].item(),
            "id_cust": df['id'].item(),
            "SeriousDlqin2yrs": df['SeriousDlqin2yrs'].item(),
            "RevolvingUtilizationOfUnsecuredLines": df['RevolvingUtilizationOfUnsecuredLines'].item(),
            "age": df['age'].item(),
            "NumberOfTime30-59DaysPastDueNotWorse": df['NumberOfTime30-59DaysPastDueNotWorse'].item(),
            "DebtRatio": df['DebtRatio'].item(),
            "MonthlyIncome": df['MonthlyIncome'].item(),
            "NumberOfOpenCreditLinesAndLoans": df['NumberOfOpenCreditLinesAndLoans'].item(),
            "NumberOfTimes90DaysLate": df['NumberOfTimes90DaysLate'].item(),
            "NumberRealEstateLoansOrLines": df['NumberRealEstateLoansOrLines'].item(),
            "NumberOfTime60-89DaysPastDueNotWorse": df['NumberOfTime60-89DaysPastDueNotWorse'].item(),
            "NumberOfDependents": df['NumberOfDependents'].item(),
            "confidence": df['confidence'].item(),
            "predict_date" : df['predict_date'].item()
        }).execute()

        # Berhasil
        flag = 0
        return flag, "Data Berhasil Disimpan!"

    except Exception as e:
        flag = 2
        st.error(f"An error occurred: {str(e)}")
        return flag, str(e)

def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

def load_model(model_path='model_after_tune/best_random_forest_model.joblib'):
    model = joblib.load(model_path)
    return model

def predict(model, data):
    proba = model.predict_proba(data)[0]
    predict = model.predict(data)[0]
    
    if predict == 0:
        result = "Layak"
        confidence = int(proba[0] * 100)
    else:
        result = "Tidak Layak"
        confidence = int(proba[1] * 100)
    
    return result, confidence

@st.dialog("Hasil Prediksi")
def predict_result(result, confidence, df, username_peminjam, id, importance_df):
    
    result_int = 99
    current_time = datetime.now().strftime('%d %B %Y, %H:%M')
    
    if result == "Layak":
        result_int = 0
        color = "green"
        message = f"🟢 **Status Pinjaman: {result}**\n\nTingkat keyakinan model: **{confidence}%**"
    else:
        result_int = 1
        color = "red"
        message = f"🔴 **Status Pinjaman: {result}**\n\nTingkat keyakinan model: **{confidence}%**"

    st.markdown(message)
    

    progress_bar_style = f"""
    <style>
    div[data-baseweb="progress-bar"] > div > div > div{{
        background-color: {color} !important;
        border-radius: 20px;
    }}
    </style>
    """
    st.markdown(progress_bar_style, unsafe_allow_html=True)

    progress = st.progress(0)
    for percent_complete in range(confidence):
        time.sleep(0.01)
        progress.progress(percent_complete + 1)

    st.write(f"Nama Peminjam: **{st.session_state['username_peminjam']}**")
    st.write(f"Tanggal dan Waktu: **{current_time}**")

    with st.expander("Detail Penjelasan"):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df['Feature'], importance_df['Importance (%)'], color='skyblue')
        ax.set_xlabel("Pentingnya Fitur (%)")
        ax.set_ylabel("Fitur")
        ax.set_title("Hal-hal yang Mempengaruhi Penilaian Model")
        plt.gca().invert_yaxis()
        
        st.pyplot(fig)

        st.write("**1. Rasio Pemakaian Kredit**")
        st.write("Semakin tinggi rasio pemakaian kredit (utilization ratio), semakin tinggi pula risiko gagal bayar. Hal ini menunjukkan bahwa seseorang memanfaatkan kreditnya secara maksimal, yang bisa mengindikasikan tekanan finansial atau ketergantungan yang tinggi pada kredit. Rasio pemakaian kredit berpengaruh sebesar 32% dalam penentuan keberhasilan pinjaman, sehingga biasanya menunjukkan adanya masalah finansial yang mendalam dan berisiko tinggi.") 

        st.write("**2. Jumlah Tanggungan**")
        st.write("Semakin banyak jumlah tanggungan yang dimiliki, semakin besar pula risiko gagal bayar. Hal ini disebabkan karena semakin banyak tanggungan, semakin besar pula beban keuangan yang harus ditanggung oleh individu. Misalnya, seseorang dengan lebih dari 3 tanggungan dapat memiliki kesulitan dalam mengelola pengeluaran dan cicilan. Jumlah Tanggungan berpengaruh sebesar 23% dalam penentuan resiko pinjaman") 

        st.write("**3. Rasio Utang**")
        st.write("Semakin tinggi rasio utang terhadap pendapatan, semakin tinggi risiko gagal bayar. Rasio ini mencerminkan sejauh mana penghasilan seseorang telah dialokasikan untuk membayar utang, dan rasio yang tinggi menunjukkan kapasitas pembayaran utang yang semakin terbatas. Biasanya, rasio utang lebih dari 0.4 dapat meningkatkan kemungkinan kegagalan pembayaran. Rasio utang berpengaruh terhadap 10% terhadap keputusan dalam melakukan pinjaman") 

        st.write("**4. Pendapatan Bulanan**")
        st.write("Pendapatan bulanan yang lebih rendah dari standar ekonomi dapat meningkatkan risiko gagal bayar. Pendapatan bulanan yang terlalu rendah, terutama jika kurang dari Rp 3.000.000, dapat mempersulit seseorang untuk memenuhi kewajiban utang. Pendapat bulanan ekonomi menentukan keputusan pinjaman sebesar 9%") 

        st.write("**5. Usia**")
        st.write("Usia sangat berpengaruh dalam menentukan kemampuan seseorang untuk membayar utang. Usia di bawah 25 tahun atau lebih dari 70 tahun bisa menunjukkan risiko lebih tinggi dalam pembayaran, karena usia muda mungkin belum memiliki kestabilan finansial, sementara usia lebih dari 70 tahun mungkin sudah memasuki masa pensiun dan memiliki pendapatan yang terbatas. Umumnya, usia di atas 75 tahun menunjukkan kemungkinan gagal bayar yang lebih tinggi dan umur juga berpengaruh sebesar 7% dalam pengambilan keputusan") 

        st.write("**6. Jumlah Keterlambatan Lebih dari 3 Bulan**")
        st.write("Jumlah keterlambatan pembayaran yang lebih dari 3 bulan menunjukkan masalah serius dalam kemampuan finansial seseorang. Lebih dari 1 kali keterlambatan 90 hari menunjukkan ketidakmampuan membayar yang signifikan dan sangat meningkatkan risiko gagal bayar, sehingga faktor keterlambatan lebih dari 3 bulan berpengaruh sebesar 6%") 

        st.write("**7. Jumlah Kartu Kredit/Pinjaman**")
        st.write("Semakin banyak kartu kredit atau pinjaman yang dimiliki, semakin besar potensi beban finansial yang harus dikelola. Terlalu banyak kartu kredit (lebih dari 5) atau pinjaman bisa menunjukkan ketergantungan yang tinggi pada utang, yang meningkatkan risiko gagal bayar. Jumlah kartu kredit mempengaruhi sebesar 5% terhadap pengambilan keputusan dalam pinjaman online") 

        st.write("**8. Jumlah Keterlambatan 1-2 Bulan**")
        st.write("Keterlambatan dalam pembayaran utang selama 1-2 bulan, meskipun lebih rendah risiko dibandingkan dengan keterlambatan 3 bulan atau lebih, tetap menunjukkan adanya masalah dalam pengelolaan keuangan. Jika keterlambatan ini terjadi lebih dari dua kali, risiko gagal bayar akan meningkat sehingga mempengaruhi keputusan pinjaman sebesar 4% ") 

        st.write("**9. Jumlah Pinjaman Properti**")
        st.write("Jumlah pinjaman properti yang dimiliki dapat menunjukkan apakah seseorang memiliki beban finansial yang besar. Peminjaman properti yang berlebihan, terutama jika lebih dari 2 properti, bisa menyebabkan ketegangan finansial yang berisiko menyebabkan gagal bayar, pinjaman properti yang tinggi menjadi faktor pengambilan keputusan sebesar 3%")

        st.write("**10. Jumlah Keterlambatan 2-3 Bulan**")
        st.write("Jumlah keterlambatan pembayaran utang selama 2-3 bulan meningkatkan kemungkinan gagal bayar. Jika ada lebih dari dua kali keterlambatan di rentang waktu ini, individu tersebut berisiko tinggi mengalami masalah finansial, keterlambatan ini dapat mempengaruhi pinjaman sebesar 1%")

    
    df['confidence'] = confidence
    df['username_peminjam'] = username_peminjam
    df['SeriousDlqin2yrs'] = result_int
    df['id'] = id
    df['predict_date'] = current_time

    if st.button("Simpan Prediksi"):
        flag, message = insert_data_to_database(df)

        if flag == 0:
            st.success(message)
        elif flag == 1:
            st.error(message)
        elif flag == 2:
            st.error(f"Terjadi kesalahan: {message}") 
        time.sleep(5)
        st.rerun()

def call_train():
    # st.write(st.session_state)
    info_html = """
        <style>
        .label-container {
            display: flex;
            align-items: center;
        }

        .tooltip {
            position: relative;
            display: inline-block;
            cursor: pointer;
            color: #1E90FF;
            margin-left: 8px;
        }

        .tooltip .tooltiptext {
            visibility: hidden;
            width: 300px;
            background-color: #6c757d;
            color: #fff;
            text-align: center;
            border-radius: 5px;
            padding: 5px;
            position: absolute;
            z-index: 1;
            bottom: 125%; /* Posisi tooltip di atas teks */
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
        }

        .tooltip:hover .tooltiptext {
            visibility: visible;
            opacity: 1;
        }
        </style>
    """

    st.markdown(info_html, unsafe_allow_html=True)
    with stylable_container(
        key="train-container",
        css_styles=""" 
            padding: 20px;
            border-radius: 10px;
            background-color: white;
        """
    ):

        with st.form("predict_form", enter_to_submit=False):
            model = load_model()
            # Check if the model has feature_importances_ attribute
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
            else:
                raise AttributeError("The loaded model does not support 'feature_importances_'")
            
            dataset_dataframe = pd.read_csv((os.path.join('dataset', 'cs-training.csv')))
            dataset_dataframe = pd.DataFrame(dataset_dataframe)
            
            dataset_dataframe = dataset_dataframe.drop("Unnamed: 0", axis=1)
            dataset_dataframe = dataset_dataframe.drop("SeriousDlqin2yrs", axis=1)
            # st.write(dataset_dataframe)
            
            # Assuming X_train was a pandas DataFrame
            feature_names = dataset_dataframe.columns if isinstance(dataset_dataframe, pd.DataFrame) else [f"Feature {i}" for i in range(len(feature_importances))]

            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance (%)': feature_importances
            }).sort_values(by='Importance (%)', ascending=False)

            # Mengubah Nama Fitur
            importance_df['Feature'] = importance_df['Feature'].replace({
                "RevolvingUtilizationOfUnsecuredLines": "Rasio Pemakaian Kredit",
                "NumberOfDependents": "Jumlah Tanggungan",
                "DebtRatio": "Rasio Utang",
                "MonthlyIncome": "Pendapatan Bulanan",
                "age": "Usia",
                "NumberOfTimes90DaysLate": "Jumlah Keterlambatan lebih dari 3 Bulan",
                "NumberOfOpenCreditLinesAndLoans": "Jumlah Kartu Kredit/Pinjaman",
                "NumberOfTime30-59DaysPastDueNotWorse": "Jumlah Keterlambatan 1-2 Bulan",
                "NumberRealEstateLoansOrLines": "Jumlah Pinjaman Properti",
                "NumberOfTime60-89DaysPastDueNotWorse": "Jumlah Keterlambatan 2-3 Bulan"
            })

            # Mengubah importance menjadi persen
            importance_df['Importance (%)'] = (importance_df['Importance (%)'] * 100).round(0).astype(int)

            # Print the DataFrame
            # st.dataframe(importance_df, hide_index=True)

            # Username
            st.markdown(
                """
                <div class="label-container">
                    <span><strong>Nama Peminjam</strong></span>
                    <div class="tooltip">
                        <i>ⓘ</i>
                        <span class="tooltiptext">
                            Nama Peminjam
                        </span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            username_peminjam = st.text_input("Masukkan nilai", label_visibility="hidden")
            st.session_state['username_peminjam'] = username_peminjam
            # st.divider()

            col1, col2, col3 = st.columns(3)

            with col1:
                # RevolvingUtilizationOfUnsecuredLines
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>1. Rasio Pemakaian Kredit</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Total saldo pada kartu kredit dan jalur kredit pribadi, tidak termasuk properti real estate dan utang tanpa angsuran seperti pinjaman mobil, dibagi dengan total batas kredit.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                revolving = st.number_input("Masukkan Revolving", min_value=0.00, key='revolving', label_visibility="hidden", max_value=10.00)
                # st.divider()

                # Monthly Income
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>4. Penghasilan Bulanan (IDR)</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Pendapatan bulanan peminjam. Pendapatan yang lebih tinggi dapat menunjukkan kapasitas keuangan yang lebih baik, yang dapat mengurangi kemungkinan gagal bayar.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                monthly_income = st.number_input("Masukkan pendapatan bulanan", min_value=1000000, label_visibility="hidden", key='income', max_value=1000000000)
                monthly_income = int(monthly_income/16000)
                # st.divider()

                # Number of Dependents
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>7. Jumlah Tanggungan</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah tanggungan yang dimiliki peminjam, tidak termasuk dirinya sendiri. Semakin banyak tanggungan, semakin besar kemungkinan kesulitan keuangan.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                number_dependents = st.number_input("Masukkan jumlah tanggungan", min_value=0, max_value=20, label_visibility="hidden", key='dependents')
                # st.divider()

                # NumberOfTimes90DaysLate
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>10. Jumlah Terlambat Bayar (Lebih dari 3 Bulan)</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah kali peminjam terlambat 90 hari atau lebih dalam 2 tahun terakhir.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                past_90 = st.number_input("Masukkan jumlah keterlambatan 90 hari atau lebih", min_value=0, max_value=8, label_visibility="hidden", key='past_90')
                # st.divider()
            
            with col2:
                # Age
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>2. Usia</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Usia peminjam dalam tahun. Usia dapat mempengaruhi stabilitas keuangan dan kemungkinan mengalami kesulitan keuangan seperti gagal bayar.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                age = st.number_input("Masukkan usia", min_value=18, max_value=100, label_visibility="hidden", key='age')
                # st.divider()

                # Number of Open Credit Lines and Loans
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>5. Jumlah Kartu Kredit/Pinjaman</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah pinjaman terbuka (seperti pinjaman mobil atau hipotek) dan jalur kredit (misalnya kartu kredit). Lebih banyak kredit terbuka dapat menunjukkan risiko gagal bayar lebih tinggi jika tidak dikelola dengan baik.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                number_credit = st.number_input("Masukkan jumlah kredit terbuka", min_value=0, max_value= 20,label_visibility="hidden", key='open_credit')
                # st.divider()

                # NumberOfTime30-59DaysPastDueNotWorse
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>8. Jumlah Terlambat Bayar (1-2 Bulan)</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah kali peminjam terlambat 30-59 hari namun tidak lebih buruk dalam 2 tahun terakhir.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                past_30 = st.number_input("Masukkan jumlah keterlambatan 30-59 hari", min_value=0, max_value=24, label_visibility="hidden", key='past_30')
                # st.divider()
            
            with col3:
                # Debt Ratio
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>3. Rasio Utang</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Rasio pembayaran utang bulanan, termasuk tunjangan dan biaya hidup, dibagi dengan pendapatan kotor bulanan. Rasio utang yang tinggi dapat menunjukkan stres keuangan.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                debt_ratio = st.number_input("Masukkan rasio utang", min_value=0.00, max_value=10.00, label_visibility="hidden", key='debt')
                # st.divider()
            
                # Number Real Estate Loans or Lines
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>6. Jumlah Pinjaman Properti</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah pinjaman hipotek dan real estate, termasuk jalur kredit ekuitas rumah. Semakin banyak pinjaman real estate, semakin besar kewajiban finansial yang dimiliki peminjam.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                number_real_estate = st.number_input("Masukkan jumlah kredit properti", min_value=0, max_value=20, label_visibility="hidden", key='real_estate')
                # st.divider()
            
                # NumberOfTime60-89DaysPastDueNotWorse
                st.markdown(
                    """
                    <div class="label-container">
                        <span><strong>9. Jumlah Terlambat Bayar (2-3 Bulan)</strong></span>
                        <div class="tooltip">
                            <i>ⓘ</i>
                            <span class="tooltiptext">
                                Jumlah kali peminjam terlambat 60-89 hari namun tidak lebih buruk dalam 2 tahun terakhir.
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                past_60 = st.number_input("Masukkan jumlah keterlambatan 60-89 hari", min_value=0, max_value=12, label_visibility="hidden", key='past_60')
                # st.divider()

            submit_button = st.form_submit_button("Predict!")
            if submit_button:
                if not username_peminjam:
                    st.error("Nama Peminjam Tidak Boleh Kosong")
                else:
                    input_data = {'RevolvingUtilizationOfUnsecuredLines': revolving, 'age': age, 'NumberOfTime30-59DaysPastDueNotWorse' : past_30,
                                'DebtRatio': debt_ratio, 'MonthlyIncome': monthly_income, 'NumberOfOpenCreditLinesAndLoans': number_credit, 
                                'NumberOfTimes90DaysLate' : past_90, 'NumberRealEstateLoansOrLines' : number_real_estate, 'NumberOfTime60-89DaysPastDueNotWorse' : past_60, 'NumberOfDependents': number_dependents}
                    input_data = pd.DataFrame([input_data])
                    st.session_state['username_peminjam'] = username_peminjam
                    result, confidence, = predict(model, input_data)
                    if result is not None and confidence is not None:
                        id = st.session_state['id']
                        predict_result(result, confidence, input_data, username_peminjam, id, importance_df)




