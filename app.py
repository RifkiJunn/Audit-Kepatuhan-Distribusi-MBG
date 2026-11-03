import json
import time

import joblib
import pandas as pd
import streamlit as st
from openai import OpenAI

from classifier import NAMA_FITUR, PROVIDER, KuotaHabis, classify, load_config

st.set_page_config(page_title="Audit Kepatuhan MBG", page_icon=":material/restaurant:", layout="wide")

# ---------- 1. Muat model library & konfigurasi LLM ----------
model = joblib.load("model_mbg.pkl")
with open("model_info.json", encoding="utf-8") as f:
    info = json.load(f)
cfg = load_config("config.json")
FITUR = info["fitur_numerik"] + info["fitur_kategori"]

# warna & arti tiap label
STATUS = {
    "Exemplary-Gold-Standard": ("Teladan", "#1B6E4A", "#E3F3EA", "Sangat baik di semua aspek, contoh praktik terbaik."),
    "Standard-Compliant": ("Sesuai Standar", "#1F4E8C", "#E6EEF8", "Memenuhi standar program."),
    "Marginal-Quality-At-Risk": ("Berisiko", "#9A6700", "#FFF4D6", "Masih berjalan, tetapi kualitas mulai berisiko."),
    "Non-Compliant-Intervention-Required": ("Perlu Intervensi", "#B42318", "#FDECEA", "Tidak memenuhi standar, perlu tindakan."),
}
st.markdown("""<style>
.block-container {max-width: 1100px; padding-top: 2rem;}
h1 {letter-spacing: -0.03em; font-weight: 800;}
::selection {background: #1F4E8C; color: #fff;}
.hasil {border-radius: 14px; padding: 1.2rem 1.4rem; box-shadow: 0 6px 20px rgba(20,33,61,.08);
        animation: muncul .35s cubic-bezier(.16,1,.3,1);}
.hasil small {text-transform: uppercase; letter-spacing: .06em; font-weight: 600; opacity: .85;}
.hasil b {display: block; font-size: 1.9rem; font-weight: 800; letter-spacing: -0.02em; margin: .1rem 0 .2rem;}
.hasil code {background: transparent; color: inherit; font-size: .8rem; opacity: .8;}
@keyframes muncul {from {opacity: .4; transform: translateY(6px);} to {opacity: 1; transform: none;}}
</style>""", unsafe_allow_html=True)


def kartu(judul, label):
    nama, fg, bg, arti = STATUS.get(label, (label, "#14213D", "#EAF0F7", "Jawaban model di luar daftar label."))
    st.markdown(f'<div class="hasil" style="background:{bg};color:{fg}"><small>{judul}</small>'
                f'<b>{nama}</b>{arti}<br><code>{label}</code></div>', unsafe_allow_html=True)


# ---------- 2. Client LLM (hanya jika API key tersedia) ----------
NAMA_KEY = PROVIDER[cfg["provider"]]["env_key"]
try:
    API_KEY = st.secrets[NAMA_KEY]
except Exception:                       # secrets.toml belum ada / key belum diisi
    API_KEY = None


@st.cache_resource
def get_client():
    return OpenAI(base_url=PROVIDER[cfg["provider"]]["base_url"], api_key=API_KEY)


@st.cache_data(show_spinner=False)
def prediksi_llm(data_json):
    # data yang sama tidak memanggil API dua kali (hemat kuota)
    return classify(get_client(), cfg, json.loads(data_json))


def prediksi_library(df):
    return model.predict(df[FITUR]), model.predict_proba(df[FITUR])


# ---------- 3. Sidebar ----------
with st.sidebar:
    st.subheader(":material/tune: Pilih Model")
    pilihan = st.radio("Model yang dipakai", ["Model Library", "AI Model Service", "Bandingkan keduanya"],
                       captions=[info["nama_model"] + " (scikit-learn)", f"{cfg['model']} ({cfg['provider'].capitalize()})",
                                 "Tampilkan hasil keduanya"], label_visibility="collapsed")
    if pilihan != "Model Library" and not API_KEY:
        st.error(f"{NAMA_KEY} belum diatur di Secrets, AI Model Service tidak bisa dipakai.")

    st.caption(f"Kinerja pada {info['metrik_test_llm']['n']} data uji yang sama")
    lib, ai = info["metrik_test_llm"], cfg["evaluasi"]
    c1, c2 = st.columns(2)
    c1.metric("F1 Library", f"{lib['f1_macro']:.2f}")
    c2.metric("F1 AI", f"{ai['f1_macro']:.2f}", delta=f"{ai['f1_macro'] - lib['f1_macro']:+.2f}")
    st.progress(min(lib["f1_macro"], 1.0), text=f"Library  {lib['accuracy']:.0%} akurasi")
    st.progress(min(ai["f1_macro"], 1.0), text=f"AI  {ai['accuracy']:.0%} akurasi")
    if not cfg.get("lolos", True):
        st.badge("AI belum lolos gerbang evaluasi", icon=":material/error:", color="orange")

    st.caption("Arti label")
    for nama, fg, bg, arti in STATUS.values():
        st.markdown(f'<div style="background:{bg};color:{fg};border-radius:10px;padding:.4rem .7rem;'
                    f'margin-bottom:.3rem;font-size:.82rem"><b>{nama}</b> · {arti}</div>', unsafe_allow_html=True)

    st.divider()
    with open("sampel_data_mbg.csv", "rb") as f:
        st.download_button("Unduh sampel data uji", f, file_name="sampel_data_mbg.csv", mime="text/csv",
                           icon=":material/download:", use_container_width=True)

st.title("Audit Kepatuhan Distribusi MBG")
st.caption("Prediksi tingkat kepatuhan paket Makan Bergizi Gratis dari data dapur, gizi, dan logistik pengiriman")

tab1, tab2 = st.tabs([":material/edit_note: Input Satu Paket", ":material/table: Dari File CSV"])

# ---------- 4. Input satu data ----------
with tab1:
    with st.form("form"):
        data = {}
        kolom = st.columns(3)
        for i, c in enumerate(info["fitur_kategori"]):
            data[c] = kolom[i % 3].selectbox(NAMA_FITUR[c], info["pilihan_kategori"][c])
        kolom = st.columns(3)
        for i, c in enumerate(info["fitur_numerik"]):
            lo, hi, med = info["rentang_fitur"][c]
            bulat = all(float(v).is_integer() for v in (lo, hi, med))
            data[c] = kolom[i % 3].number_input(NAMA_FITUR[c], min_value=int(lo) if bulat else lo,
                                                max_value=int(hi) if bulat else hi,
                                                value=int(med) if bulat else med, step=1 if bulat else 0.1)
        kirim = st.form_submit_button("Prediksi", type="primary", icon=":material/psychology:",
                                      use_container_width=True)

    if kirim:
        baris = pd.DataFrame([data])
        tampil = st.columns(2) if pilihan == "Bandingkan keduanya" else [st.container()]
        if pilihan != "AI Model Service":
            with tampil[0]:
                pred, proba = prediksi_library(baris)
                kartu(f"Model Library · {info['nama_model']}", pred[0])
                st.bar_chart(pd.Series(proba[0], index=[STATUS[k][0] for k in model.classes_], name="Probabilitas"),
                             horizontal=True, height=200)
        if pilihan != "Model Library" and API_KEY:
            with tampil[-1]:
                with st.spinner("Menghubungi AI model service..."):
                    try:
                        kartu(f"AI Model Service · {cfg['model']}", prediksi_llm(json.dumps(data, default=lambda o: o.item())))
                    except KuotaHabis:
                        st.error("Kuota API habis atau server sibuk. Coba lagi beberapa saat lagi.")

# ---------- 5. Prediksi dari CSV ----------
with tab2:
    st.write("Unggah CSV dengan kolom yang sama seperti sampel data (unduh di sidebar). "
             "AI Model Service dibatasi 30 baris agar hemat kuota.")
    file = st.file_uploader("Pilih file CSV", type="csv")
    if file is not None:
        df = pd.read_csv(file)
        kurang = [c for c in FITUR if c not in df.columns]
        if kurang:
            st.error(f"Kolom berikut tidak ditemukan: {', '.join(kurang)}")
        else:
            st.dataframe(df, hide_index=True)
            if st.button(f"Prediksi {len(df)} paket", type="primary", icon=":material/psychology:"):
                if pilihan != "AI Model Service":
                    df["prediksi_library"] = prediksi_library(df)[0]
                if pilihan != "Model Library" and API_KEY:
                    df = df.head(30)
                    bar, hasil = st.progress(0.0), []
                    for i, (_, b) in enumerate(df.iterrows(), start=1):
                        try:
                            hasil.append(prediksi_llm(json.dumps(b[FITUR].to_dict(), default=lambda o: o.item())))
                        except KuotaHabis:
                            hasil.append("gagal (kuota)")
                        bar.progress(i / len(df))
                        time.sleep(5)               # jaga batas request per menit
                    df["prediksi_ai"] = hasil
                kol_pred = [c for c in ["prediksi_library", "prediksi_ai"] if c in df]
                warna = lambda v: f"color:{STATUS[v][1]};background:{STATUS[v][2]};font-weight:600" if v in STATUS else ""
                st.dataframe(df.style.map(warna, subset=kol_pred), hide_index=True)
                st.download_button("Unduh hasil prediksi", df.to_csv(index=False), file_name="hasil_prediksi_mbg.csv",
                                   mime="text/csv", icon=":material/download:")
