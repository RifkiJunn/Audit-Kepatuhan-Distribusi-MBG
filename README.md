# Audit Kepatuhan Distribusi MBG

Aplikasi web untuk memprediksi tingkat kepatuhan paket distribusi **Makan Bergizi Gratis (MBG)** dari data dapur, gizi, dan logistik pengiriman. Ada dua model yang bisa dipilih, atau dibandingkan berdampingan.

**Demo:** https://audit-kepatuhan-distribusi-mbg.streamlit.app/

## Model

| Model | Cara kerja | F1 macro (200 data test yang sama) |
|---|---|---|
| Model Library | Random Forest dari scikit-learn, dipilih lewat 5-fold CV | 1,00 |
| AI Model Service | Gemini `gemini-3.1-flash-lite` dengan prompt few-shot + profil kelas | 0,72 |

Label: `Exemplary-Gold-Standard`, `Standard-Compliant`, `Marginal-Quality-At-Risk`, `Non-Compliant-Intervention-Required`.

## Fitur aplikasi

- **Input satu paket**: isi form, hasil prediksi langsung tampil (beserta probabilitas untuk model library).
- **Dari file CSV**: unggah banyak data sekaligus, lalu unduh hasil prediksinya. AI Model Service dibatasi 30 baris agar hemat kuota.
- **Sampel data** dapat diunduh dari sidebar (`sampel_data_mbg.csv`).

## Dataset

`indonesia_mbg_nationwide_enriched_2026.csv` dari Kaggle (13.000 baris). Dipakai 14 fitur; kolom yang membocorkan label dibuang.

## Menjalankan secara lokal

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Untuk AI Model Service, buat `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "isi-api-key"
```

File ini tidak di-commit. Di Streamlit Community Cloud, key diisi lewat **Settings → Secrets**.

## Isi folder

| File | Fungsi |
|---|---|
| `app.py` | Aplikasi Streamlit |
| `classifier.py` | Prompt dan pemanggilan LLM (sama dengan saat evaluasi) |
| `model_mbg.pkl`, `model_info.json` | Model Random Forest dan metadatanya |
| `config.json` | Konfigurasi prompt LLM terpilih dan hasil evaluasinya |
| `sampel_data_mbg.csv` | Sampel data untuk dicoba |
