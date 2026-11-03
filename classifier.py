"""Klasifikasi tingkat kepatuhan distribusi MBG menggunakan LLM (AI model service).
Dipakai saat EVALUASI (2_evaluasi_llm.py) dan di WEB (app.py) agar perilakunya sama persis."""
import json
import re
import time

import openai

# AI service yang kompatibel dengan format OpenAI: base URL + nama API key
PROVIDER = {
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai/", "env_key": "GEMINI_API_KEY"},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "env_key": "OPENROUTER_API_KEY"},
}
MODEL_DEFAULT = {"gemini": "gemini-3.1-flash-lite", "openrouter": "google/gemma-4-31b-it:free"}

# nama fitur yang ditampilkan ke LLM (dan label form di web)
NAMA_FITUR = {
    "regional_cluster": "Klaster wilayah",
    "beneficiary_category": "Kategori penerima manfaat",
    "kitchen_daily_capacity_portions": "Kapasitas dapur SPPG (porsi/hari)",
    "raw_ingredient_cost_per_portion_idr": "Biaya bahan baku per porsi (Rp)",
    "protein_type": "Jenis protein",
    "estimated_calories_kcal": "Estimasi kalori (kkal)",
    "estimated_protein_grams": "Estimasi protein (gram)",
    "delivery_distance_km": "Jarak pengiriman (km)",
    "delivery_transport_mode": "Moda transportasi",
    "transit_time_minutes": "Waktu transit (menit)",
    "food_arrival_temperature_celsius": "Suhu makanan saat tiba (°C)",
    "plate_waste_percentage": "Sisa makanan di piring (%)",
    "student_attendance_rate_pct": "Tingkat kehadiran penerima (%)",
    "local_farmer_sourcing_pct": "Bahan dari petani lokal (%)",
}


class KuotaHabis(Exception):
    """Pemanggilan API terus gagal (mis. kuota free tier habis)."""


def load_config(path="config.json"):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def format_data(data):
    return "; ".join(f"{nama}: {data[kolom]}" for kolom, nama in NAMA_FITUR.items())


def build_messages(cfg, data):
    msgs = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot", []):
        msgs.append({"role": "user", "content": format_data(ex["data"])})
        msgs.append({"role": "assistant", "content": ex["label"]})
    msgs.append({"role": "user", "content": format_data(data)})
    return msgs


def _norm(teks):
    return re.sub(r"[^a-z]", "", (teks or "").lower())


def parse_label(output, labels):
    """'standard-compliant.' -> 'Standard-Compliant'. Tepat satu label disebut -> label itu, selain itu 'invalid'."""
    jawab = _norm(output)
    peta = {_norm(l): l for l in labels}
    if jawab in peta:
        return peta[jawab]
    cocok = [l for k, l in peta.items() if k in jawab]
    return cocok[0] if len(cocok) == 1 else "invalid"


def classify(client, cfg, data, max_retry=4):
    for percobaan in range(max_retry):
        try:
            resp = client.chat.completions.create(
                model=cfg["model"], temperature=cfg["temperature"], max_tokens=cfg["max_tokens"],
                messages=build_messages(cfg, data),
            )
            return parse_label(resp.choices[0].message.content, cfg["labels"])
        except openai.RateLimitError:
            time.sleep(15 * (percobaan + 1))                # kena batas kuota: tunggu lalu coba lagi
        except (openai.APIConnectionError, openai.InternalServerError, IndexError, TypeError):
            time.sleep(5)
    raise KuotaHabis(f"Gagal memanggil {cfg['model']} setelah {max_retry} percobaan")


if __name__ == "__main__":                                   # cek parse_label
    L = ["Exemplary-Gold-Standard", "Standard-Compliant", "Marginal-Quality-At-Risk",
         "Non-Compliant-Intervention-Required"]
    assert parse_label("Standard-Compliant", L) == "Standard-Compliant"
    assert parse_label("**non-compliant-intervention-required.**", L) == "Non-Compliant-Intervention-Required"
    assert parse_label("Label: Marginal-Quality-At-Risk", L) == "Marginal-Quality-At-Risk"
    assert parse_label("Aman", L) == "invalid"
    print("parse_label ok")
