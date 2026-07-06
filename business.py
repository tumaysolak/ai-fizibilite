#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
İŞ MANTIĞI KATMANI (business.py)
================================================================================
Kullanıcıya TEKNİK sorular (token, batch, bağlam) yerine İŞ soruları sorar:
  • Neden AI kullanmak istiyorsun? (kullanım amacı)
  • Kaç kişi kullanacak? / şirket kaç kişi?
  • Hangi sektör?
  • Şu an AI aboneliklerine toplam aylık ne ödüyorsun?
  • Yatırım baremin (bütçe)?

Bu girdileri, düzeltilmiş hesap motorunun (engine.py / feasibility.py) anladığı
teknik parametrelere (aylık token hacmi, bağlam, önerilen model) ÇEVİRİR ve
birden fazla senaryo (Düşük / Orta / Yüksek kullanım) + yol karşılaştırması
(Mevcut abonelik / Bulut API / Kendi sunucu) üretir.
================================================================================
"""
from __future__ import annotations

from typing import Dict, Any, List, Optional

from feasibility import FeasibilityAnalyzer, FeasibilityInputs

WORKING_DAYS_PER_MONTH = 22

# ------------------------------------------------------------------ KULLANIM AMAÇLARI
# Her amaç, teknik parametrelere çeviri katsayılarını taşır:
#   tokens_per_query : bir etkileşimde işlenen ~toplam token (giriş+çıkış)
#   queries_per_user_day : aktif kullanıcı başına günlük etkileşim
#   context : tipik bağlam uzunluğu
#   primary/budget/premium : katalogdaki model adları (kalite katmanları)
#   unit : "token" | "image" | "audio_min"
USE_CASES: Dict[str, Dict[str, Any]] = {
    "musteri_destek": {
        "label": "Müşteri destek / chatbot",
        "desc": "Gelen soruları yanıtlayan destek asistanı",
        "unit": "token", "tokens_per_query": 700, "queries_per_user_day": 60, "context": 4096,
        "primary": "Meta Llama 3.1 (8B)", "budget": "Google Gemma 2 (2B)", "premium": "Meta Llama 3.1 (70B)",
        "cloud_rate_usd_per_1k": 0.010,
    },
    "kodlama": {
        "label": "Yazılım / kod asistanı",
        "desc": "Geliştiricilere kod önerisi, refactor, hata ayıklama",
        "unit": "token", "tokens_per_query": 1800, "queries_per_user_day": 40, "context": 8192,
        "primary": "Meta Llama 3.1 (70B)", "budget": "Mistral v0.3 (7B)", "premium": "Alibaba Qwen 2.5 (72B)",
        "cloud_rate_usd_per_1k": 0.012,
    },
    "dokuman_analiz": {
        "label": "Doküman analizi / RAG",
        "desc": "Uzun belgeleri okuyup özetleme, soru-cevap",
        "unit": "token", "tokens_per_query": 6000, "queries_per_user_day": 20, "context": 32768,
        "primary": "Alibaba Qwen 2.5 (72B)", "budget": "Meta Llama 3.1 (8B)", "premium": "Alibaba Qwen 2.5 (72B)",
        "cloud_rate_usd_per_1k": 0.011,
    },
    "icerik_uretim": {
        "label": "İçerik / pazarlama üretimi",
        "desc": "Blog, e-posta, sosyal medya metni üretimi",
        "unit": "token", "tokens_per_query": 1500, "queries_per_user_day": 25, "context": 4096,
        "primary": "Mixtral 8x7B (MoE)", "budget": "Mistral v0.3 (7B)", "premium": "Meta Llama 3.1 (70B)",
        "cloud_rate_usd_per_1k": 0.010,
    },
    "genel_asistan": {
        "label": "Genel ofis asistanı",
        "desc": "Günlük yazışma, özet, çeviri, beyin fırtınası",
        "unit": "token", "tokens_per_query": 800, "queries_per_user_day": 20, "context": 4096,
        "primary": "Meta Llama 3.1 (8B)", "budget": "Google Gemma 2 (2B)", "premium": "Meta Llama 3.1 (70B)",
        "cloud_rate_usd_per_1k": 0.010,
    },
    "gorsel_uretim": {
        "label": "Görsel üretim",
        "desc": "Pazarlama görselleri, ürün görselleri",
        "unit": "image", "images_per_user_day": 30, "context": 0,
        "primary": "Stable Diffusion XL", "budget": "Stable Diffusion XL", "premium": "Stable Diffusion XL",
        "cloud_rate_usd_per_image": 0.04,
    },
    "ses_metin": {
        "label": "Toplantı / ses → metin",
        "desc": "Toplantı kayıtlarını yazıya dökme, altyazı",
        "unit": "audio_min", "audio_min_per_user_day": 120, "tokens_per_audio_min": 200, "context": 1500,
        "primary": "OpenAI Whisper V3", "budget": "OpenAI Whisper V3", "premium": "OpenAI Whisper V3",
        "cloud_rate_usd_per_1k": 0.006,
    },
}

# ------------------------------------------------------------------ SEKTÖRLER
# Bazı sektörler veri gizliliği/regülasyon nedeniyle kendi sunucusuna yönelir.
PRIVACY_SENSITIVE = {"finans", "saglik", "hukuk", "kamu", "savunma"}
SECTORS = {
    "finans": "Finans / Bankacılık", "saglik": "Sağlık", "hukuk": "Hukuk",
    "kamu": "Kamu", "savunma": "Savunma", "eticaret": "E-ticaret / Perakende",
    "yazilim": "Yazılım / Teknoloji", "egitim": "Eğitim", "uretim": "Üretim / Sanayi",
    "medya": "Medya / Ajans", "diger": "Diğer",
}

# Kullanım yoğunluğu senaryoları (aktif kullanıcı başına etkileşim çarpanı)
SCENARIOS = {"Düşük": 0.5, "Orta": 1.0, "Yüksek": 2.0}

# Kalite katmanı seçimi -> USE_CASES anahtarı
QUALITY_KEYS = {"ekonomik": "budget", "dengeli": "primary", "premium": "premium"}


def catalog_for_ui() -> Dict[str, Any]:
    return {
        "use_cases": [{"key": k, "label": v["label"], "desc": v["desc"], "unit": v["unit"]}
                      for k, v in USE_CASES.items()],
        "sectors": [{"key": k, "label": v} for k, v in SECTORS.items()],
        "quality": [{"key": "ekonomik", "label": "Ekonomik (küçük model, en ucuz)"},
                    {"key": "dengeli", "label": "Dengeli (önerilen)"},
                    {"key": "premium", "label": "Premium (en yetenekli, büyük model)"}],
    }


class BusinessInputs:
    def __init__(self, **kw):
        self.use_case: str = kw.get("use_case", "genel_asistan")
        self.employees: int = int(kw.get("employees", 50))
        self.ai_users: Optional[int] = kw.get("ai_users")          # AI kullanacak kişi (yoksa tahmin)
        self.sector: str = kw.get("sector", "diger")
        self.current_ai_spend_usd_month: float = float(kw.get("current_ai_spend_usd_month", 0) or 0)
        self.budget_usd: Optional[float] = kw.get("budget_usd")     # yatırım baremi (capex)
        self.quality: str = kw.get("quality", "dengeli")
        self.usd_try: float = float(kw.get("usd_try", 34.0))
        self.electricity_price_usd_per_kwh: float = float(kw.get("electricity_price_usd_per_kwh", 0.12))


class BusinessFeasibility:
    def __init__(self, gpu_catalog, model_catalog):
        self.fa = FeasibilityAnalyzer(gpu_catalog, model_catalog)

    # ---- iş girdisi -> aylık hacim ----
    def _monthly_volume(self, uc: Dict[str, Any], ai_users: int, factor: float) -> float:
        unit = uc["unit"]
        if unit == "token":
            return ai_users * uc["queries_per_user_day"] * uc["tokens_per_query"] * WORKING_DAYS_PER_MONTH * factor
        if unit == "image":
            return ai_users * uc["images_per_user_day"] * WORKING_DAYS_PER_MONTH * factor
        if unit == "audio_min":
            mins = ai_users * uc["audio_min_per_user_day"] * WORKING_DAYS_PER_MONTH * factor
            return mins * uc["tokens_per_audio_min"]   # tokene çevir
        return 0.0

    def _estimate_ai_users(self, bi: BusinessInputs) -> int:
        if bi.ai_users:
            return max(1, int(bi.ai_users))
        # Sektöre göre kaba benimseme oranı
        adoption = 0.6 if bi.sector in ("yazilim", "medya", "eticaret") else 0.35
        return max(1, round(bi.employees * adoption))

    def _cloud_rate(self, uc: Dict[str, Any]):
        return uc.get("cloud_rate_usd_per_1k"), uc.get("cloud_rate_usd_per_image")

    def analyze(self, bi: BusinessInputs) -> Dict[str, Any]:
        uc = USE_CASES.get(bi.use_case)
        if uc is None:
            return {"error": f"Bilinmeyen kullanım amacı: {bi.use_case}"}

        ai_users = self._estimate_ai_users(bi)
        model_name = uc[QUALITY_KEYS.get(bi.quality, "primary")]
        rate_1k, rate_img = self._cloud_rate(uc)
        cloud_rate_arg = rate_1k if uc["unit"] != "image" else rate_img

        # --- Üç kullanım senaryosu ---
        # Donanım yalnız VRAM'e değil, aylık hacmi GERÇEKTEN üretebilecek
        # THROUGHPUT'a göre de boyutlanır: gerekli min-TPS'i hacimden türetip
        # MILP'e kısıt olarak veriyoruz (target %70 kullanım, headroom bırakır).
        TARGET_UTIL = 0.70
        SECONDS_PER_MONTH = 730.0 * 3600.0
        scenarios = []
        base_scn = None
        for sname, factor in SCENARIOS.items():
            volume = self._monthly_volume(uc, ai_users, factor)
            min_tps = volume / (SECONDS_PER_MONTH * TARGET_UTIL) if volume > 0 else None
            inp = FeasibilityInputs(
                model_name=model_name, context_length=uc["context"] or None,
                batch_size=8, monthly_volume=volume,
                electricity_price_usd_per_kwh=bi.electricity_price_usd_per_kwh,
                usd_try=bi.usd_try, cloud_rate_usd_per_1k=cloud_rate_arg,
                min_tps=min_tps, horizon_months=36,
            )
            res = self.fa.analyze(inp)
            scn = self._friendly_scenario(sname, factor, volume, uc, res, bi)
            scenarios.append(scn)
            if sname == "Orta":
                base_scn = scn

        # --- Yol karşılaştırması (Orta senaryo bazında) ---
        comparison = self._path_comparison(base_scn, bi)
        recommendation = self._recommend(base_scn, bi, uc, model_name, comparison)

        return {
            "inputs_echo": {
                "use_case": uc["label"], "unit": uc["unit"], "sector": SECTORS.get(bi.sector, bi.sector),
                "employees": bi.employees, "ai_users": ai_users, "quality": bi.quality,
                "recommended_model": model_name,
                "current_ai_spend_usd_month": bi.current_ai_spend_usd_month,
                "budget_usd": bi.budget_usd,
            },
            "privacy_note": (f"{SECTORS.get(bi.sector)} sektörü veri gizliliği açısından hassastır; "
                             f"veriyi dışarı göndermeyen KENDİ SUNUCUNUZDA barındırma güçlü bir avantaj sağlar."
                             if bi.sector in PRIVACY_SENSITIVE else None),
            "scenarios": scenarios,
            "comparison": comparison,
            "recommendation": recommendation,
        }

    # ---- bir senaryoyu sadeleştir ----
    def _friendly_scenario(self, sname, factor, volume, uc, res, bi) -> Dict[str, Any]:
        unit = uc["unit"]
        roi = res.get("roi")
        config = res.get("config")
        vol_label = self._volume_label(volume, unit, uc)
        out = {
            "name": sname, "factor": factor, "monthly_volume": round(volume),
            "volume_label": vol_label,
            "hardware": config["label"] if config else None,
            "upfront_usd": roi["upfront_investment_usd"] if roi else None,
            "upfront_try": roi["upfront_investment_try"] if roi else None,
            "local_opex_month_usd": roi["opex_month_usd"] if roi else None,
            "cloud_month_usd": roi["cloud_month_usd"] if roi else None,
            "capacity_ok": roi["capacity_ok"] if roi else None,
            "tps": res.get("throughput", {}).get("aggregate"),
            "unit": res.get("unit"),
        }
        # Geri ödeme: "kendin kurmasan ne öderdin?" = mevcut abonelik ile bulut
        # API'nin UCUZ OLANI (mantıklı alternatif). Böylece bulut zaten daha
        # ucuzsa yerel yatırım haklı çıkarılmaz.
        cloud_m = roi["cloud_month_usd"] if roi else 0
        sub_m = bi.current_ai_spend_usd_month
        if sub_m > 0 and cloud_m > 0:
            baseline = min(sub_m, cloud_m)
        else:
            baseline = sub_m if sub_m > 0 else cloud_m
        out["baseline_month_usd"] = round(baseline, 2)
        if roi and roi["upfront_investment_usd"] and baseline > 0:
            monthly_saving = baseline - roi["opex_month_usd"]
            out["monthly_saving_usd"] = round(monthly_saving, 2)
            out["monthly_saving_try"] = round(monthly_saving * bi.usd_try, 2)
            out["payback_months"] = round(roi["upfront_investment_usd"] / monthly_saving, 1) if monthly_saving > 0 else None
            out["annual_saving_usd"] = round(monthly_saving * 12, 2)
            out["annual_saving_try"] = round(monthly_saving * 12 * bi.usd_try, 2)
        else:
            out["monthly_saving_usd"] = None
            out["payback_months"] = None
        out["within_budget"] = (bi.budget_usd is None or (roi and roi["upfront_investment_usd"] and roi["upfront_investment_usd"] <= bi.budget_usd))
        return out

    def _volume_label(self, volume, unit, uc) -> str:
        if unit == "image":
            return f"{volume/1e3:.0f}K görsel/ay"
        if unit == "audio_min":
            mins = volume / uc["tokens_per_audio_min"]
            return f"{mins/60:.0f} saat ses/ay"
        return f"{volume/1e6:.1f}M token/ay"

    # ---- 3 yol: mevcut abonelik / bulut API / kendi sunucu ----
    def _path_comparison(self, base: Dict[str, Any], bi: BusinessInputs) -> Dict[str, Any]:
        if base is None:
            return {}
        sub = bi.current_ai_spend_usd_month
        cloud = base.get("cloud_month_usd") or 0
        local_opex = base.get("local_opex_month_usd") or 0
        upfront = base.get("upfront_usd") or 0
        rate = bi.usd_try
        # 36 aylık toplam sahip olma maliyeti (TCO)
        months = 36
        tco_sub = sub * months
        tco_cloud = cloud * months
        tco_local = upfront + local_opex * months
        return {
            "months": months,
            "paths": [
                {"key": "abonelik", "label": "Mevcut abonelikte kal",
                 "monthly_usd": round(sub, 2), "upfront_usd": 0,
                 "tco_usd": round(tco_sub, 2), "tco_try": round(tco_sub * rate, 2),
                 "note": "Bugün ödediğin sabit üyelik gideri (kullanım artınca koltuk başına artabilir)."},
                {"key": "bulut", "label": "Bulut API'ye geç (token başına)",
                 "monthly_usd": round(cloud, 2), "upfront_usd": 0,
                 "tco_usd": round(tco_cloud, 2), "tco_try": round(tco_cloud * rate, 2),
                 "note": "Yatırım yok; maliyet kullanımla doğru orantılı artar."},
                {"key": "kendi", "label": "Kendi sunucunu kur (önerilen)",
                 "monthly_usd": round(local_opex, 2), "upfront_usd": round(upfront, 2),
                 "tco_usd": round(tco_local, 2), "tco_try": round(tco_local * rate, 2),
                 "note": "Peşin yatırım + düşük aylık işletme; hacim büyüdükçe en avantajlısı."},
            ],
        }

    def _recommend(self, base, bi, uc, model_name, comparison) -> Dict[str, Any]:
        if base is None or not base.get("hardware"):
            return {"headline": "Bu iş yükü için katalogdaki donanım yetersiz kalıyor — bulut API önerilir.",
                    "detail": "Kullanımı bölmeyi veya daha küçük bir model katmanını deneyin.", "tone": "warn"}
        if not base.get("capacity_ok", True):
            return {"headline": "Seçilen kalite katmanı bu hacmi karşılayamıyor.",
                    "detail": "Daha güçlü/çok donanım gerekir veya kullanımı bölün. 'Ekonomik' model katmanını deneyin.",
                    "tone": "warn"}

        payback = base.get("payback_months")
        upfront = base.get("upfront_usd")
        annual = base.get("annual_saving_usd")
        within = base.get("within_budget", True)

        # 36 aylık TCO'ya göre EN UCUZ yolu bul — öneri buna göre verilir (tutarlılık).
        paths = {p["key"]: p for p in comparison.get("paths", [])}
        best_key = min(paths, key=lambda k: paths[k]["tco_usd"]) if paths else "kendi"
        labels = {"abonelik": "mevcut aboneliğinizde kalmak",
                  "bulut": "bulut API'ye geçmek", "kendi": "kendi sunucunuzu kurmak"}

        if best_key == "kendi":
            parts = [f"Öneri: <b>{model_name}</b> modelini <b>{base['hardware']}</b> üzerinde kendi sunucunuzda çalıştırın."]
            if payback:
                parts.append(f"~${upfront:,.0f} yatırım, mantıklı alternatife göre ~<b>{payback:.0f} ayda</b> kendini öder.")
            if annual and annual > 0:
                parts.append(f"Yıllık ~${annual:,.0f} (₺{annual*bi.usd_try:,.0f}) tasarruf; 36 ayda en düşük toplam maliyet bu yolda.")
            tone = "good" if (payback and payback <= 24) else "good"
            if bi.budget_usd and not within:
                parts.append(f"⚠ Önerilen yatırım (${upfront:,.0f}) baremini (${bi.budget_usd:,.0f}) aşıyor; 'Ekonomik' katmana bakın.")
                tone = "warn"
        else:
            # Kendi sunucu 36 ayda en ucuz DEĞİL → dürüst öneri
            be = base_break_even = None
            headline = (f"Bu kullanım seviyesinde en ekonomik yol: <b>{labels[best_key]}</b>."
                        if best_key != "kendi" else "")
            parts = [headline,
                     f"Kendi sunucu kurulumu (${upfront:,.0f} yatırım) bu hacimde 36 ay içinde kendini çıkarmıyor; "
                     f"kullanımınız belirgin şekilde artarsa yeniden değerlendirin."]
            tone = "warn"
            if bi.sector in PRIVACY_SENSITIVE:
                parts.append("Yine de sektörünüz veri gizliliği açısından hassas — veriyi dışarı vermemek için "
                             "kendi sunucunuz stratejik olarak tercih edilebilir (maliyet biraz yüksek olsa da).")
            return {"headline": parts[0], "detail": " ".join([p for p in parts[1:] if p]), "tone": tone,
                    "payback_months": payback, "upfront_usd": upfront, "annual_saving_usd": annual,
                    "best_path": best_key}

        if bi.sector in PRIVACY_SENSITIVE:
            parts.append("Sektörünüz veri gizliliği açısından hassas olduğundan kendi sunucunuzda barındırma ayrıca stratejik avantaj sağlar.")
        return {"headline": parts[0], "detail": " ".join(parts[1:]), "tone": tone,
                "payback_months": payback, "upfront_usd": upfront, "annual_saving_usd": annual,
                "best_path": best_key}
