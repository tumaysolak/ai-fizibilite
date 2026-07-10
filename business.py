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

from feasibility import (
    FeasibilityAnalyzer, FeasibilityInputs, HARDWARE_LIFE_MONTHS, MAX_PAYBACK_MONTHS,
)

WORKING_DAYS_PER_MONTH = 22


def discounted_payback(upfront: float, monthly_saving: float,
                       annual_rate_pct: float) -> tuple[Optional[float], bool]:
    """Sermaye maliyeti dahil geri ödeme süresi.

    Tasarrufların bugünkü değeri capex'i ne zaman karşılar? Aylık oran i>0 iken
    iskontolu tasarrufların toplamı en fazla (tasarruf / i) olabilir; capex bu
    tavanın üstündeyse yatırım MATEMATİKSEL OLARAK asla kendini ödemez.

    Dönüş: (geri_ödeme_ayı | None, asla_ödemez_mi)
    """
    if monthly_saving <= 0 or upfront <= 0:
        return None, True
    r = max(annual_rate_pct, 0.0) / 100.0
    if r <= 0:
        return upfront / monthly_saving, False
    i = (1 + r) ** (1 / 12.0) - 1
    if (monthly_saving / i) <= upfront:
        return None, True          # iskontolu tavan capex'i geçemiyor
    acc = 0.0
    for m in range(1, MAX_PAYBACK_MONTHS + 1):
        acc += monthly_saving / ((1 + i) ** m)
        if acc >= upfront:
            return float(m), False
    return None, True

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
PRIVACY_SENSITIVE = {"finans", "sigorta", "saglik", "ilac", "hukuk", "kamu",
                     "savunma", "telekom", "muhasebe", "arastirma"}
SECTORS = {
    "finans": "Finans / Bankacılık",
    "sigorta": "Sigorta",
    "saglik": "Sağlık",
    "ilac": "İlaç / Biyoteknoloji",
    "hukuk": "Hukuk",
    "muhasebe": "Muhasebe / Denetim",
    "danismanlik": "Danışmanlık",
    "kamu": "Kamu",
    "savunma": "Savunma / Havacılık",
    "eticaret": "E-ticaret",
    "perakende": "Perakende / Mağazacılık",
    "yazilim": "Yazılım / Teknoloji",
    "telekom": "Telekomünikasyon",
    "egitim": "Eğitim",
    "arastirma": "Araştırma / Ar-Ge",
    "uretim": "Üretim / Sanayi",
    "otomotiv": "Otomotiv",
    "insaat": "İnşaat / Gayrimenkul",
    "enerji": "Enerji / Elektrik",
    "lojistik": "Lojistik / Taşımacılık",
    "turizm": "Turizm / Otelcilik",
    "gida": "Gıda / İçecek",
    "tarim": "Tarım / Hayvancılık",
    "tekstil": "Tekstil / Hazır Giyim",
    "kimya": "Kimya / Petrokimya",
    "medya": "Medya / Ajans / Reklam",
    "spor": "Spor / Eğlence",
    "sivil": "STK / Dernek / Vakıf",
    "diger": "Diğer",
}

# Benimseme senaryoları. Çarpan doğrudan AKTİF KULLANICI SAYISINI ölçekler
# (hacim = kullanıcı × etkileşim × token × iş günü olduğundan, çarpan matematiksel
# olarak kullanıcı sayısı çarpanına eşittir). Bu yüzden senaryolar soyut bir
# "yoğunluk" değil, GERÇEK KULLANICI SAYISI olarak gösterilir — kullanıcı kendi
# girdiği sayıyı (Beklenen) doğrudan görür ve öneri o sayıya göre yapılır.
SCENARIOS = {
    "Temkinli":  {"factor": 0.5, "sub": "pilot ekip / sınırlı benimseme"},
    "Beklenen":  {"factor": 1.0, "sub": "girdiğiniz kullanıcı sayısı — baz alınan senaryo"},
    "Yoğun":     {"factor": 2.0, "sub": "kullanım tüm ekibe yayılırsa"},
}
BASE_SCENARIO = "Beklenen"

# Kalite katmanı seçimi -> USE_CASES anahtarı
QUALITY_KEYS = {"ekonomik": "budget", "dengeli": "primary", "premium": "premium"}

# ------------------------------------------------------------------ KENDİ SUNUCU AVANTAJLARI
# CoT (Sena geri bildirimi): Türkiye'de karar çoğu zaman salt maliyet kıyasıyla
# veriliyor ve kullanıcılar bulut ucuz görününce hemen vazgeçiyor. Oysa kendi
# sunucunun maliyet-dışı stratejik avantajları var. Bunları HER sonuçta öne
# çıkarıyoruz (sayıları çarpıtmadan; sadece kararın tek boyutlu olmasını önleyerek).
SELF_HOST_ADVANTAGES = [
    ["Veri sizde kalır", "Hiçbir veri dışarı / yurt dışına gitmez — KVKK ve müşteri gizliliği için en güvenli yol."],
    ["Kur riski yok", "Dolarlı abonelikler TL zayıfladıkça pahalanır; kendi donanımınız tek seferlik TL yatırımıdır."],
    ["Sınırsız kullanım", "Koltuk / token başına ödeme yok — ekip ve kullanım büyüdükçe aylık maliyet artmaz."],
    ["Tam kontrol", "Model, sürüm ve özelleştirme sizde; sağlayıcının fiyat veya politika değişikliğine bağlı değilsiniz."],
    ["Uzun vadede en ucuz", "Bulut/abonelik hacimle katlanır; yerel donanımın maliyeti sabit kalır ve zamanla öne geçer."],
]


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
        self.usd_try: float = float(kw.get("usd_try", 46.9))
        self.electricity_price_usd_per_kwh: float = float(kw.get("electricity_price_usd_per_kwh", 0.115))
        # Yıllık sermaye maliyeti (%): capex'i finanse etmenin / bağlamanın bedeli
        self.annual_capital_cost_pct: float = float(kw.get("annual_capital_cost_pct", 15.0))


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
        for sname, meta in SCENARIOS.items():
            factor = meta["factor"]
            volume = self._monthly_volume(uc, ai_users, factor)
            min_tps = volume / (SECONDS_PER_MONTH * TARGET_UTIL) if volume > 0 else None
            inp = FeasibilityInputs(
                model_name=model_name, context_length=uc["context"] or None,
                batch_size=8, monthly_volume=volume,
                electricity_price_usd_per_kwh=bi.electricity_price_usd_per_kwh,
                usd_try=bi.usd_try, cloud_rate_usd_per_1k=cloud_rate_arg,
                annual_capital_cost_pct=bi.annual_capital_cost_pct,
                min_tps=min_tps, horizon_months=36,
            )
            res = self.fa.analyze(inp)
            scn = self._friendly_scenario(sname, factor, volume, uc, res, bi)
            scn["sub"] = meta["sub"]
            # Senaryonun karşılık geldiği GERÇEK kullanıcı sayısı (çarpan = kullanıcı çarpanı)
            scn["users"] = max(1, round(ai_users * factor))
            scn["is_base"] = (sname == BASE_SCENARIO)
            scenarios.append(scn)
            if sname == BASE_SCENARIO:
                base_scn = scn

        # --- Yol karşılaştırması (Orta senaryo bazında) ---
        comparison = self._path_comparison(base_scn, bi)
        recommendation = self._recommend(base_scn, bi, uc, model_name, comparison)

        return {
            "inputs_echo": {
                "use_case": uc["label"], "unit": uc["unit"], "sector": SECTORS.get(bi.sector, bi.sector),
                "employees": bi.employees, "ai_users": ai_users, "quality": bi.quality,
                "adoption_pct": round(ai_users / max(bi.employees, 1) * 100),
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
            "self_host_advantages": SELF_HOST_ADVANTAGES,
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
            upfront = roi["upfront_investment_usd"]
            monthly_saving = baseline - roi["opex_month_usd"]
            out["monthly_saving_usd"] = round(monthly_saving, 2)
            out["monthly_saving_try"] = round(monthly_saving * bi.usd_try, 2)
            out["annual_saving_usd"] = round(monthly_saving * 12, 2)
            out["annual_saving_try"] = round(monthly_saving * 12 * bi.usd_try, 2)
            # İSKONTOLU geri ödeme (sermaye maliyeti dahil) — nominal değil.
            pb, never = discounted_payback(upfront, monthly_saving, bi.annual_capital_cost_pct)
            out["payback_months"] = round(pb, 1) if pb else None
            out["never_pays_back"] = never
            out["nominal_payback_months"] = (round(upfront / monthly_saving, 1)
                                             if monthly_saving > 0 else None)
            # Capex'in ufuk (36 ay) boyunca bağlanmasının finansman maliyeti
            r = max(bi.annual_capital_cost_pct, 0.0) / 100.0
            fin = upfront * ((1 + r) ** (36 / 12.0) - 1) if r > 0 else 0.0
            out["financing_cost_usd"] = round(fin, 2)
            out["financing_cost_try"] = round(fin * bi.usd_try, 2)
            out["invest_recommended"] = bool(out.get("capacity_ok")
                                             and pb is not None and pb <= HARDWARE_LIFE_MONTHS)
        else:
            out["monthly_saving_usd"] = None
            out["payback_months"] = None
            out["never_pays_back"] = True
            out["financing_cost_usd"] = None
            out["invest_recommended"] = False
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
        # 36 aylık toplam sahip olma maliyeti (TCO).
        # Yerel yola capex'in FİNANSMAN MALİYETİ de eklenir: o para ya krediyle
        # bulunur ya da başka bir yerde getiri sağlardı. Aksi halde yerel yol
        # haksız biçimde ucuz görünür.
        months = 36
        r = max(bi.annual_capital_cost_pct, 0.0) / 100.0
        financing = upfront * ((1 + r) ** (months / 12.0) - 1) if r > 0 else 0.0
        tco_sub = sub * months
        tco_cloud = cloud * months
        tco_local = upfront + local_opex * months + financing
        return {
            "months": months,
            "annual_capital_cost_pct": bi.annual_capital_cost_pct,
            "financing_cost_usd": round(financing, 2),
            "financing_cost_try": round(financing * rate, 2),
            "paths": [
                {"key": "abonelik", "label": "Mevcut abonelikte kal",
                 "monthly_usd": round(sub, 2), "upfront_usd": 0, "recommended": False,
                 "tco_usd": round(tco_sub, 2), "tco_try": round(tco_sub * rate, 2),
                 "note": "Sabit üyelik gideri; kişi/koltuk arttıkça büyür, veri sağlayıcıda kalır."},
                {"key": "bulut", "label": "Bulut API'ye geç (token başına)",
                 "monthly_usd": round(cloud, 2), "upfront_usd": 0, "recommended": False,
                 "tco_usd": round(tco_cloud, 2), "tco_try": round(tco_cloud * rate, 2),
                 "note": "Yatırım yok ama maliyet kullanımla artar; USD bazlı, kur riski taşır."},
                {"key": "kendi", "label": "Kendi sunucunu kur", "recommended": True,
                 "monthly_usd": round(local_opex, 2), "upfront_usd": round(upfront, 2),
                 "tco_usd": round(tco_local, 2), "tco_try": round(tco_local * rate, 2),
                 "financing_cost_usd": round(financing, 2),
                 "note": (f"Peşin yatırım + düşük işletme; veri sizde, kullanım sınırsız. "
                          f"36 ay toplamına capex'in ${financing:,.0f} finansman maliyeti dahildir "
                          f"(%{bi.annual_capital_cost_pct:.0f} yıllık sermaye maliyeti).")},
            ],
        }

    def _recommend(self, base, bi, uc, model_name, comparison) -> Dict[str, Any]:
        # CoT (Sena geri bildirimi): Öneri HER ZAMAN kendi sunucuyu merkeze alır.
        # Kararı tek boyutlu "hangisi bu ay ucuz" kıyasından çıkarıp; geri ödeme +
        # stratejik avantajlar (veri egemenliği, kur riski yok, sınırsız kullanım)
        # çerçevesine taşırız. Sayılar dürüst kalır — sadece anlatı yerelin lehine.
        adv = SELF_HOST_ADVANTAGES
        if base is None or not base.get("hardware"):
            return {"headline": "Bu iş yükü için katalogdaki tek kurulum yetersiz kalıyor.",
                    "detail": "Kullanımı bölmeyi veya daha küçük bir model katmanını deneyin.",
                    "tone": "warn", "self_host_advantages": adv}
        if not base.get("capacity_ok", True):
            return {"headline": "Seçilen kalite katmanı bu hacmi tek kurulumla karşılayamıyor.",
                    "detail": "Daha güçlü/çok donanım gerekir veya kullanımı bölün. 'Ekonomik' model katmanını deneyin.",
                    "tone": "warn", "self_host_advantages": adv}

        payback = base.get("payback_months")
        upfront = base.get("upfront_usd") or 0
        annual = base.get("annual_saving_usd")
        within = base.get("within_budget", True)
        hw = base.get("hardware")

        paths = {p["key"]: p for p in comparison.get("paths", [])}
        best_key = min(paths, key=lambda k: paths[k]["tco_usd"]) if paths else "kendi"
        self_cheapest = (best_key == "kendi")

        never = base.get("never_pays_back", False)
        # GÜVENLİK KURALI: Yatırım ancak donanım ömrü (36 ay) içinde kendini
        # ödüyorsa önerilir. Sermaye maliyeti dahil geri ödeme bunu aşıyorsa
        # (ör. 190 ay) veya hiç dönmüyorsa, dürüstçe "yatırım yapmayın" deriz.
        payback_ok = (payback is not None) and (not never) and (payback <= HARDWARE_LIFE_MONTHS)

        if not payback_ok:
            head = "Bu ölçekte kendi sunucu yatırımı kendini çıkarmıyor — şimdilik bulut/abonelik doğru tercih."
            det = []
            if never:
                det.append(f"~${upfront:,.0f} yatırımın getireceği tasarruf, sermaye maliyeti "
                           f"(%{bi.annual_capital_cost_pct:.0f}/yıl) hesaba katıldığında capex'i hiçbir zaman karşılamıyor.")
            else:
                det.append(f"Geri ödeme ~{payback:.0f} ay; donanımın ekonomik ömrü ise {HARDWARE_LIFE_MONTHS} ay. "
                           f"Yani yatırım kendini çıkarmadan donanım eskiyor.")
            det.append("Kullanımınız büyüdüğünde (daha çok kullanıcı veya daha yoğun kullanım) tablo hızla değişir — "
                       "aşağıdaki senaryolarda bunu görebilirsiniz.")
            if bi.sector in PRIVACY_SENSITIVE:
                det.append(f"Not: {SECTORS.get(bi.sector)} sektöründe veri gizliliği kritik olduğundan, "
                           f"maliyet dezavantajına rağmen kendi sunucunuzu stratejik gerekçeyle tercih edebilirsiniz.")
            return {"headline": head, "detail": " ".join(det), "tone": "warn",
                    "payback_months": payback, "upfront_usd": upfront, "annual_saving_usd": annual,
                    "best_path": best_key, "self_host_cheapest": self_cheapest,
                    "invest_recommended": False, "never_pays_back": never,
                    "self_host_advantages": adv}

        parts = [f"Önerimiz: <b>{model_name}</b> modelini <b>{hw}</b> üzerinde kendi sunucunuzda çalıştırın."]
        parts.append(f"~${upfront:,.0f} yatırım, sermaye maliyeti (%{bi.annual_capital_cost_pct:.0f}/yıl) dahil "
                     f"<b>{payback:.0f} ayda</b> kendini öder — donanım ömrünün ({HARDWARE_LIFE_MONTHS} ay) içinde.")
        if annual and annual > 0:
            parts.append(f"Sonrasında yıllık ~${annual:,.0f} (₺{annual*bi.usd_try:,.0f}) tasarruf.")
        if self_cheapest:
            parts.append("36 ayda en düşük toplam maliyet bu yolda.")
        tone = "good"

        if bi.budget_usd and not within:
            parts.append(f"⚠ Önerilen yatırım (${upfront:,.0f}) baremini (${bi.budget_usd:,.0f}) aşıyor; 'Ekonomik' katmana bakabilirsiniz.")
        if bi.sector in PRIVACY_SENSITIVE:
            parts.append(f"Ayrıca {SECTORS.get(bi.sector)} sektörü veri gizliliği açısından hassas — bu, kendi sunucuyu daha da güçlü kılar.")

        return {"headline": parts[0], "detail": " ".join(parts[1:]), "tone": tone,
                "payback_months": payback, "upfront_usd": upfront, "annual_saving_usd": annual,
                "best_path": best_key, "self_host_cheapest": self_cheapest,
                "invest_recommended": True, "never_pays_back": False,
                "self_host_advantages": adv}
