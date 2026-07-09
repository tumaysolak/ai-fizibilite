#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
AI DONANIM FİZİBİLİTE PLATFORMU — FastAPI backend (app.py)
================================================================================
Çalıştırma:
    pip install -r requirements.txt
    python app.py
    # veya: uvicorn app:app --reload
Sonra tarayıcıda:  http://127.0.0.1:8000

Uç noktalar:
    GET  /                 → web arayüzü (static/index.html)
    GET  /api/models       → model kataloğu
    GET  /api/gpus         → GPU kataloğu
    POST /api/analyze      → fizibilite analizi (JSON gövde: FeasibilityInputs)
================================================================================
"""
from __future__ import annotations

import os
import json
import time
from typing import Optional, Any, Dict

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import load_gpu_catalog, load_model_catalog
from feasibility import FeasibilityAnalyzer, FeasibilityInputs
from business import BusinessFeasibility, BusinessInputs, catalog_for_ui
import storage

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = FastAPI(title="AI Donanım Fizibilite Platformu", version="1.0")

GPU_CATALOG = load_gpu_catalog()
MODEL_CATALOG = load_model_catalog()
ANALYZER = FeasibilityAnalyzer(GPU_CATALOG, MODEL_CATALOG)
BUSINESS = BusinessFeasibility(GPU_CATALOG, MODEL_CATALOG)

ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "").strip()


@app.on_event("startup")
def _startup():
    print("[storage]", storage.init_db(), flush=True)
    print("[storage] e-posta bildirimi:",
          "AÇIK" if storage.email_configured() else "KAPALI (SMTP değişkenleri tanımlı değil)", flush=True)


def _record_lead(rec: Dict[str, Any], bg: BackgroundTasks) -> str:
    """Talebi kalıcı olarak kaydeder, e-postayı arka planda gönderir (isteği bekletmez)."""
    rec["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    where = storage.save_lead(rec)
    print(f"=== YENİ KAYIT ({where}) ===", json.dumps(rec, ensure_ascii=False), flush=True)
    bg.add_task(storage.notify_email, rec)
    return where


class AnalyzeRequest(BaseModel):
    model_name: str
    context_length: Optional[int] = None
    batch_size: int = 8
    monthly_volume: float = 50_000_000
    electricity_price_usd_per_kwh: float = 0.12
    pue: float = 1.4
    usd_try: float = 34.0
    server_overhead_pct: float = 0.35
    fixed_monthly_ops_usd: float = 0.0
    cloud_rate_usd_per_1k: Optional[float] = None
    min_tps: Optional[float] = None
    max_power_watts: Optional[float] = None
    carbon_intensity_g_per_kwh: Optional[float] = None
    horizon_months: int = 36
    always_on: bool = False


@app.get("/api/models")
def api_models():
    return [
        {
            "name": m.name, "use_case": m.use_case, "deployment": m.deployment.name,
            "architecture": m.architecture.name, "precision": m.quant().label,
            "params_billions": m.params_billions,
            "default_context_length": m.default_context_length,
            "cloud_rate_usd_per_1k": m.cloud_token_cost_usd_per_1k, "notes": m.notes,
        }
        for m in MODEL_CATALOG
    ]


@app.get("/api/gpus")
def api_gpus():
    return [
        {
            "name": g.name, "tier": g.tier.name, "vram_gb": g.vram_gb,
            "memory_bandwidth_gbps": g.memory_bandwidth_gbps, "fp16_tflops": g.fp16_tflops,
            "price_usd": g.price_usd, "power_watts": g.power_watts,
            "supports_nvlink": g.supports_nvlink,
        }
        for g in GPU_CATALOG
    ]


@app.post("/api/analyze")
def api_analyze(req: AnalyzeRequest):
    inp = FeasibilityInputs(**req.model_dump())
    result = ANALYZER.analyze(inp)
    return JSONResponse(result)


class QuickRequest(BaseModel):
    use_case: str = "genel_asistan"
    employees: int = 50
    ai_users: Optional[int] = None
    sector: str = "diger"
    current_ai_spend_usd_month: float = 0.0
    budget_usd: Optional[float] = None
    quality: str = "dengeli"
    usd_try: float = 34.0
    electricity_price_usd_per_kwh: float = 0.12


@app.get("/api/usecases")
def api_usecases():
    return catalog_for_ui()


@app.post("/api/quick")
def api_quick(req: QuickRequest):
    bi = BusinessInputs(**req.model_dump())
    return JSONResponse(BUSINESS.analyze(bi))


class LeadRequest(BaseModel):
    name: str
    company: Optional[str] = ""
    email: str
    phone: Optional[str] = ""
    message: Optional[str] = ""
    context: Optional[Dict[str, Any]] = None


@app.post("/api/lead")
def api_lead(req: LeadRequest, bg: BackgroundTasks):
    """Kullanıcı talebini Postgres'e (yoksa dosyaya) kaydeder ve e-posta bildirimi atar."""
    rec = req.model_dump()
    rec["kind"] = "talep"
    _record_lead(rec, bg)
    return {"ok": True, "message": "Talebiniz alındı. En kısa sürede sizinle iletişime geçeceğiz."}


@app.get("/api/health")
def api_health():
    """Depolama ve e-posta yapılandırmasını gösterir (yayın sonrası doğrulama için)."""
    return storage.status()


@app.get("/api/test-email")
def api_test_email(token: str = ""):
    """SMTP ayarlarını canlıda test eder: gerçekten mail atmayı dener ve
    başarısızsa HATANIN KENDİSİNİ döner (loglara bakmaya gerek kalmaz)."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN tanımlı değil; bu uç nokta kapalı.")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Geçersiz token.")
    if not storage.email_configured():
        return {"ok": False, "email_configured": False,
                "hint": "SMTP_HOST / SMTP_USER / SMTP_PASS(WORD) / NOTIFY_TO(ADMIN_EMAIL) değişkenlerini kontrol edin.",
                "gorulen": {"host": bool(storage.SMTP_HOST), "user": bool(storage.SMTP_USER),
                            "pass": bool(storage.SMTP_PASS), "to": storage.NOTIFY_TO or None}}
    result = storage.notify_email({
        "kind": "test", "name": "Test Gönderimi", "company": "AI Fizibilite",
        "email": storage.NOTIFY_TO, "phone": "-", "message": "Bu bir SMTP test e-postasıdır.",
        "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "context": {"kaynak": "/api/test-email"},
    })
    return {"ok": result == "gönderildi", "sonuc": result, "gonderildi_adres": storage.NOTIFY_TO}


@app.get("/api/leads")
def api_leads(token: str = ""):
    """Gelen talepleri listeler. ADMIN_TOKEN tanımlıysa korumalıdır."""
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail="ADMIN_TOKEN tanımlı değil; bu uç nokta kapalı.")
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Geçersiz token.")
    return storage.list_leads()


class ReportRequest(BaseModel):
    """Kurumsal rapor indirme: kullanıcı bilgileri (lead) + analiz girdileri."""
    name: str
    email: str
    company: Optional[str] = ""
    phone: Optional[str] = ""
    inputs: QuickRequest


def _csv_escape(v: Any) -> str:
    s = "" if v is None else str(v)
    if any(c in s for c in [",", '"', "\n"]):
        s = '"' + s.replace('"', '""') + '"'
    return s


def _row(*cells) -> str:
    return ",".join(_csv_escape(c) for c in cells)


def _build_csv(d: Dict[str, Any], lead: Dict[str, str]) -> str:
    """Analiz sonucunu kurumsal raporlama için düz CSV veri setine çevirir.
    Bölümler: Girdiler / Öneri / Yol karşılaştırması / Senaryolar."""
    echo = d.get("inputs_echo", {})
    rec = d.get("recommendation", {})
    L: list[str] = []
    L.append(_row("bolum", "alan", "deger", "birim"))

    L.append(_row("Rapor", "olusturulma", time.strftime("%Y-%m-%d %H:%M:%S"), ""))
    L.append(_row("Rapor", "talep_eden", lead.get("name", ""), ""))
    L.append(_row("Rapor", "sirket", lead.get("company", ""), ""))
    L.append(_row("Rapor", "eposta", lead.get("email", ""), ""))

    for k, label in [("use_case", "kullanim_amaci"), ("sector", "sektor"),
                     ("employees", "toplam_calisan"), ("ai_users", "ai_kullanici_sayisi"),
                     ("adoption_pct", "benimseme_orani"), ("recommended_model", "onerilen_model"),
                     ("current_ai_spend_usd_month", "mevcut_aylik_ai_harcamasi_usd")]:
        L.append(_row("Girdiler", label, echo.get(k, ""), "%" if k == "adoption_pct" else ""))

    L.append(_row("Oneri", "baslik", _strip_html(rec.get("headline", "")), ""))
    L.append(_row("Oneri", "onerilen_yol", rec.get("best_path", ""), ""))
    L.append(_row("Oneri", "yatirim_usd", rec.get("upfront_usd", ""), "USD"))
    L.append(_row("Oneri", "geri_odeme", rec.get("payback_months", ""), "ay"))
    L.append(_row("Oneri", "yillik_tasarruf_usd", rec.get("annual_saving_usd", ""), "USD"))

    for p in d.get("comparison", {}).get("paths", []):
        L.append(_row("Yol_36ay", p.get("label", ""), p.get("tco_usd", ""), "USD toplam"))
        L.append(_row("Yol_36ay_aylik", p.get("label", ""), p.get("monthly_usd", ""), "USD/ay"))
        L.append(_row("Yol_36ay_pesin", p.get("label", ""), p.get("upfront_usd", ""), "USD pesin"))

    for s in d.get("scenarios", []):
        n = s.get("name", "")
        L.append(_row("Senaryo_kullanici", n, s.get("users", ""), "kisi"))
        L.append(_row("Senaryo_hacim", n, s.get("monthly_volume", ""), "birim/ay"))
        L.append(_row("Senaryo_donanim", n, s.get("hardware", ""), ""))
        L.append(_row("Senaryo_yatirim", n, s.get("upfront_usd", ""), "USD"))
        L.append(_row("Senaryo_aylik_isletme", n, s.get("local_opex_month_usd", ""), "USD/ay"))
        L.append(_row("Senaryo_geri_odeme", n, s.get("payback_months", ""), "ay"))

    return "\n".join(L) + "\n"


def _strip_html(s: str) -> str:
    import re
    return re.sub(r"<[^>]+>", "", s or "")


@app.post("/api/report")
def api_report(req: ReportRequest, bg: BackgroundTasks):
    """Kullanıcı bilgilerini kaydeder (+e-posta bildirimi) ve analiz sonucunu CSV veri seti olarak döner."""
    lead = {"name": req.name, "email": req.email, "company": req.company, "phone": req.phone}
    rec = dict(lead)
    rec["kind"] = "rapor_indirme"
    rec["context"] = {"use_case": req.inputs.use_case, "ai_users": req.inputs.ai_users,
                      "employees": req.inputs.employees, "sector": req.inputs.sector}
    _record_lead(rec, bg)

    bi = BusinessInputs(**req.inputs.model_dump())
    result = BUSINESS.analyze(bi)
    if result.get("error"):
        return JSONResponse(result, status_code=400)

    csv_text = _build_csv(result, lead)
    fname = "ai-fizibilite-raporu.csv"
    return Response(content="﻿" + csv_text, media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/pro")
def pro():
    return FileResponse(os.path.join(STATIC_DIR, "pro.html"))


# static klasörünü de sun (ileride görsel/asset için)
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    # Bulut sağlayıcıları PORT ortam değişkeni verir; yerelde 8000'e düşer.
    port = int(os.environ.get("PORT", "8000"))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("app:app", host=host, port=port, reload=False)
