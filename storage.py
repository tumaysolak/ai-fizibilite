#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
TALEP DEPOLAMA VE E-POSTA BİLDİRİMİ (storage.py)
================================================================================
Talepler Postgres'e (varsa) veya JSON dosyasına kaydedilir.
E-posta bildirimi Gmail SMTP üzerinden gönderilir.
"""
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Any, List, Optional
from datetime import datetime

# ============================================================================
# SMTP YAPILANDIRMASI
# ============================================================================
SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD", "")
ADMIN_EMAIL = os.environ.get("ADMIN_EMAIL", "m.okayozaydin@gmail.com")

LEADS_FILE = "leads.json"


def email_configured() -> bool:
    """E-posta yapılandırması kontrol et."""
    return bool(SMTP_USER and SMTP_PASSWORD)


def init_db() -> str:
    """Depolama başlat (Postgres veya dosya)."""
    if not os.path.exists(LEADS_FILE):
        with open(LEADS_FILE, "w", encoding="utf-8") as f:
            json.dump([], f)
    return f"Dosya depolama başlatıldı: {LEADS_FILE}"


def save_lead(rec: Dict[str, Any]) -> str:
    """Talebi JSON dosyasına kaydet."""
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            leads = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        leads = []
    
    leads.append(rec)
    
    with open(LEADS_FILE, "w", encoding="utf-8") as f:
        json.dump(leads, f, ensure_ascii=False, indent=2)
    
    return LEADS_FILE


def list_leads() -> List[Dict[str, Any]]:
    """Tüm talepler listele."""
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []


def status() -> Dict[str, Any]:
    """Sistem durumu."""
    return {
        "status": "ok",
        "storage": "file",
        "email_configured": email_configured(),
        "admin_email": ADMIN_EMAIL,
        "timestamp": datetime.now().isoformat(),
    }


def notify_email(rec: Dict[str, Any]) -> None:
    """Talep oluşturulduğunda admin'e mail gönder."""
    if not email_configured():
        print(f"[email] SMTP yapılandırılmamış, mail gönderilemedi", flush=True)
        return
    
    try:
        # E-posta içeriğini oluştur
        subject = f"🚀 Yeni Talep: {rec.get('name', 'Bilinmiyor')}"
        
        # HTML body
        html_body = f"""
        <html>
            <head>
                <meta charset="utf-8">
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #007bff; color: white; padding: 20px; border-radius: 5px; }}
                    .content {{ background-color: #f9f9f9; padding: 20px; margin-top: 20px; border-radius: 5px; }}
                    .field {{ margin-bottom: 15px; }}
                    .label {{ font-weight: bold; color: #007bff; }}
                    .value {{ margin-top: 5px; padding: 10px; background-color: white; border-left: 3px solid #007bff; }}
                    .footer {{ margin-top: 20px; font-size: 12px; color: #666; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h2>🚀 Yeni Talep Alındı!</h2>
                    </div>
                    <div class="content">
                        <div class="field">
                            <div class="label">Ad Soyad:</div>
                            <div class="value">{rec.get('name', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">E-posta:</div>
                            <div class="value">{rec.get('email', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">Şirket:</div>
                            <div class="value">{rec.get('company', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">Telefon:</div>
                            <div class="value">{rec.get('phone', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">Mesaj:</div>
                            <div class="value">{rec.get('message', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">Talep Türü:</div>
                            <div class="value">{rec.get('kind', '-')}</div>
                        </div>
                        <div class="field">
                            <div class="label">Oluşturulma Tarihi:</div>
                            <div class="value">{rec.get('ts', '-')}</div>
                        </div>
                        {_format_context_html(rec.get('context', {}))}
                    </div>
                    <div class="footer">
                        <p>Bu e-posta AI Donanım Fizibilite Platformu tarafından otomatik olarak gönderilmiştir.</p>
                    </div>
                </div>
            </body>
        </html>
        """
        
        # E-posta gönder
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = SMTP_USER
        msg["To"] = ADMIN_EMAIL
        
        msg.attach(MIMEText(html_body, "html", "utf-8"))
        
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.send_message(msg)
        
        print(f"[email] ✅ Mail gönderildi: {ADMIN_EMAIL}", flush=True)
    
    except Exception as e:
        print(f"[email] ❌ Mail gönderilemedi: {e}", flush=True)


def _format_context_html(context: Dict[str, Any]) -> str:
    """Ek bilgileri HTML formatında döner."""
    if not context:
        return ""
    
    html = '<div class="field"><div class="label">Ek Bilgiler:</div>'
    for key, value in context.items():
        html += f'<div class="value"><strong>{key}:</strong> {value}</div>'
    html += '</div>'
    return html

