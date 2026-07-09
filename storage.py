#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
KALICI DEPOLAMA + E-POSTA BİLDİRİMİ (storage.py)
================================================================================
Talepler (lead) iki yerde saklanabilir:

  1) PostgreSQL  — DATABASE_URL ortam değişkeni varsa (Railway Postgres eklenince
     otomatik gelir). Kalıcıdır; deploy'lar arasında kaybolmaz.
  2) leads.jsonl — DATABASE_URL yoksa veya DB'ye yazılamazsa güvenli yedek.
     (Railway dosya sistemi kalıcı değildir; bu yalnız yerel geliştirme/yedek içindir.)

Her talepte ayrıca e-posta bildirimi gönderilir — SMTP ortam değişkenleri
tanımlıysa. Tanımlı değilse sessizce atlanır (uygulama asla çökmez).

Gerekli ortam değişkenleri (Railway → Variables):
  DATABASE_URL   Postgres eklenince otomatik (referans değişken olarak bağlayın)
  SMTP_HOST      örn. smtp.gmail.com
  SMTP_PORT      587 (varsayılan)
  SMTP_USER      gönderen e-posta adresi
  SMTP_PASS      uygulama şifresi (Gmail'de "App Password")
  NOTIFY_TO      bildirimlerin gideceği adres
  NOTIFY_FROM    (opsiyonel) görünen gönderen; boşsa SMTP_USER
  ADMIN_TOKEN    (opsiyonel) /api/leads listesini korumak için
================================================================================
"""
from __future__ import annotations

import json
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage
from typing import Any, Dict, List

LOG = logging.getLogger("storage")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEADS_FILE = os.path.join(BASE_DIR, "leads.jsonl")
DATABASE_URL = os.environ.get("DATABASE_URL", "").strip()

# psycopg2 yalnızca DB kullanılacaksa gerekli; yoksa dosya moduna düşeriz.
_pg = None
if DATABASE_URL:
    try:
        import psycopg2  # type: ignore
        _pg = psycopg2
    except Exception as e:  # pragma: no cover
        LOG.warning("psycopg2 yüklenemedi (%s) — dosya moduna düşülüyor.", e)
        _pg = None

DDL = """
CREATE TABLE IF NOT EXISTS leads (
    id          SERIAL PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    kind        TEXT NOT NULL DEFAULT 'talep',
    name        TEXT,
    company     TEXT,
    email       TEXT,
    phone       TEXT,
    message     TEXT,
    context     JSONB
);
"""


def _conn():
    return _pg.connect(DATABASE_URL, connect_timeout=6)


def db_enabled() -> bool:
    return bool(DATABASE_URL) and _pg is not None


def init_db() -> str:
    """Uygulama açılışında çağrılır. Asla exception fırlatmaz."""
    if not DATABASE_URL:
        return "DATABASE_URL tanımlı değil — talepler dosyaya yazılacak (kalıcı değil)."
    if _pg is None:
        return "psycopg2 kurulu değil — talepler dosyaya yazılacak."
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute(DDL)
            c.commit()
        return "PostgreSQL bağlı — 'leads' tablosu hazır."
    except Exception as e:
        return f"PostgreSQL bağlanamadı ({e}) — talepler dosyaya yazılacak."


def save_lead(rec: Dict[str, Any]) -> str:
    """Talebi kalıcı olarak kaydeder. Dönen değer: 'db' | 'file'."""
    if db_enabled():
        try:
            with _conn() as c:
                with c.cursor() as cur:
                    cur.execute(
                        "INSERT INTO leads (kind, name, company, email, phone, message, context) "
                        "VALUES (%s,%s,%s,%s,%s,%s,%s)",
                        (rec.get("kind", "talep"), rec.get("name"), rec.get("company"),
                         rec.get("email"), rec.get("phone"), rec.get("message"),
                         json.dumps(rec.get("context") or {}, ensure_ascii=False)),
                    )
                c.commit()
            return "db"
        except Exception as e:
            LOG.warning("DB'ye yazılamadı (%s) — dosyaya yazılıyor.", e)

    try:
        with open(LEADS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception as e:
        LOG.warning("Dosyaya da yazılamadı: %s", e)
    return "file"


def list_leads(limit: int = 200) -> List[Dict[str, Any]]:
    if db_enabled():
        try:
            with _conn() as c:
                with c.cursor() as cur:
                    cur.execute(
                        "SELECT id, created_at, kind, name, company, email, phone, message, context "
                        "FROM leads ORDER BY id DESC LIMIT %s", (limit,))
                    cols = [d[0] for d in cur.description]
                    rows = []
                    for r in cur.fetchall():
                        d = dict(zip(cols, r))
                        d["created_at"] = str(d["created_at"])
                        rows.append(d)
                    return rows
        except Exception as e:
            LOG.warning("DB okunamadı: %s", e)
    out: List[Dict[str, Any]] = []
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
    except FileNotFoundError:
        pass
    return out[-limit:][::-1]


# ------------------------------------------------------------------ E-POSTA
SMTP_HOST = os.environ.get("SMTP_HOST", "").strip()
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587") or 587)
SMTP_USER = os.environ.get("SMTP_USER", "").strip()
SMTP_PASS = os.environ.get("SMTP_PASS", "")
NOTIFY_TO = os.environ.get("NOTIFY_TO", "").strip()
NOTIFY_FROM = os.environ.get("NOTIFY_FROM", "").strip() or SMTP_USER


def email_configured() -> bool:
    return all([SMTP_HOST, SMTP_USER, SMTP_PASS, NOTIFY_TO])


def notify_email(rec: Dict[str, Any]) -> str:
    """Yeni talep gelince bildirim e-postası atar. Yapılandırılmamışsa atlar.
    Asla exception fırlatmaz — istek akışını bozmaz."""
    if not email_configured():
        return "atlandı (SMTP yapılandırılmamış)"
    try:
        kind = "Rapor indirme" if rec.get("kind") == "rapor_indirme" else "Yeni talep"
        msg = EmailMessage()
        msg["Subject"] = f"[AI Fizibilite] {kind}: {rec.get('name', '')} — {rec.get('company') or '-'}"
        msg["From"] = NOTIFY_FROM
        msg["To"] = NOTIFY_TO
        if rec.get("email"):
            msg["Reply-To"] = rec["email"]

        lines = [
            f"Tür       : {kind}",
            f"Ad Soyad  : {rec.get('name', '')}",
            f"Şirket    : {rec.get('company', '') or '-'}",
            f"E-posta   : {rec.get('email', '')}",
            f"Telefon   : {rec.get('phone', '') or '-'}",
            f"Mesaj     : {rec.get('message', '') or '-'}",
            f"Zaman     : {rec.get('ts', '')}",
        ]
        ctx = rec.get("context") or {}
        if ctx:
            lines += ["", "--- Fizibilite bağlamı ---"]
            for k, v in ctx.items():
                lines.append(f"{k}: {v}")
        msg.set_content("\n".join(lines))

        ctx_ssl = ssl.create_default_context()
        if SMTP_PORT == 465:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15, context=ctx_ssl) as s:
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as s:
                s.starttls(context=ctx_ssl)
                s.login(SMTP_USER, SMTP_PASS)
                s.send_message(msg)
        return "gönderildi"
    except Exception as e:
        LOG.warning("E-posta gönderilemedi: %s", e)
        return f"hata: {e}"


def db_ping() -> tuple[bool, str]:
    """DB'ye gerçekten bağlanılabiliyor mu? (health için)"""
    if not DATABASE_URL:
        return False, "DATABASE_URL tanımlı değil"
    if _pg is None:
        return False, "psycopg2 kurulu değil"
    try:
        with _conn() as c:
            with c.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
        return True, "bağlı"
    except Exception as e:
        return False, f"bağlanamadı: {e}"


def status() -> Dict[str, Any]:
    ok, detail = db_ping()
    lead_count = None
    if ok:
        try:
            with _conn() as c:
                with c.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM leads")
                    lead_count = cur.fetchone()[0]
        except Exception:
            pass
    return {
        "storage": "postgres" if ok else "dosya (leads.jsonl — kalıcı değil)",
        "database_connected": ok,
        "database_detail": detail,
        "lead_count": lead_count,
        "email_configured": email_configured(),
        "notify_to": NOTIFY_TO or None,
    }
