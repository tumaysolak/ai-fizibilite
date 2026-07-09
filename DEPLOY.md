# Yayına Alma (Hosting) Rehberi

Bu bir **Python (FastAPI + scipy)** uygulamasıdır → statik hosting (Netlify, Vercel,
GitHub Pages) **çalışmaz**. Gerçek bir Python sunucusu gerekir. Aşağıda kolaydan
zora doğru seçenekler var. Repoya eklenen `Dockerfile`, `render.yaml`, `Procfile`
hepsini tek-tık yapar.

## Önce: kodu GitHub'a atın
```bash
cd feasibility_platform
git init && git add . && git commit -m "AI fizibilite platformu"
# GitHub'da boş bir repo açıp:
git remote add origin https://github.com/KULLANICI/ai-fizibilite.git
git push -u origin main
```

## Seçenek 1 — Render.com  ⭐ (en kolay, ücretsiz başlangıç)
1. render.com → hesap açın, GitHub'ı bağlayın.
2. **New + → Blueprint** → repoyu seçin. `render.yaml` otomatik okunur.
   (Alternatif: **New + → Web Service**, Build: `pip install -r requirements.txt`,
   Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`.)
3. Deploy → birkaç dakikada `https://ai-fizibilite.onrender.com` gibi bir adres.
- Ücretsiz katman uykuya dalar (ilk istek ~30 sn), küçük demo için ideal.

## Seçenek 2 — Railway.app (çok kolay, cömert ücretsiz kredi)
1. railway.app → "Deploy from GitHub repo" → repoyu seçin.
2. Railway `Procfile`'ı otomatik kullanır; başka ayar gerekmez.
3. Settings → Networking → "Generate Domain" ile public URL alın.

## Seçenek 3 — Fly.io (Docker, global, ücretsiz katman)
```bash
# fly CLI kurulu olmalı: https://fly.io/docs/hands-on/install-flyctl/
fly launch          # Dockerfile'ı algılar, uygulama adı sorar
fly deploy
```

## Seçenek 4 — Google Cloud Run (Docker, ölçeklenir, kullandıkça öde)
```bash
gcloud run deploy ai-fizibilite --source . --region europe-west1 --allow-unauthenticated
```
`Dockerfile`'ı kullanır; trafik yokken sıfıra iner (maliyet ~0).

## Seçenek 5 — Kendi sunucunuz / VPS (Hetzner, DigitalOcean, AWS EC2)
En çok kontrol; kalıcı ve uyumaz. Türkiye/yakın bölge için Hetzner (Almanya) ucuz.
```bash
# Sunucuda:
git clone https://github.com/KULLANICI/ai-fizibilite.git && cd ai-fizibilite
pip install -r requirements.txt
# Üretimde birden çok worker + otomatik yeniden başlatma için:
pip install gunicorn
gunicorn -k uvicorn.workers.UvicornWorker -w 2 -b 0.0.0.0:8000 app:app
```
Önüne **nginx** koyup alan adı + HTTPS (Let's Encrypt / certbot) bağlayın.
Kalıcı çalışması için `systemd` servisi veya `pm2`/`supervisor` kullanın.

## Kendi alan adınız (ör. fizibilite.sirketiniz.com)
Yukarıdaki sağlayıcıların hepsi "Custom Domain" destekler: sağlayıcının verdiği
adres için DNS'te bir **CNAME** kaydı açmanız yeterli; HTTPS sertifikası otomatik gelir.

## Öneri
- **Sadece göstermek/paylaşmak** için → **Render** veya **Railway** (5 dakika).
- **Kurumsal, kalıcı, kendi alan adı** → **Cloud Run** ya da **VPS + nginx**.

## Veritabanı (kalıcı talepler) ve e-posta bildirimi

Varsayılan olarak talepler `leads.jsonl` dosyasına yazılır — **Railway'de bu dosya
her deploy'da silinir.** Kalıcı olması için PostgreSQL ekleyin.

### 1) Railway'e PostgreSQL ekleyin
1. Railway projenizde **New → Database → Add PostgreSQL**.
2. `ai-fizibilite` servisine gidin → **Variables** → **New Variable** →
   **Add Reference** → `Postgres` → `DATABASE_URL` seçin.
   (Böylece `DATABASE_URL` servise otomatik geçer.)
3. Servis yeniden deploy olur. Uygulama açılışta `leads` tablosunu **kendi oluşturur**.

### 2) E-posta bildirimi (her yeni talepte)
`ai-fizibilite` servisi → **Variables** → şu değişkenleri ekleyin:

| Değişken | Örnek | Not |
|---|---|---|
| `SMTP_HOST` | `smtp.gmail.com` | |
| `SMTP_PORT` | `587` | 465 kullanırsanız SSL'e otomatik geçer |
| `SMTP_USER` | `siz@gmail.com` | gönderen hesap |
| `SMTP_PASS` | *(uygulama şifresi)* | Gmail'de **App Password** — normal şifre değil |
| `NOTIFY_TO` | `siz@gmail.com` | bildirimlerin gideceği adres |
| `NOTIFY_FROM` | *(opsiyonel)* | boşsa `SMTP_USER` |
| `ADMIN_TOKEN` | *(opsiyonel)* | `/api/leads` listesini korur |

> **Gmail için:** hesabınızda 2 adımlı doğrulama açık olmalı; ardından
> Google Hesabı → Güvenlik → Uygulama şifreleri'nden 16 haneli bir şifre üretip
> `SMTP_PASS` olarak girin. Şifreyi yalnızca Railway arayüzüne siz girin.

SMTP değişkenleri tanımlı değilse e-posta **sessizce atlanır**; uygulama çalışmaya devam eder.

### 3) Doğrulama
```
https://<siteniz>/api/health
```
Beklenen çıktı:
```json
{"storage":"postgres","database_connected":true,"lead_count":0,"email_configured":true, ...}
```
Talepleri listelemek için (ADMIN_TOKEN tanımlıysa):
```
https://<siteniz>/api/leads?token=SIZIN_TOKENINIZ
```

### Neden hem DB hem dosya?
DB erişilemezse uygulama **çökmez**; talep dosyaya yazılır ve ayrıca sunucu
loglarına basılır. Böylece hiçbir talep kaybolmaz.

## Notlar
- Uygulama `PORT` ortam değişkenini otomatik okur (bulut sağlayıcılar bunu verir).
- Dış bağımlılık yok; internet gerektiren bir çağrı yapmaz — kapalı ağda da çalışır.
- Hesap motoru saf hesaplama olduğu için 1 küçük instance yüzlerce sorguyu rahat kaldırır.
