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

## Notlar
- Uygulama `PORT` ortam değişkenini otomatik okur (bulut sağlayıcılar bunu verir).
- Dış bağımlılık yok; internet gerektiren bir çağrı yapmaz — kapalı ağda da çalışır.
- Hesap motoru saf hesaplama olduğu için 1 küçük instance yüzlerce sorguyu rahat kaldırır.
