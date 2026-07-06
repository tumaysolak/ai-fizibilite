# AI Donanım Fizibilite Platformu

"Hangi AI modelini seçmeliyim, ne kadar donanım/sunucu yatırımı yapmalıyım ve
kaç ayda kendini öder?" sorularını cevaplayan bir web fizibilite platformu.
Düzeltilmiş MILP hesap motoru (`engine.py`) üzerine kuruludur.

## Kurulum & Çalıştırma

```bash
cd feasibility_platform
pip install -r requirements.txt
python app.py
```

Sonra tarayıcıda: **http://127.0.0.1:8000**

(Alternatif: `uvicorn app:app --reload`)

## İki mod

- **/** — **Kullanıcı dostu sihirbaz** (varsayılan). Teknik bilgi gerektirmez. 3 adımda iş
  sorularını sorar (kullanım amacı, ekip/sektör, mevcut AI harcaması & bütçe), arka planda
  otomatik olarak token hacmine + model önerisine çevirir, üç kullanım senaryosu ve
  "mevcut abonelik / bulut API / kendi sunucu" karşılaştırmasıyla dürüst bir öneri verir
  (36 aylık en düşük toplam maliyet hangisiyse onu önerir).
- **/pro** — **Teknik mod**. Token/bağlam/batch/min-TPS gibi teknik parametreleri doğrudan
  girmek isteyenler için ayrıntılı ekran.

## Ne yapar?

Bir model + aylık iş yükü (token/görsel hacmi, bağlam, batch) girersiniz; platform:

1. **Gerekli VRAM'i** hesaplar (ağırlık + KV-cache + aktivasyon + güvenlik marjı).
2. **En ucuz uygun donanımı** MILP (gerçek `scipy.optimize.milp`/HiGHS) ile seçer —
   VRAM, opsiyonel min-TPS ve maks-güç kısıtlarıyla; NVLink'siz kartlarda ≤4 GPU
   fiziksel gerçekçilik kısıtıyla.
3. **Hızı (TPS)** tahmin eder — decode adımında KV-cache okuması dahil (uzun bağlamda
   gerçekçi), tek-istek ve çoklu-GPU toplam throughput ayrı.
4. **Yatırımı** çıkarır: donanım + %sunucu ek maliyeti (şasi, ağ, kurulum).
5. **Aylık maliyeti** (enerji + ops) ve **bulut alternatifini** karşılaştırır.
6. **Geri ödeme süresini**, ROI'yi, başabaş hacmi ve kümülatif maliyet eğrisini verir.
7. Tüm tutarları **USD + TRY** gösterir.

## Dosya yapısı

```
feasibility_platform/
├── app.py              # FastAPI backend + web arayüzü servisi
├── engine.py           # Düzeltilmiş hesap motoru (VRAM, TPS, MILP, maliyet)
├── feasibility.py      # Yatırım/ROI/geri-ödeme katmanı
├── static/index.html   # Web arayüzü (tek dosya, Chart.js grafikli)
├── data/
│   ├── gpu_catalog.json    # GPU verisi (kod değişmeden düzenlenebilir)
│   └── model_catalog.json  # Model verisi
├── requirements.txt
└── README.md
```

Yeni GPU/model eklemek için ilgili JSON'a satır eklemek yeterli — kod değişmez.

## API

| Yöntem | Uç nokta | Açıklama |
|--------|----------|----------|
| GET | `/api/models` | Model kataloğu |
| GET | `/api/gpus` | GPU kataloğu |
| POST | `/api/analyze` | Fizibilite analizi (JSON gövde) |

Örnek:
```bash
curl -X POST http://127.0.0.1:8000/api/analyze -H 'Content-Type: application/json' \
  -d '{"model_name":"Meta Llama 3.1 (70B)","monthly_volume":50000000,"context_length":4096,"usd_try":34}'
```

## Orijinal koda göre düzeltilen hatalar

`engine.py` başındaki docstring'de numaralı liste var. Öne çıkanlar:
- **A1** Decode hızına KV-cache okuması eklendi → uzun bağlamda TPS artık gerçekçi düşüyor.
- **A2** Diffusion modelleri "token/sn" yerine "görsel/sn" birimiyle ele alınıyor.
- **A3** Aktivasyon belleği batch×seq×hidden ile ölçekleniyor.
- **B1** Sağlanamayan min-TPS/güç kısıtları artık **sessizce geçilmiyor**; sonuç işaretleniyor.
- **B2/B3** MILP ve heuristic aynı VRAM muhasebesini kullanıyor; NVLink cezası VRAM'e değil hıza uygulanıyor.

## Önemli not

Bu bir **analitik tahmin (hesaplama) modelidir**; sonuçlar donanım/model varsayımlarına
ve katsayılara dayanır. Gerçek benchmark ölçümleriniz varsa kalibrasyon ile motorun
teorik tahminini geçersiz kılabilirsiniz (`engine.load_calibration_file`). Fiyatlar
yaklaşık liste değerleridir; kendi tedarik fiyatlarınızı `data/gpu_catalog.json`'a girin.
