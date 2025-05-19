# Yapay Zeka ile Veri Analizi Uygulaması

Bu uygulama, Python'da geliştirilmiş veri analizi ve yapay zeka fonksiyonlarını kullanıcı dostu bir arayüz üzerinden erişilebilir hale getirmek için tasarlanmıştır.

## Özellikler

- **Zaman Serisi Analizi ve Satış Tahmini**: ARIMA ve ML modelleri ile gelecek satışları tahmin edin
- **Müşteri Segmentasyonu**: K-means algoritması ile müşterilerinizi segmentlere ayırın
- **Anomali Tespiti**: Isolation Forest algoritması ile anormal müşteri davranışlarını tespit edin
- **RFM Analizi**: Müşterilerinizi değerlerine göre segmentlere ayırın
- **Duygu Analizi**: Müşteri yorumlarını analiz edin
- **Maliyet ve Karlılık Analizi**: Ürünlerinizin karlılığını analiz edin
- **Trend Analizi**: Pazar trendlerini belirleyin ve gelecek eğilimleri tahmin edin

## Kurulum

# Repo'yu klonlayın
```bash
git clone https://github.com/kullaniciadi/yapay-zeka-veri-analizi.git
cd yapay-zeka-veri-analizi
```
# Gerekli paketleri yükleyin
```bash
pip install -r requirements.txt
```

# Uygulamayı çalıştırın
```bash
streamlit run app.py
```

Kullanım

Ana sayfada kullanılabilir analiz modüllerini görüntüleyin
Sekmelerden bir modül seçin:

Satış Tahmini: Zaman serisi analizi ve gelecek satış tahminleri yapın
Müşteri Analizi: Segmentasyon, RFM analizi ve duygu analizi yapın
Gelişmiş Analizler: Karlılık analizi ve trend analizi gibi ileri düzey analizler yapın


Veri kaynağını seçin (örnek veri kullanabilir veya kendi verinizi yükleyebilirsiniz)
Analiz parametrelerini ayarlayın ve analizi başlatın
Sonuçları grafikler ve tablolar şeklinde görüntüleyin

Veri Formatı
Satış Tahminleri için CSV Formatı

```bash
date,sales,is_holiday,is_promotion,weekday,month,year
2022-01-01,350,1,0,5,1,2022
2022-01-02,280,0,0,6,1,2022
...
```
Müşteri Analizi için CSV Formatı
```bash
customer_id,avg_purchase_value,purchase_frequency,return_rate
CUST_00001,2500,12,0.05
CUST_00002,1800,3,0.12
...
```
Lisans
MIT
## Nasıl GitHub'a Yüklersiniz

1. İlk olarak, klasörleri ve dosyaları oluşturun:

```bash
mkdir -p yapay-zeka-veri-analizi/modules
cd yapay-zeka-veri-analizi
```
2. Her dosyayı uygun konuma kaydedin:

app.py ana dizine
modül dosyaları modules/ klasörüne
veri_analizi.py ana dizine (mevcut kodunuz)
requirements.txt ana dizine
README.md ana dizine

3. Bir Git deposu oluşturun:
```bash
git init
git add .
git commit -m "İlk commit: Yapay Zeka Veri Analizi Uygulaması"
```
4. GitHub'da yeni bir depo oluşturun ve yerel depoyu uzak depoya bağlayın:
```bash
git remote add origin https://github.com/kullaniciadi/yapay-zeka-veri-analizi.git
git push -u origin master
   
```
