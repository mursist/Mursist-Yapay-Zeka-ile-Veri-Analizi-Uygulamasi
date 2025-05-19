import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import veri_analizi as va

# Modülleri içe aktar
from modules.dashboard import add_dashboard
from modules.sales_analysis import seasonal_analysis, price_analysis
from modules.customer_analysis import rfm_analysis, sentiment_analysis
from modules.advanced_analytics import profitability_analysis, trend_analysis

st.set_page_config(page_title="Yapay Zeka ile Veri Analizi", layout="wide")

st.title("Yapay Zeka ile Veri Analizi")

# Sekmeleri oluşturma - Ana modüller ve yeni modüller eklendi
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Ana Sayfa", 
    "Satış Tahmini", 
    "Müşteri Analizi", 
    "Gelişmiş Analizler",
    "Trendler",
    "Kullanım Kılavuzu"
])

# Ana Sayfa Sekmesi
with tab1:
    st.header("Yapay Zeka ile Veri Analizi Uygulamasına Hoş Geldiniz")
    
    st.info("Bu uygulama, Python'da geliştirilmiş veri analizi ve yapay zeka fonksiyonlarını kullanıcı dostu bir arayüz üzerinden erişilebilir hale getirmek için tasarlanmıştır.")
    
    # Dashboard ekle
    add_dashboard()
    
    # Modülleri görsel kutularda göster
    st.write("### Analiz Modülleri")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#1E88E5;">📈 Satış Tahmini</h4>
            <p>ARIMA ve makine öğrenmesi modelleri ile gelecek satışları tahmin edin.</p>
            <p>Mevsimsel analizler ve trend analizleri yapın.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#43A047;">👥 Müşteri Analizi</h4>
            <p>K-means ile müşteri segmentasyonu yapın.</p>
            <p>RFM analizi ile değerli müşterilerinizi tanımlayın.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="border:1px solid #ddd; border-radius:5px; padding:10px;">
            <h4 style="color:#E53935;">🔍 Gelişmiş Analizler</h4>
            <p>Duygu analizi, sepet analizi ve karlılık analizi gibi gelişmiş analizler yapın.</p>
        </div>
        """, unsafe_allow_html=True)

# Satış Tahmini Sekmesi
with tab2:
    st.header("Zaman Serisi Analizi ve Satış Tahmini")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Temel Tahmin", "Dönemsel Analiz", "Fiyat Analizi"])
    
    with sub_tab1:
        # Mevcut satış tahmini kodu
        sales_file = st.file_uploader("CSV Dosyası Yükleyin (veya örnek veri kullanın)", type="csv")
        if sales_file:
            sales_data = pd.read_csv(sales_file)
        else:
            if st.button("Örnek Veri Oluştur"):
                st.info("Örnek veri oluşturuluyor...")
                sales_data = va.create_sample_sales_data()
                st.success("Örnek veri oluşturuldu!")
                st.session_state['sales_data'] = sales_data
        
        if 'sales_data' in st.session_state:
            sales_data = st.session_state['sales_data']
            st.write("Veri Önizleme:")
            st.dataframe(sales_data.head())
            
            forecast_days = st.slider("Tahmin Günü Sayısı", 7, 90, 30)
            
            if st.button("Analizi Başlat"):
                st.info("Analiz yapılıyor...")
                try:
                    # Zaman serisi analizi
                    with st.spinner("Zaman serisi analizi yapılıyor..."):
                        result = va.analyze_time_series(sales_data)
                    
                    # ARIMA tahmin
                    with st.spinner(f"{forecast_days} günlük tahmin yapılıyor..."):
                        forecast = va.forecast_sales(sales_data, forecast_days)
                    
                    st.success("Analiz tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("Zaman Serisi Ayrıştırma")
                    
                    # Gözlemlenen satışlar
                    st.write("#### Gözlemlenen Satışlar")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.observed.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Trend bileşeni
                    st.write("#### Trend Bileşeni")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.trend.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Mevsimsel bileşen
                    st.write("#### Mevsimsel Bileşen")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.seasonal.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # Artık bileşen
                    st.write("#### Artık Bileşeni")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    result.resid.plot(ax=ax)
                    st.pyplot(fig)
                    
                    # ARIMA tahmin sonuçları
                    st.subheader("ARIMA Tahmin Sonuçları")
                    
                    fig, ax = plt.subplots(figsize=(12, 6))
                    # Son 90 gün + tahmin
                    ax.plot(sales_data.set_index('date')['sales'][-90:].index, 
                            sales_data.set_index('date')['sales'][-90:].values, 
                            label='Geçmiş Veriler')
                    ax.plot(forecast.index, forecast.values, color='red', label='Tahmin')
                    ax.set_title(f'{forecast_days} Günlük Tahmin')
                    ax.legend()
                    st.pyplot(fig)
                    
                    # Machine Learning modeli sonuçları
                    st.subheader("Makine Öğrenmesi Model Sonuçları")
                    with st.spinner("Makine öğrenmesi modelleri eğitiliyor..."):
                        rf_model, xgb_model = va.train_ml_sales_model(sales_data)
                    
                    # Model sonuçlarını göster
                    st.success("Modeller başarıyla eğitildi!")
                    
                except Exception as e:
                    st.error(f"Analiz sırasında bir hata oluştu: {e}")
    
    with sub_tab2:
        # Dönemsel analiz
        seasonal_analysis()
    
    with sub_tab3:
        # Fiyat elastikiyeti analizi
        price_analysis()

# Müşteri Analizi Sekmesi
with tab3:
    st.header("Müşteri Analizi")
    
    sub_tab1, sub_tab2, sub_tab3 = st.tabs(["Segmentasyon", "RFM Analizi", "Duygu Analizi"])
    
    with sub_tab1:
        # Müşteri segmentasyonu
        customer_file = st.file_uploader("Müşteri CSV Dosyası Yükleyin (veya örnek veri kullanın)", type="csv")
        if customer_file:
            customer_data = pd.read_csv(customer_file)
        else:
            if st.button("Örnek Müşteri Verisi Oluştur"):
                st.info("Örnek müşteri verisi oluşturuluyor...")
                customer_data = va.create_customer_data()
                st.success("Örnek müşteri verisi oluşturuldu!")
                st.session_state['customer_data'] = customer_data
        
        if 'customer_data' in st.session_state:
            customer_data = st.session_state['customer_data']
            st.write("Veri Önizleme:")
            st.dataframe(customer_data.head())
            
            cluster_count = st.slider("Küme Sayısı", 2, 8, 4)
            
            if st.button("Segmentasyon Analizini Başlat"):
                st.info("Segmentasyon analizi yapılıyor...")
                try:
                    with st.spinner("Müşteriler segmentlere ayrılıyor..."):
                        segmented_data, kmeans_model, scaler = va.segment_customers(customer_data, cluster_count)
                    
                    st.success("Segmentasyon tamamlandı!")
                    
                    # Sonuçları göster
                    st.subheader("Segmentasyon Sonuçları")
                    
                    # Küme görselleştirme
                    st.write("#### Küme Görselleştirmesi")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    scatter = ax.scatter(customer_data['avg_purchase_value'], 
                                        customer_data['purchase_frequency'],
                                        c=segmented_data['cluster'], 
                                        cmap='viridis', 
                                        alpha=0.6)
                    ax.set_xlabel('Ortalama Satın Alma Değeri')
                    ax.set_ylabel('Satın Alma Sıklığı')
                    ax.set_title('Müşteri Segmentasyonu')
                    legend1 = ax.legend(*scatter.legend_elements(),
                                      title="Kümeler")
                    ax.add_artist(legend1)
                    st.pyplot(fig)
                    
                    # Küme istatistikleri
                    st.write("#### Küme İstatistikleri")
                    cluster_stats = segmented_data.groupby('cluster').agg({
                        'customer_id': 'count',
                        'avg_purchase_value': 'mean',
                        'purchase_frequency': 'mean',
                        'return_rate': 'mean',
                        'customer_value': 'mean'
                    }).reset_index()
                    
                    cluster_stats.columns = ['Küme', 'Müşteri Sayısı', 'Ort. Satın Alma', 'Satın Alma Sıklığı', 'İade Oranı', 'Müşteri Değeri']
                    st.dataframe(cluster_stats)
                    
                except Exception as e:
                    st.error(f"Segmentasyon sırasında bir hata oluştu: {e}")
    
    with sub_tab2:
        # RFM analizi
        rfm_analysis()
    
    with sub_tab3:
        # Duygu analizi
        sentiment_analysis()

# Gelişmiş Analizler Sekmesi
with tab4:
    st.header("Gelişmiş Analizler")
    
    sub_tab1, sub_tab2 = st.tabs(["Karlılık Analizi", "Ürün Öneri Motoru"])
    
    with sub_tab1:
        # Karlılık analizi
        profitability_analysis()
    
    with sub_tab2:
        # Öneri motoru
        st.write("Ürün öneri motoru yakında eklenecek...")

# Trendler Sekmesi
with tab5:
    # Trend analizi
    trend_analysis()

# Kullanım Kılavuzu Sekmesi
with tab6:
    st.header("Kullanım Kılavuzu ve Teknik Detaylar")
    
    # Genel Bakış
    st.subheader("1. Genel Bakış")
    st.write("Bu uygulama, veri analizi ve yapay zeka yöntemlerini kullanarak satış tahmini, müşteri segmentasyonu ve anomali tespiti yapmanızı sağlayan etkileşimli bir araçtır.")
    st.markdown("""
    - **Zaman Serisi Analizi ve Satış Tahmini**: Geçmiş verileri analiz ederek gelecek satışlarını tahmin eder
    - **Müşteri Segmentasyonu**: Benzer davranış gösteren müşterileri gruplandırır
    - **Anomali Tespiti**: Normal müşteri davranışından sapan anormal desenleri tespit eder
    """)
    
    # Zaman Serisi Analizi 
    st.subheader("2. Zaman Serisi Analizi ve Satış Tahmini")
    st.write("Bu modül, geçmiş satış verilerini analiz ederek gelecekteki satışları tahmin etmek için kullanılır.")
    
    # Veri Formatı
    st.write("#### 2.1. Veri Formatı")
    st.markdown("""
    CSV dosyanızın aşağıdaki sütunları içermesi gerekir:
    - `date`: YYYY-MM-DD formatında tarih (ör. 2022-01-01)
    - `sales`: Sayısal satış değeri
    
    İsteğe bağlı olarak şu sütunları da ekleyebilirsiniz:
    - `is_holiday`: Tatil günü olup olmadığını belirten 1/0 değeri
    - `is_promotion`: Promosyon dönemi olup olmadığını belirten 1/0 değeri
    - `weekday`: Haftanın günü (0-6, 0=Pazartesi)
    - `month`: Ay (1-12)
    - `year`: Yıl
    - `is_weekend`: Hafta sonu olup olmadığını belirten 1/0 değeri
    """)
    
    # Detaylı teknik açıklamaları burada gösterebilirsiniz
    # Önceki örnekte verdiğim detaylı kılavuzu buraya ekleyin

if __name__ == "__main__":
    # Burada ihtiyaç duyulabilecek başlangıç işlemleri yapılabilir
    pass