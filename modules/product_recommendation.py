import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def product_recommendation():
    st.subheader("📱 Teknolojik Ürün Öneri Motoru")

    # Ürün veri seti (örnek teknolojik ürünler)
    df = pd.DataFrame({
        'product_id': [101, 102, 103, 104, 105, 106, 107, 108],
        'product_name': [
            'Gaming Laptop', 'Ultrabook', 'Akıllı Telefon',
            'Tablet', 'Masaüstü Bilgisayar', 'Kablosuz Kulaklık',
            'Akıllı Saat', 'Bluetooth Hoparlör'
        ],
        'description': [
            'Yüksek performanslı oyun laptopu. RTX ekran kartı ve 16GB RAM.',
            'Hafif, şık ve taşınabilir dizüstü bilgisayar. SSD ve uzun pil ömrü.',
            'Android işletim sistemi, güçlü kamera, uzun batarya ömrü.',
            'Çok amaçlı tablet. Eğitim, eğlence ve iş için uygun.',
            'Ofis ve grafik için masaüstü bilgisayar. Yüksek işlemci gücü.',
            'Bluetooth kulaklık. Gürültü engelleme ve uzun pil ömrü.',
            'Spor ve sağlık takibi için akıllı saat. AMOLED ekran, sensörler.',
            'Taşınabilir Bluetooth hoparlör. Yüksek ses kalitesi, su geçirmez.'
        ]
    })

    # Ürün seçimi
    selected_product = st.selectbox("Bir ürün seçin:", df['product_name'])
    selected_index = df[df['product_name'] == selected_product].index[0]

    # TF-IDF modelini oluştur
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(df['description'])

    # Kosinüs benzerliği hesapla
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    # Benzer ürünleri sırala
    sim_scores = list(enumerate(cosine_sim[selected_index]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]  # İlk satır kendisi olduğundan atla

    recommended_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[recommended_indices]

    # Seçilen ürünü göster
    st.markdown("### 🎯 Seçilen Ürün")
    st.write(df.iloc[[selected_index]][['product_name', 'description']])

    # Benzer ürün önerileri
    st.markdown("### 🤖 Önerilen Benzer Ürünler")
    st.table(recommendations[['product_name', 'description']])
