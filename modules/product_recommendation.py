import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

def product_recommendation():
    st.subheader("Teknolojik Ürün Öneri Motoru")

    df = pd.DataFrame({
        'product_id': [101, 102, 103, 104, 105, 106, 107],
        'product_name': [
            'Gaming Laptop', 'Ultrabook', 'Akıllı Telefon',
            'Tablet', 'Masaüstü Bilgisayar', 'Kulaklık', 'Akıllı Saat'
        ],
        'description': [
            'Intel i7 işlemci, 16GB RAM, NVIDIA RTX 3060 ekran kartı, 1TB SSD ile yüksek performanslı oyun deneyimi.',
            'Intel i5 işlemci, 8GB RAM, hafif tasarım, 512GB SSD, uzun pil ömrü ile taşınabilirlik odaklı.',
            'Snapdragon işlemci, 128GB hafıza, 6.5 inç ekran, Android 13, 5000mAh batarya ile güçlü akıllı telefon.',
            '10.1 inç ekran, 4GB RAM, 64GB depolama, hafif ve taşınabilir, Android tabanlı tablet.',
            'Ryzen 5 işlemci, 32GB RAM, 2TB SSD, 4K destekli ekran kartı ile ofis ve oyun için masaüstü bilgisayar.',
            'Kablosuz bluetooth kulaklık, aktif gürültü engelleme, 40 saat pil ömrü.',
            '1.43 inç AMOLED ekran, kalp ritmi takibi, adım sayar, 7 gün pil ömrü ile akıllı saat.'
        ]
    })

    selected_product = st.selectbox("Bir ürün seçin:", df['product_name'].tolist())
    selected_id = df[df['product_name'] == selected_product]['product_id'].values[0]

    st.markdown("### Seçilen Ürün")
    st.write(df[df['product_id'] == selected_id][['product_name', 'description']])

    tfidf = TfidfVectorizer() 
    tfidf_matrix = tfidf.fit_transform(df['description'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    idx = df.index[df['product_id'] == selected_id][0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:4]

    product_indices = [i[0] for i in sim_scores]
    recommended = df.iloc[product_indices][['product_name', 'description']]

    st.markdown("Benzer Ürün Önerileri")
    st.dataframe(recommended, use_container_width=True)
