import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache_resource

def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

def product_recommendation():
    st.subheader("Teknolojik Ürün Öneri Motoru (AI Destekli)")

    # Örnek veri
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

    model = load_model()

    st.markdown("**Bir ürün seçin veya kendi ürün tanımınızı yazın:**")
    user_choice = st.radio("Seçim tipi:", ["Listeden Seç", "Kendi Tanımını Yaz"], horizontal=True)

    if user_choice == "Listeden Seç":
        selected_product = st.selectbox("Ürün Seç:", df['product_name'])
        selected_index = df[df['product_name'] == selected_product].index[0]
        user_description = df.iloc[selected_index]['description']
        st.markdown("Seçilen Ürün")
        st.dataframe(df.iloc[[selected_index]][['product_name', 'description']])

    else:
        if st.button("Örnek Tanım Getir"):
            st.session_state['user_description'] = "Android sistemli, güçlü kamera, uzun pil ömrü olan mobil cihaz"

        user_description = st.text_area("Ürün Tanımını Girin:",
                                        value=st.session_state.get('user_description', ''),
                                        height=100)

    if st.button("Benzer Ürünleri Göster") and user_description:
        # Embed hesaplama
        descriptions = df['description'].tolist()
        all_descriptions = descriptions + [user_description]
        embeddings = model.encode(all_descriptions)

        user_embed = embeddings[-1].reshape(1, -1)
        similarities = cosine_similarity(user_embed, embeddings[:-1])[0]

        # En benzer 3 ürün
        top_indices = similarities.argsort()[::-1][:3]
        recommendations = df.iloc[top_indices]

        st.markdown("Önerilen Benzer Ürünler")
        st.table(recommendations[['product_name', 'description']])
