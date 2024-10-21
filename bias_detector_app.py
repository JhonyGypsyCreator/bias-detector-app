import streamlit as st
from transformers import pipeline
import nltk
import torch

# Načítanie potrebných zdrojov z NLTK
nltk.download('vader_lexicon')

# Definovanie zoznamu hanlivých a stereotypných výrazov
negative_words = [
    # Prirovnania k zvieratám
    'prasa', 'pes', 'koza', 'ovca', 'krava', 'had', 'vlk', 'hmyz', 'opica',

    # Zlé vlastnosti
    'klamár', 'lenivý', 'hrubý', 'nespoľahlivý', 'nezodpovedný', 'podvodník', 
    'zákerák', 'nenávistný', 'úlisný', 'závisť', 'zákerný', 'zločinec',

    # Hygiena
    'špinavý', 'neumytý', 'smrdí', 'zanedbaný', 'páchnuci', 'neporiadny',

    # Vzdelanie
    'nevzdelaný', 'hlúpy', 'tupý', 'neinteligentný', 'negramotný', 'bez vzdelania',

    # Pôvodné hanlivé výrazy
    'neprispôsobiví', 'problémová komunita', 'príživník', 'neprispôsobivý', 
    'kriminálnik', 'podvodník', 'problematický', 'nečestný', 'zlodej', 'nekultúrny'
]

# Načítanie modelu pre sentimentovú analýzu zo služby Hugging Face
@st.cache_resource
def load_sentiment_model():
    return pipeline("sentiment-analysis", model="distilbert-base-multilingual-cased")

# Funkcia na detekciu hanlivých slov v texte
def find_negative_words(text):
    words = text.lower().split()
    return [word for word in words if word in negative_words]

# Funkcia na analýzu sentimentu
def analyze_sentiment(text, sentiment_model):
    result = sentiment_model(text)
    label = result[0]['label']
    score = result[0]['score']
    return label, score

# Hlavná funkcia pre Streamlit aplikáciu
def main():
    st.title("BiasDetector: Analýza textu na predsudky")
    st.write("Analyzujte texty a zistite, či obsahujú hanlivé alebo zaujaté opisy.")

    # Vstup pre text
    text = st.text_area("Zadajte text na analýzu", height=200)
    sentiment_model = load_sentiment_model()

    if st.button("Analyzovať"):
        # Vykonaj analýzu textu
        results = analyze_text(text, sentiment_model)
        st.write("### Výsledky analýzy:")
        st.write("**Výskyt hanlivých výrazov:**", results['negative_terms'])
        st.write("**Počet hanlivých výrazov:**", results['num_negative_terms'])
        st.write("**Sentiment textu:**", results['sentiment'])
        st.write("**Skóre sentimentu:**", results['sentiment_score'])

        # Upozornenie, ak sú zistené problematické časti
        if results['num_negative_terms'] > 0:
            st.warning("Pozor! Text obsahuje hanlivé výrazy alebo prirovnania.")
        elif results['sentiment'] == 'NEGATIVE':
            st.warning("Pozor! Text má negatívny sentiment.")
        else:
            st.success("Text neobsahuje výrazné známky zaujatosti.")

# Funkcia na analýzu textu
def analyze_text(text, sentiment_model):
    negative_terms = find_negative_words(text)
    num_negative_terms = len(negative_terms)
    sentiment, sentiment_score = analyze_sentiment(text, sentiment_model)
    
    return {
        'negative_terms': negative_terms,
        'num_negative_terms': num_negative_terms,
        'sentiment': sentiment,
        'sentiment_score': sentiment_score
    }

if __name__ == "__main__":
    main()


