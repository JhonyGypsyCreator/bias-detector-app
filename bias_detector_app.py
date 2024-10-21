import streamlit as st
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# Načítanie potrebných zdrojov z NLTK
nltk.download('vader_lexicon')

# Definovanie zoznamu hanlivých a stereotypných výrazov
negative_words = [
    'neprispôsobiví', 'problémová komunita', 'leniví', 'príživník', 
    'neprispôsobivý', 'kriminálnik', 'podvodník', 'problematický'
]

# Funkcia na detekciu hanlivých slov v texte
def find_negative_words(text):
    words = text.lower().split()
    return [word for word in words if word in negative_words]

# Funkcia na analýzu sentimentu
def analyze_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    score = sia.polarity_scores(text)
    return score['compound']

# Hlavná funkcia pre Streamlit aplikáciu
def main():
    st.title("BiasDetector: Analýza textu na predsudky")
    st.write("Analyzujte texty a zistite, či obsahujú hanlivé alebo zaujaté opisy.")

    # Vstup pre text
    text = st.text_area("Zadajte text na analýzu", height=200)

    if st.button("Analyzovať"):
        # Vykonaj analýzu textu
        results = analyze_text(text)
        st.write("### Výsledky analýzy:")
        st.write("**Výskyt hanlivých výrazov:**", results['negative_terms'])
        st.write("**Počet hanlivých výrazov:**", results['num_negative_terms'])
        st.write("**Sentiment textu:**", results['sentiment'])
        st.write("**Skóre sentimentu:**", results['sentiment_score'])

        # Upozornenie, ak sú zistené problematické časti
        if results['num_negative_terms'] > 0 or results['sentiment'] == 'Negatívny':
            st.warning("Pozor! Text obsahuje potenciálne zaujaté alebo hanlivé výrazy.")
        else:
            st.success("Text neobsahuje výrazné známky zaujatosti.")

# Funkcia na analýzu textu
def analyze_text(text):
    negative_terms = find_negative_words(text)
    num_negative_terms = len(negative_terms)
    sentiment_score = analyze_sentiment(text)
    sentiment = 'Neutrálny'
    if sentiment_score > 0.1:
        sentiment = 'Pozitívny'
    elif sentiment_score < -0.1:
        sentiment = 'Negatívny'
    
    return {
        'negative_terms': negative_terms,
        'num_negative_terms': num_negative_terms,
        'sentiment': sentiment,
        'sentiment_score': sentiment_score
    }

if __name__ == "__main__":
    main()
