import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Descargas necesarias de NLTK (solo primera vez)
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt_tab')

# 1. Cargar datos y muestreo (igual que en el documento)
records_in_file = 568454  # Número total de registros
sample_size = 5000        # Tamaño deseado de muestra
filename = "Reviews.csv"

# Muestreo aleatorio
np.random.seed(101)
skip = sorted(np.random.choice(range(1, records_in_file+1), 
              size=records_in_file-sample_size, 
              replace=False))
amazon_reviews = pd.read_csv(filename, skiprows=skip)

# 2. Convertir ratings a sentimiento binario (1=positivo, 0=negativo)
amazon_reviews['Sentiment_rating'] = np.where(amazon_reviews['Score'] > 3, 1, 0)
amazon_reviews = amazon_reviews[amazon_reviews['Score'] != 3]  # Eliminar neutrales (3)

# 3. Preprocesamiento de texto
def preprocess_text(text):
    # Convertir a minúsculas
    text = text.lower()
    
    # Eliminar caracteres especiales (conserva espacios y alfanuméricos)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    
    # Tokenización
    tokens = word_tokenize(text)
    
    # Eliminar stopwords y palabras de ruido
    stop_words = set(stopwords.words('english'))
    
    custom_noise_words = {'br', 'nbsp'}  #CUSTOM STOP WORDS PARA ELIMINAR RESIDUOS HTML
    
    noise_words = stop_words.union(custom_noise_words)
    tokens = [word for word in tokens if word not in noise_words]
    
    # Lemmatización (mejor que stemming para interpretación)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # 'v' para verbos
    
    return ' '.join(tokens)

# Aplicar preprocesamiento
amazon_reviews['Processed_Text'] = amazon_reviews['Text'].apply(preprocess_text)

# 4. Análisis de palabras frecuentes (Word Cloud)

# Generar Word Cloud
all_text = ' '.join(amazon_reviews['Processed_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)

plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud de Reseñas Preprocesadas')
plt.show()