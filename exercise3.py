# Importar todas las bibliotecas necesarias
import pandas as pd
import numpy as np
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# 1. CARGAR Y PREPARAR LOS DATOS --------------------------------------------
# Cargar el archivo CSV con muestreo aleatorio para eficiencia
np.random.seed(101)  # Semilla para reproducibilidad
filename = "Reviews.csv"
sample_size = 5000
total_records = 568454
skip_rows = sorted(np.random.choice(range(1, total_records+1), size=total_records-sample_size, replace=False))
amazon_reviews = pd.read_csv(filename, skiprows=skip_rows)

# 2. PREPROCESAMIENTO BÁSICO -----------------------------------------------
# Convertir ratings a sentimiento binario (1=positivo, 0=negativo)
amazon_reviews['Sentiment_rating'] = np.where(amazon_reviews['Score'] > 3, 1, 0)
amazon_reviews = amazon_reviews[amazon_reviews['Score'] != 3]  # Eliminar neutrales

# Función para limpieza de texto
def clean_text(text):
    text = text.lower()  # Convertir a minúsculas
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # Eliminar caracteres especiales
    tokens = word_tokenize(text)  # Tokenización
    # Eliminar stopwords y palabras personalizadas no relevantes
    stop_words = set(stopwords.words('english')).union({'br', 'nbsp'})
    tokens = [word for word in tokens if word not in stop_words]
    # Lematización (mejor que stemming para interpretación)
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word, pos='v') for word in tokens]  # 'v' para verbos
    return ' '.join(tokens)

# Aplicar limpieza a todas las reseñas
amazon_reviews['Processed_Text'] = amazon_reviews['Text'].apply(clean_text)

# Generar nube de palabras con texto preprocesado
all_text = ' '.join(amazon_reviews['Processed_Text'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud de Reseñas Preprocesadas')
plt.show()

# Tokenizar todo el texto preprocesado
all_tokens = [word for review in amazon_reviews['Processed_Text'] for word in review.split()]

# Calcular frecuencias y umbrales
word_freq = Counter(all_tokens)
total_unique_words = len(word_freq)
high_freq_threshold = int(total_unique_words * 0.01)  # Top 1%
low_freq_threshold = int(total_unique_words * 0.99)   # Bottom 1%

# Obtener palabras más y menos frecuentes
sorted_words = word_freq.most_common()
high_freq_words = [word for word, count in sorted_words[:high_freq_threshold]]
low_freq_words = [word for word, count in sorted_words[low_freq_threshold:]]

print("\nPalabras de alta frecuencia (top 1%):", high_freq_words[:10])  # Mostrar 10 ejemplos
print("Palabras de baja frecuencia (bottom 1%):", low_freq_words[:10])

# 5. ACTUALIZAR LISTA DE PALABRAS RUIDO ------------------------------------
# Combinar stopwords existentes con palabras de alta/baja frecuencia
noise_words = set(stopwords.words('english')).union({'br', 'nbsp'})
noise_words.update(high_freq_words)
noise_words.update(low_freq_words)

# Filtrar tokens eliminando palabras ruido
filtered_tokens = [word for word in all_tokens if word not in noise_words]
unique_filtered_tokens = set(filtered_tokens)

print("\nResumen de tokens:")
print(f"- Tokens únicos iniciales: {len(set(all_tokens))}")
print(f"- Tokens únicos después de filtrar: {len(unique_filtered_tokens)}")
print(f"- Total palabras en lista de ruido: {len(noise_words)}")

# 6. VERIFICACIÓN FINAL ----------------------------------------------------
# Mostrar ejemplo de reseña antes/después de filtrar palabras ruido
sample = amazon_reviews['Processed_Text'].iloc[0]
filtered_sample = ' '.join([word for word in sample.split() if word not in noise_words])

print("\nEjemplo de filtrado:")
print("Original:", sample)
print("Filtrado:", filtered_sample)