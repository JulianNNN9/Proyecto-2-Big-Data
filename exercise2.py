#Ejercicio 2: ¿Eliminar caracteres especiales es realmente una buena idea? 
# ¿Cuáles son algunos ejemplos de caracteres que probablemente sean seguros de eliminar 
# y cuáles no lo serían?

'''
-Eliminar caracteres especiales puede ser útil en algunos contextos, como la limpieza
 de datos para análisis de texto o procesamiento de lenguaje natural, donde ciertos símbolos
 no aportan significado. Sin embargo, no siempre es una buena idea, ya que algunos caracteres 
 especiales pueden tener valor semántico o estructural.

-Caracteres que probablemente sean seguros de eliminar incluyen signos de puntuación repetidos,
 símbolos decorativos (como "*", "~") o caracteres de formato innecesarios.

-Sin embargo, no todos los caracteres especiales deben eliminarse. Por ejemplo, símbolos como 
@ o # son importantes en textos de redes sociales, ya que indican usuarios o temas relevantes. 
También, términos de negación (como "no") o signos de puntuación que modifican el tono (!, ?)
 pueden ser claves para interpretar correctamente el sentimiento. Incluso los emojis,
 en algunos contextos, expresan emociones fuertes y pueden aportar valor al análisis.
'''

import pandas as pd
import re
from collections import Counter

# 1. Extraer caracteres especiales del texto preprocesado (antes de eliminarlos)
def get_special_chars(text):
    return [char for char in text if not char.isalnum() and char != ' ']

# Cargar el dataset en amazon_reviews (asegúrate de que 'dataset.csv' sea el archivo correcto)
amazon_reviews = pd.read_csv('Reviews.csv')

# Aplicar a una columna de texto (ejemplo con 'Text' original)
special_chars = amazon_reviews['Text'].apply(get_special_chars)

# 2. Lista plana de todos los caracteres especiales únicos
flat_list = [char for sublist in special_chars for char in sublist]
unique_special_chars = set(flat_list)
print("Caracteres especiales únicos encontrados:\n", unique_special_chars)

# 3. Análisis de frecuencia de caracteres especiales
char_freq = Counter(flat_list)
print("\nFrecuencia de caracteres especiales:\n", char_freq.most_common(10))

# 4. Ejemplos de texto antes/después de eliminar caracteres especiales
sample_review = amazon_reviews['Text'].iloc[0]  # Primera reseña como ejemplo
cleaned_review = re.sub(r'[^A-Za-z0-9 ]+', ' ', sample_review)

print("\n--- Ejemplo de reseña ---")
print("Original:\n", sample_review)
print("\nLimpia:\n", cleaned_review)

# 5. Identificación de caracteres "seguros" vs. "problemáticos"
safe_to_remove = {'!', '?', ',', '.', ';', ':', '"', "'", '(', ')', '[', ']'}  # Símbolos de puntuación
potentially_important = {'@', '#', '$', '%', '&', '*', '+', '-', '/', '=', '<', '>', '~'}  # Emoticones, abreviaciones

print("\nCaracteres seguros para eliminar:\n", safe_to_remove)
print("\nCaracteres a evaluar antes de eliminar:\n", potentially_important)

#Detector de emoticonos básicos en reseñas
def detect_emoticons(text_series):
    """
    Detecta emoticonos básicos en una serie de textos y cuenta su frecuencia.
    Devuelve un contador con los emoticonos encontrados.
    """
    # Patrones comunes de emoticones
    emoticon_patterns = {
        r':\)': 'sonrisa',
        r':\(': 'triste', 
        r';\)': 'guiño',
        r':D': 'risa',
        r':/': 'confusión',
        r':O|:o': 'sorpresa',
        r':P': 'lengua',
        r'<3': 'corazón',
        r'XD': 'carcajada'
    }
    
    emoticon_counts = Counter()
    reviews_with_emoticons = 0
    
    # Revisar cada texto
    for text in text_series:
        if not isinstance(text, str):
            continue
            
        found_in_this_review = False
        # Buscar cada patrón de emoticono
        for pattern, name in emoticon_patterns.items():
            if re.search(pattern, text, re.IGNORECASE):
                emoticon_counts[name] += 1
                found_in_this_review = True
                
        if found_in_this_review:
            reviews_with_emoticons += 1
    
    return {
        'emoticon_counts': emoticon_counts,
        'reviews_with_emoticons': reviews_with_emoticons,
        'percentage': (reviews_with_emoticons / len(text_series)) * 100
    }

# Ejecutar el detector de emoticonos
emoticon_results = detect_emoticons(amazon_reviews['Text'])

# Mostrar resultados
print("\n--- ANÁLISIS DE EMOTICONOS ---")
print(f"Se encontraron emoticonos en {emoticon_results['reviews_with_emoticons']} reseñas")
print(f"({emoticon_results['percentage']:.2f}% del total)")
print("\nEmoticonos más frecuentes:")
for emoticon, count in emoticon_results['emoticon_counts'].most_common(5):
    print(f"- {emoticon}: {count} veces")