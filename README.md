# restaurant-recsys-tfg

Trabajo Final de Grado Daniel Torres Candil - UAM EPS

---

# Step Guide

1. Extraer de yelp_academic_dataset_business.json los businesses que son restaurantes (**_pandas_**).
2. Extraer solo las reviews relacionadas con los restaurantes a otro dataset (**_pandas_**).
3. Crear varios datasets más pequeños con los que poder trabajar (**_pandas_**).
4. Procesar las reviews en un nuevo módulo y guardarlas en un nuevo dataset - _annotations_ (**_pandas_** - **_nltk_**).
   1. Importar los datasets de aspects, lexicon y modifiers.
   2. Crear dataset con la estructura determinada en el tfg.
   3. Para cada una de las reviews, ir frase por frase sacando el sentimiento y los aspects, lexicon y modifiers de cada una.
   4. Guardar cada lexicon como una nueva entrada en el dataset.

---

# Python Modules

- pandas
- nltk

---