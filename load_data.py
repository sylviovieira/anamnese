import os
import django

# Configurar o módulo de configurações do Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'AnamneseIntuitivaOrinetadaPorIA.settings')
django.setup()

from scipy.io import arff
import pandas as pd
from app.models import Dados_treino  # Nome do app é 'app'

# Carregar o ARFF
data, meta = arff.loadarff('heart_disease.arff')
df = pd.DataFrame(data)

# Converter bytes para strings (comum em ARFF)
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].str.decode('utf-8')

# Inserir na tabela
for index, row in df.iterrows():
    Dados_treino.objects.create(
        age=row['Age'],
        gender=row['Gender'],
        blood_pressure=row['Blood Pressure'],
        cholesterol_level=row['Cholesterol Level'],
        exercise_habits=row['Exercise Habits'],
        smoking=row['Smoking'],
        family_heart_disease=row['Family Heart Disease'],
        diabetes=row['Diabetes'],
        bmi=row['BMI'],
        high_blood_pressure=row['High Blood Pressure'],
        low_hdl_cholesterol=row['Low HDL Cholesterol'],
        high_ldl_cholesterol=row['High LDL Cholesterol'],
        alcohol_consumption=row['Alcohol Consumption'],
        stress_level=row['Stress Level'],
        sleep_hours=row['Sleep Hours'],
        sugar_consumption=row['Sugar Consumption'],
        triglyceride_level=row['Triglyceride Level'],
        fasting_blood_sugar=row['Fasting Blood Sugar'],
        crp_level=row['CRP Level'],
        homocysteine_level=row['Homocysteine Level'],
        heart_disease_status=row['Heart Disease Status']
    )

print("Dados carregados! Total de linhas:", len(df))