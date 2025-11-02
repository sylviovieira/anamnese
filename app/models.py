from django.db import models
from datetime import datetime


class DadosTeste(models.Model):
    # --- Coluna 1 do Formulário ---
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    blood_pressure = models.CharField(max_length=10, null=True, blank=True)
    cholesterol_level = models.CharField(max_length=10, null=True, blank=True)
    exercise_habits = models.CharField(max_length=10, null=True, blank=True)
    smoking = models.CharField(max_length=10, null=True, blank=True)
    family_heart_disease = models.CharField(
        max_length=10, null=True, blank=True)
    diabetes = models.CharField(max_length=10, null=True, blank=True)
    weight = models.FloatField(null=True, blank=True)
    height = models.FloatField(null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)
    heart_disease_status = models.CharField(
    max_length=255, null=True, blank=True)

    # --- Coluna 2 do Formulário ---
    high_blood_pressure = models.CharField(
        max_length=10, null=True, blank=True)
    low_hdl_cholesterol = models.CharField(
        max_length=10, null=True, blank=True)
    high_ldl_cholesterol = models.CharField(
        max_length=10, null=True, blank=True)
    alcohol_consumption = models.CharField(
        max_length=10, null=True, blank=True)
    stress_level = models.CharField(max_length=10, null=True, blank=True)
    sleep_hours = models.CharField(max_length=10, null=True, blank=True)
    sugar_consumption = models.CharField(max_length=10, null=True, blank=True)
    triglyceride_level = models.CharField(max_length=10, null=True, blank=True)
    fasting_blood_sugar = models.CharField(
        max_length=10, null=True, blank=True)
    crp_level = models.CharField(max_length=10, null=True, blank=True)
    homocysteine_level = models.CharField(max_length=10, null=True, blank=True)

    class Meta:
        db_table = 'app_dados_teste'

    def __str__(self):
        return f"Dados ID {self.id} - Idade {self.age} - Doença Cardíaca {self.heart_disease_status}"

class DadosTreino(models.Model):
    # --- Definindo os campos como eles SÃO no banco de dados (tipos crus) ---
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    blood_pressure = models.CharField(max_length=10, null=True, blank=True)
    cholesterol_level = models.CharField(max_length=10, null=True, blank=True)
    exercise_habits = models.CharField(max_length=10, null=True, blank=True)
    smoking = models.CharField(max_length=10, null=True, blank=True)
    family_heart_disease = models.CharField(
        max_length=10, null=True, blank=True)
    diabetes = models.CharField(max_length=10, null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)
    high_blood_pressure = models.CharField(
        max_length=10, null=True, blank=True)
    low_hdl_cholesterol = models.CharField(
        max_length=10, null=True, blank=True)
    high_ldl_cholesterol = models.CharField(
        max_length=10, null=True, blank=True)
    alcohol_consumption = models.CharField(
        max_length=10, null=True, blank=True)
    stress_level = models.CharField(max_length=10, null=True, blank=True)
    sleep_hours = models.CharField(max_length=10, null=True, blank=True)
    sugar_consumption = models.CharField(max_length=10, null=True, blank=True)
    triglyceride_level = models.CharField(max_length=10, null=True, blank=True)
    fasting_blood_sugar = models.CharField(
        max_length=10, null=True, blank=True)
    crp_level = models.CharField(max_length=10, null=True, blank=True)
    
    heart_disease_status = models.CharField(
        max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')], default='Unknown', null=False) 
    # O campo que adicionámos
    homocysteine_level = models.CharField(max_length=10, default='Unknown', null=True, blank=True)


    class Meta:
        db_table = 'app_dados_treino'

    def __str__(self):
        return f"DadosTreino {self.id}"


class DadosPacientes(models.Model):
    nome_paciente = models.CharField(max_length=100, null=False, blank=False)
    age = models.IntegerField(null=True, blank=True)
    gender = models.CharField(max_length=10, null=True, blank=True)
    blood_pressure = models.CharField(max_length=20, null=True, blank=True)
    cholesterol_level = models.CharField(max_length=20, null=True, blank=True)
    exercise_habits = models.CharField(max_length=20, null=True, blank=True)
    smoking = models.CharField(max_length=10, null=True, blank=True)
    family_heart_disease = models.CharField(max_length=10, null=True, blank=True)
    diabetes = models.CharField(max_length=10, null=True, blank=True)
    bmi = models.FloatField(null=True, blank=True)
    high_blood_pressure = models.CharField(max_length=10, null=True, blank=True)
    low_hdl_cholesterol = models.CharField(max_length=10, null=True, blank=True)
    high_ldl_cholesterol = models.CharField(max_length=10, null=True, blank=True)
    alcohol_consumption = models.CharField(max_length=10, null=True, blank=True)
    stress_level = models.CharField(max_length=10, null=True, blank=True)
    sleep_hours = models.CharField(max_length=10, null=True, blank=True)
    sugar_consumption = models.CharField(max_length=10, null=True, blank=True)
    triglyceride_level = models.CharField(max_length=10, null=True, blank=True)
    fasting_blood_sugar = models.CharField(max_length=10, null=True, blank=True)
    crp_level = models.CharField(max_length=10, null=True, blank=True)
    homocysteine_level = models.CharField(max_length=10, null=True, blank=True)

    risk_class = models.CharField(max_length=10, null=True, blank=True)
    risk_percentage = models.FloatField(null=True, blank=True)
    heart_disease_status = models.CharField(max_length=255, null=True, blank=True)

    anotacoes = models.TextField(null=True, blank=True)  
    data_envio = models.DateField(default=datetime.now)  
    horario_envio = models.TimeField(default=datetime.now)  

    created_at = models.DateTimeField(auto_now_add=True)  # mantém histórico interno do Django

    class Meta:
        db_table = 'app_dados_pacientes'

    @property
    def nome_formatado(self):
        """Retorna o nome do paciente com espaços em vez de underlines."""
        return self.nome_paciente.replace("_", " ")

    def __str__(self):
        return f"{self.nome_paciente} - {self.risk_class} ({self.risk_percentage:.2f}%)"
