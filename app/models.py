from django.db import models

class DadosTreino(models.Model):
    age = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    gender = models.CharField(max_length=10, choices=[('Female', 'Female'), ('Male', 'Male'), ('Unknown', 'Unknown')])
    blood_pressure = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    cholesterol_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    exercise_habits = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    smoking = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    family_heart_disease = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    diabetes = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    bmi = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    high_blood_pressure = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    low_hdl_cholesterol = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    high_ldl_cholesterol = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    alcohol_consumption = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('None', 'None'), ('Unknown', 'Unknown')])
    stress_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    sleep_hours = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    sugar_consumption = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    triglyceride_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    fasting_blood_sugar = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    crp_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    homocysteine_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    heart_disease_status = models.CharField(max_length=3, choices=[('No', 'No'), ('Yes', 'Yes')])

    class Meta:
        db_table = "app_dados_treino"


class DadosTeste(models.Model):
    age = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    gender = models.CharField(max_length=10, choices=[('Female', 'Female'), ('Male', 'Male'), ('Unknown', 'Unknown')])
    blood_pressure = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    cholesterol_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    exercise_habits = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    smoking = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    family_heart_disease = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    diabetes = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    bmi = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    high_blood_pressure = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    low_hdl_cholesterol = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    high_ldl_cholesterol = models.CharField(max_length=10, choices=[('No', 'No'), ('Unknown', 'Unknown'), ('Yes', 'Yes')])
    alcohol_consumption = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('None', 'None'), ('Unknown', 'Unknown')])
    stress_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    sleep_hours = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    sugar_consumption = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium'), ('Unknown', 'Unknown')])
    triglyceride_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    fasting_blood_sugar = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    crp_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    homocysteine_level = models.CharField(max_length=10, choices=[('High', 'High'), ('Low', 'Low'), ('Medium', 'Medium')])
    heart_disease_status = models.CharField(max_length=3, choices=[('No', 'No'), ('Yes', 'Yes')])

    class Meta:
        db_table = "app_dados_teste"
