from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd
from django.apps import apps
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import io
from django.shortcuts import render, redirect
from .models import DadosTeste
from django.shortcuts import render


def treinar_page(request):
    return render(request, "treinar.html")


def interface(request):
    return render(request, "index.html")


def categorize_age(age_value):
    age_value = int(age_value)
    if age_value < 30:
        return "Low"
    elif age_value < 60:
        return "Medium"
    else:
        return "High"


def categorize_bmi(bmi_value):
    bmi_value = float(bmi_value)
    if bmi_value < 18.5:
        return "Low"
    elif bmi_value < 25:
        return "Medium"
    else:
        return "High"


def anamnese(request):
    if request.method == "POST":

        age_value = request.POST.get("age_value")
        weight = request.POST.get("weight")
        height = request.POST.get("height")
        bmi_value = request.POST.get("bmi")

        age_category = categorize_age(age_value)
        bmi_category = categorize_bmi(bmi_value)

        dados = Dados_teste(
            age=age_category,
            gender=request.POST.get("gender"),
            blood_pressure=request.POST.get("blood_pressure"),
            cholesterol_level=request.POST.get("cholesterol_level"),
            exercise_habits=request.POST.get("exercise_habits"),
            smoking=request.POST.get("smoking"),
            family_heart_disease=request.POST.get("family_heart_disease"),
            diabetes=request.POST.get("diabetes"),
            bmi=bmi_category,
            high_blood_pressure=request.POST.get("high_blood_pressure"),
            low_hdl_cholesterol=request.POST.get("low_hdl_cholesterol"),
            high_ldl_cholesterol=request.POST.get("high_ldl_cholesterol"),
            alcohol_consumption=request.POST.get("alcohol_consumption"),
            stress_level=request.POST.get("stress_level"),
            sleep_hours=request.POST.get("sleep_hours"),
            sugar_consumption=request.POST.get("sugar_consumption"),
            triglyceride_level=request.POST.get("triglyceride_level"),
            fasting_blood_sugar=request.POST.get("fasting_blood_sugar"),
            crp_level=request.POST.get("crp_level"),
            homocysteine_level=request.POST.get("homocysteine_level"),
            heart_disease_status=request.POST.get("heart_disease_status"),
        )
        dados.save()
        return redirect("resultado")

    return render(request, "Anamnese Intuitiva Orientada por IA.html")


def resultado(request):
    return render(request, "Resultado da Análise de Pré-Diagnóstico para o Profissional de Saúde.html")


@csrf_exempt
def treinar_modelo_view(request):
    if request.method == "POST":
        DadosTreino = apps.get_model("app", "DadosTreino")
        queryset = DadosTreino.objects.all().values()
        df = pd.DataFrame(list(queryset))

        if df.empty:
            return JsonResponse({"resultado": "Tabela de treino está vazia!"})

        # LabelEncoder
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])

        X = df.drop("heart_disease_status", axis=1)
        y = df["heart_disease_status"]

        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        clf = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )

        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_res, y_res, cv=cv)
        scores = cross_val_score(clf, X_res, y_res, cv=cv)

        buffer = io.StringIO()
        buffer.write("=== Summary ===\n")
        buffer.write(
            f"Correctly Classified Instances  {accuracy_score(y_res, y_pred)*100:.2f} %\n")
        buffer.write(
            f"Incorrectly Classified Instances {(1-accuracy_score(y_res, y_pred))*100:.2f} %\n")
        buffer.write(
            f"Mean Accuracy (10-fold CV)       {scores.mean()*100:.2f} % ± {scores.std()*100:.2f}%\n\n")

        buffer.write("=== Confusion Matrix ===\n")
        buffer.write(str(confusion_matrix(y_res, y_pred)) + "\n\n")

        buffer.write("=== Classification Report ===\n")
        buffer.write(classification_report(y_res, y_pred))

        return JsonResponse({"resultado": buffer.getvalue()})
