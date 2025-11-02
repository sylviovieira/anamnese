from .utils import categorize_age, categorize_bmi
from .models import DadosTeste
from django.shortcuts import render
import traceback
from datetime import datetime
import io
import os
import traceback

import joblib
import mysql.connector
import pandas as pd
from django.apps import apps
from django.conf import settings
from django.db import connection as django_db_connection
from django.http import JsonResponse
from django.shortcuts import render, redirect, get_object_or_404
from django.views.decorators.csrf import csrf_exempt
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from .models import DadosTeste, DadosTreino, DadosPacientes
from .utils import categorize_age, categorize_bmi


def treinar_page(request):

    return render(request, "treinar.html")


def resultado(request):

    return render(request, "resultado.html")


@csrf_exempt
def treinar_modelo_view(request):

    if request.method == "POST":
        try:
            queryset = DadosTreino.objects.all().values()
            df = pd.DataFrame(list(queryset))

            if df.empty:
                return JsonResponse({"resultado": "Tabela de treino está vazia!"})

            try:
                if df['age'].dtype != 'object':
                    df['age'] = df['age'].apply(categorize_age)
                if df['bmi'].dtype != 'object':
                    df['bmi'] = df['bmi'].apply(categorize_bmi)
            except Exception as e_cat:
                return JsonResponse({"resultado": f"Erro ao categorizar colunas: {e_cat}"}, status=500)

            encoders = {}

            for col in df.columns:
                if df[col].dtype == "object":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le

            X = df.drop(["id", "heart_disease_status"], axis=1)
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

            model_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'decision_tree_model.pkl')
            model_dir = os.path.dirname(model_save_path)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            clf.fit(X_res, y_res)
            joblib.dump(clf, model_save_path)

            encoder_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'label_encoders.pkl')
            joblib.dump(encoders, encoder_save_path)

            return JsonResponse({"resultado": buffer.getvalue()})

        except Exception as e:
            error_message = f"ERRO AO TREINAR:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_message)
            return JsonResponse({"resultado": error_message}, status=500)



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


def interface(request):

    return render(request, "index.html")


def anamnese_page(request):

    return render(request, "anamnese_form.html")


def salvar_dados_page(request):

    return render(request, "salvar_dados.html")

def treinar_page(request):

    return render(request, "treinar.html")


def resultado(request):

    return render(request, "resultado.html")


@csrf_exempt
def treinar_modelo_view(request):

    if request.method == "POST":
        try:
            queryset = DadosTreino.objects.all().values()
            df = pd.DataFrame(list(queryset))

            if df.empty:
                return JsonResponse({"resultado": "Tabela de treino está vazia!"})

            try:
                if df['age'].dtype != 'object':
                    df['age'] = df['age'].apply(categorize_age)
                if df['bmi'].dtype != 'object':
                    df['bmi'] = df['bmi'].apply(categorize_bmi)
            except Exception as e_cat:
                return JsonResponse({"resultado": f"Erro ao categorizar colunas: {e_cat}"}, status=500)

            encoders = {}

            for col in df.columns:
                if df[col].dtype == "object":
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col])
                    encoders[col] = le

            X = df.drop(["id", "heart_disease_status"], axis=1)
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

            model_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'decision_tree_model.pkl')
            model_dir = os.path.dirname(model_save_path)

            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            clf.fit(X_res, y_res)
            joblib.dump(clf, model_save_path)

            encoder_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'label_encoders.pkl')
            joblib.dump(encoders, encoder_save_path)

            return JsonResponse({"resultado": buffer.getvalue()})

        except Exception as e:
            error_message = f"ERRO AO TREINAR:\n{str(e)}\n\n{traceback.format_exc()}"
            print(error_message)
            return JsonResponse({"resultado": error_message}, status=500)


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


def interface(request):

    return render(request, "index.html")


def anamnese_page(request):

    return render(request, "anamnese_form.html")


def salvar_dados_page(request):

    return render(request, "salvar_dados.html")
@csrf_exempt
def anamnese(request):
    if request.method == "POST":
        try:
            # Caminhos dos arquivos do modelo e encoders
            model_path = os.path.join(settings.BASE_DIR, 'app', 'model', 'decision_tree_model.pkl')
            encoders_path = os.path.join(settings.BASE_DIR, 'app', 'model', 'label_encoders.pkl')

            # Carrega modelo e encoders
            model = joblib.load(model_path)
            encoders = joblib.load(encoders_path)

            # Nome do paciente
            nome_paciente = request.POST.get("nome_paciente", "Paciente_Desconhecido").replace(" ", "_")

            # Colunas esperadas pelo modelo
            colunas = [
                'age', 'gender', 'blood_pressure', 'cholesterol_level',
                'exercise_habits', 'smoking', 'family_heart_disease', 'diabetes',
                'bmi', 'high_blood_pressure', 'low_hdl_cholesterol',
                'high_ldl_cholesterol', 'alcohol_consumption', 'stress_level',
                'sleep_hours', 'sugar_consumption', 'triglyceride_level',
                'fasting_blood_sugar', 'crp_level', 'homocysteine_level'
            ]

            # Captura dados originais do formulário
            dados_paciente_raw = {col: request.POST.get(col) for col in colunas}

            # Cópia para processar
            dados_para_modelo = dados_paciente_raw.copy()

            # Categoriza idade e IMC
            try:
                if dados_para_modelo['age'] and dados_para_modelo['age'].isdigit():
                    dados_para_modelo['age'] = categorize_age(dados_para_modelo['age'])
            except Exception as e_age:
                print(f"Erro ao categorizar idade: {e_age}")

            try:
                if dados_para_modelo['bmi']:
                    float(dados_para_modelo['bmi'])
                    dados_para_modelo['bmi'] = categorize_bmi(dados_para_modelo['bmi'])
            except Exception as e_bmi:
                print(f"Erro ao categorizar IMC: {e_bmi}")

            # Transforma dados com LabelEncoder
            input_data_list = []
            for col in colunas:
                valor = dados_para_modelo[col] or 'Unknown'
                le = encoders.get(col)
                if le:
                    if valor not in le.classes_ and 'Unknown' in le.classes_:
                        valor = 'Unknown'
                    input_data_list.append(le.transform([valor])[0])
                else:
                    input_data_list.append(0)

            # Faz a predição e calcula a probabilidade
            probabilities = model.predict_proba([input_data_list])[0]
            prob_neg, prob_pos = probabilities[0] * 100, probabilities[1] * 100
            classe = "Positivo" if prob_pos >= 50 else "Negativo"
            resultado = f"{classe} ({prob_pos:.2f}% de chance de Doença Cardíaca)"
            
            dados_paciente_raw.pop('heart_disease_status', None)
            # Salva os dados no banco (app_dados_pacientes)
            DadosPacientes.objects.create(
                nome_paciente=nome_paciente,
                **dados_paciente_raw,
                risk_class=classe,
                risk_percentage=prob_pos,
                heart_disease_status="No",
                anotacoes="",  # começa vazio
                data_envio=datetime.now().date(),
                horario_envio=datetime.now().time()
            )

            print(f"✅ Dados do paciente '{nome_paciente}' salvos com sucesso na tabela app_dados_pacientes.")

            # Redireciona para a view de resultado
            request.session["mensagem"] = "Anamnese salva com sucesso!"
            request.session["sucesso"] = True
            return redirect("resultado_anamnese")

        except Exception as e:
            print(f"❌ Erro na view anamnese: {e}")
            traceback.print_exc()

            # Redireciona para a view de erro
            request.session["mensagem"] = "Erro ao salvar a anamnese."
            request.session["sucesso"] = False
            return redirect("resultado_anamnese")

    # Se for GET → exibe o formulário
    return render(request, "anamnese.html")

@csrf_exempt
def treinar(request):

    db_connection = None

    cursor = None

    try:

        db_config = settings.DATABASES['default']

        try:

            db_connection = mysql.connector.connect(

                host=db_config['HOST'],

                user=db_config['USER'],

                password=db_config['PASSWORD'],

                database=db_config['NAME'],

                port=db_config.get('PORT', 3306)

            )

            cursor = db_connection.cursor()

        except mysql.connector.Error as err:

            print(f"FALHA ao conectar ao DB: {err}")

            return JsonResponse({'status': 'error', 'message': f'Erro ao conectar ao banco de dados: {err}'}, status=500)

        query = "SELECT age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope, ca, thal, heart_disease_status FROM app_dados_treino"

        df = pd.read_sql(query, db_connection)

        if df.empty:

            return JsonResponse({'status': 'error', 'message': 'Nenhum dado encontrado no banco para treinar.'}, status=400)

        df.dropna(axis=1, thresh=int(0.5 * len(df)), inplace=True)

        encoders = {}

        for col in df.columns:

            if df[col].isnull().any():

                if df[col].dtype == 'object':

                    df[col].fillna(df[col].mode()[0], inplace=True)

                else:

                    df[col].fillna(df[col].median(), inplace=True)

        for col in df.columns:

            if df[col].dtype == "object":

                if col in df:

                    le = LabelEncoder()

                    df[col] = le.fit_transform(df[col])

                else:

                    print(
                        f"Aviso: Coluna {col} não encontrada após processamento inicial.")

        if "heart_disease_status" not in df.columns:

            print(
                "Erro: Coluna 'heart_disease_status' não encontrada após pré-processamento.")

            return JsonResponse({'status': 'error', 'message': 'Coluna alvo "heart_disease_status" perdida durante o pré-processamento.'}, status=500)

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

        buffer = io.StringIO()

        buffer.write("=== Summary ===\n")

        buffer.write(

            f"Correctly Classified Instances  {accuracy_score(y_res, y_pred)*100:.2f} %\n")

        buffer.write(

            f"Incorrectly Classified Instances {(1-accuracy_score(y_res, y_pred))*100:.2f} %\n")

        buffer.write(f"Standard deviation {y_pred.std():.4f}\n\n")

        buffer.write("=== Detailed Accuracy By Class ===\n")

        report = classification_report(y_res, y_pred, output_dict=True)

        buffer.write("                 Precision  Recall  F1-Score\n")

        for label, metrics in report.items():

            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:

                try:

                    buffer.write(
                        f"Class {label:<5}      {metrics['precision']:.3f}   {metrics['recall']:.3f}   {metrics['f1-score']:.3f}\n")

                except KeyError:

                    pass

        buffer.write("\n=== Confusion Matrix ===\n")

        cm = confusion_matrix(y_res, y_pred)

        buffer.write("   a   b   <-- classified as\n")

        for i, row in enumerate(cm):

            label = chr(ord('a') + i)

            if len(row) == 2:

                buffer.write(
                    f" {row[0]:>3} {row[1]:>3} |   {label} = class {i}\n")

            else:

                buffer.write(f" {' '.join(map(lambda x: f'{x:>3}', row))} | {label} = class {i}\n")

        resultado = buffer.getvalue()

        buffer.close()

        clf.fit(X_res, y_res)

        model_save_path = os.path.join(
            settings.BASE_DIR, 'app', 'model', 'decision_tree_model.pkl')

        ClassifierClass = apps.get_app_config('app').Classifier
        classifier_instance = ClassifierClass()
        classifier_instance.save_model(clf, model_save_path)

        try:
            encoders_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'label_encoders.pkl')
            joblib.dump(encoders, encoders_save_path)
            print(f"Encoders salvos em {encoders_save_path}")
        except Exception as e:
            print(f"ERRO AO SALVAR ENCODERS: {e}")
            return JsonResponse({'status': 'error', 'message': f'Modelo treinado, mas erro ao salvar encoders: {e}'}, status=500)

        return JsonResponse({'status': 'success', 'resultado': resultado})

    except Exception as e:

        print(f"ERRO INESPERADO na view treinar: {e}")

        traceback.print_exc()

        return JsonResponse({'status': 'error', 'message': f'Um erro inesperado ocorreu: {e}'}, status=500)

    finally:

        if cursor:

            cursor.close()

        if db_connection and db_connection.is_connected():

            db_connection.close()


@csrf_exempt
def salvar_dados_teste(request):

    if request.method == 'POST':

        try:

            data = request.POST.dict()

            dados_teste = DadosTeste(**data)

            dados_teste.save()

            return JsonResponse({'status': 'success', 'message': 'Dados salvos com sucesso!'})

        except Exception as e:

            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    else:

        return JsonResponse({'error': 'Método GET não permitido para esta URL'}, status=405)


@csrf_exempt
def debug_db_view(request):
    try:
        # --- 1. A que base de dados é que o Gunicorn PENSA que está ligado? ---
        db_settings = settings.DATABASES['default']
        db_name = db_settings.get('NAME')
        db_user = db_settings.get('USER')
        db_host = db_settings.get('HOST')

        # --- 2. O que é que o ORM (o seu método) vê? ---
        DadosTreino = apps.get_model("app", "DadosTreino")
        orm_count = DadosTreino.objects.count()

        # --- 3. O que é que o SQL puro (dentro do Django) vê? ---
        raw_count = 0
        with django_db_connection.cursor() as cursor:
            # Usamos o nome da tabela que sabemos estar correto
            cursor.execute("SELECT COUNT(*) FROM app_dados_treino")
            raw_count = cursor.fetchone()[0]

        # Retorna todas as respostas
        return JsonResponse({
            "status": "Informação de Debug da Base de Dados",
            "SETTINGS_DB_NAME": db_name,
            "SETTINGS_DB_USER": db_user,
            "SETTINGS_DB_HOST": db_host,
            "ORM_COUNT (o que a sua view vê)": orm_count,
            "RAW_SQL_COUNT (o que o MySQL vê)": raw_count
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "traceback": traceback.format_exc()}, status=500)

@csrf_exempt
def treinar(request):

    db_connection = None

    cursor = None

    try:

        db_config = settings.DATABASES['default']

        try:

            db_connection = mysql.connector.connect(

                host=db_config['HOST'],

                user=db_config['USER'],

                password=db_config['PASSWORD'],

                database=db_config['NAME'],

                port=db_config.get('PORT', 3306)

            )

            cursor = db_connection.cursor()

        except mysql.connector.Error as err:

            print(f"FALHA ao conectar ao DB: {err}")

            return JsonResponse({'status': 'error', 'message': f'Erro ao conectar ao banco de dados: {err}'}, status=500)

        query = "SELECT age, sex, chest_pain_type, resting_blood_pressure, cholesterol, fasting_blood_sugar, resting_ecg, max_heart_rate, exercise_angina, oldpeak, st_slope, ca, thal, heart_disease_status FROM app_dados_treino"

        df = pd.read_sql(query, db_connection)

        if df.empty:

            return JsonResponse({'status': 'error', 'message': 'Nenhum dado encontrado no banco para treinar.'}, status=400)

        df.dropna(axis=1, thresh=int(0.5 * len(df)), inplace=True)

        encoders = {}

        for col in df.columns:

            if df[col].isnull().any():

                if df[col].dtype == 'object':

                    df[col].fillna(df[col].mode()[0], inplace=True)

                else:

                    df[col].fillna(df[col].median(), inplace=True)

        for col in df.columns:

            if df[col].dtype == "object":

                if col in df:

                    le = LabelEncoder()

                    df[col] = le.fit_transform(df[col])

                else:

                    print(
                        f"Aviso: Coluna {col} não encontrada após processamento inicial.")

        if "heart_disease_status" not in df.columns:

            print(
                "Erro: Coluna 'heart_disease_status' não encontrada após pré-processamento.")

            return JsonResponse({'status': 'error', 'message': 'Coluna alvo "heart_disease_status" perdida durante o pré-processamento.'}, status=500)

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

        buffer = io.StringIO()

        buffer.write("=== Summary ===\n")

        buffer.write(

            f"Correctly Classified Instances  {accuracy_score(y_res, y_pred)*100:.2f} %\n")

        buffer.write(

            f"Incorrectly Classified Instances {(1-accuracy_score(y_res, y_pred))*100:.2f} %\n")

        buffer.write(f"Standard deviation {y_pred.std():.4f}\n\n")

        buffer.write("=== Detailed Accuracy By Class ===\n")

        report = classification_report(y_res, y_pred, output_dict=True)

        buffer.write("                 Precision  Recall  F1-Score\n")

        for label, metrics in report.items():

            if isinstance(metrics, dict) and label not in ['accuracy', 'macro avg', 'weighted avg']:

                try:

                    buffer.write(
                        f"Class {label:<5}      {metrics['precision']:.3f}   {metrics['recall']:.3f}   {metrics['f1-score']:.3f}\n")

                except KeyError:

                    pass

        buffer.write("\n=== Confusion Matrix ===\n")

        cm = confusion_matrix(y_res, y_pred)

        buffer.write("   a   b   <-- classified as\n")

        for i, row in enumerate(cm):

            label = chr(ord('a') + i)

            if len(row) == 2:

                buffer.write(
                    f" {row[0]:>3} {row[1]:>3} |   {label} = class {i}\n")

            else:

                buffer.write(f" {' '.join(map(lambda x: f'{x:>3}', row))} | {label} = class {i}\n")

        resultado = buffer.getvalue()

        buffer.close()

        clf.fit(X_res, y_res)

        model_save_path = os.path.join(
            settings.BASE_DIR, 'app', 'model', 'decision_tree_model.pkl')

        ClassifierClass = apps.get_app_config('app').Classifier
        classifier_instance = ClassifierClass()
        classifier_instance.save_model(clf, model_save_path)

        try:
            encoders_save_path = os.path.join(
                settings.BASE_DIR, 'app', 'model', 'label_encoders.pkl')
            joblib.dump(encoders, encoders_save_path)
            print(f"Encoders salvos em {encoders_save_path}")
        except Exception as e:
            print(f"ERRO AO SALVAR ENCODERS: {e}")
            return JsonResponse({'status': 'error', 'message': f'Modelo treinado, mas erro ao salvar encoders: {e}'}, status=500)

        return JsonResponse({'status': 'success', 'resultado': resultado})

    except Exception as e:

        print(f"ERRO INESPERADO na view treinar: {e}")

        traceback.print_exc()

        return JsonResponse({'status': 'error', 'message': f'Um erro inesperado ocorreu: {e}'}, status=500)

    finally:

        if cursor:

            cursor.close()

        if db_connection and db_connection.is_connected():

            db_connection.close()


@csrf_exempt
def salvar_dados_teste(request):

    if request.method == 'POST':

        try:

            data = request.POST.dict()

            dados_teste = DadosTeste(**data)

            dados_teste.save()

            return JsonResponse({'status': 'success', 'message': 'Dados salvos com sucesso!'})

        except Exception as e:

            return JsonResponse({'status': 'error', 'message': str(e)}, status=400)

    else:

        return JsonResponse({'error': 'Método GET não permitido para esta URL'}, status=405)


@csrf_exempt
def debug_db_view(request):
    try:
        # --- 1. A que base de dados é que o Gunicorn PENSA que está ligado? ---
        db_settings = settings.DATABASES['default']
        db_name = db_settings.get('NAME')
        db_user = db_settings.get('USER')
        db_host = db_settings.get('HOST')

        # --- 2. O que é que o ORM (o seu método) vê? ---
        DadosTreino = apps.get_model("app", "DadosTreino")
        orm_count = DadosTreino.objects.count()

        # --- 3. O que é que o SQL puro (dentro do Django) vê? ---
        raw_count = 0
        with django_db_connection.cursor() as cursor:
            # Usamos o nome da tabela que sabemos estar correto
            cursor.execute("SELECT COUNT(*) FROM app_dados_treino")
            raw_count = cursor.fetchone()[0]

        # Retorna todas as respostas
        return JsonResponse({
            "status": "Informação de Debug da Base de Dados",
            "SETTINGS_DB_NAME": db_name,
            "SETTINGS_DB_USER": db_user,
            "SETTINGS_DB_HOST": db_host,
            "ORM_COUNT (o que a sua view vê)": orm_count,
            "RAW_SQL_COUNT (o que o MySQL vê)": raw_count
        })

    except Exception as e:
        return JsonResponse({"error": str(e), "traceback": traceback.format_exc()}, status=500)

def resultado_anamnese(request):
    mensagem = request.session.pop("mensagem", None)
    sucesso = request.session.pop("sucesso", None)

    if mensagem is None:
        mensagem = "Acesso inválido."
        sucesso = False

    return render(request, "resultado_anamnese.html", {
        "mensagem": mensagem,
        "sucesso": sucesso
    })

from django.shortcuts import render, redirect
from .models import DadosPacientes

def pacientes(request):
    pacientes = DadosPacientes.objects.all()

    for p in pacientes:
        p.nome_exibicao = p.nome_paciente.replace("_", " ")

    if request.method == "POST":
        paciente_id = request.POST.get("paciente")
        if paciente_id:
            return redirect('resultado', paciente_id=paciente_id)

    return render(request, 'pacientes.html', {'pacientes': pacientes})


def resultado(request, paciente_id):
    paciente = get_object_or_404(DadosPacientes, id=paciente_id)
    
    # Substituir '_' por espaço no nome do paciente
    paciente.nome_exibicao = paciente.nome_paciente.replace('_', ' ')

    # Dicionário de tradução dos indicadores
    TRADUCOES = {
        "blood_pressure": {"Low": "Baixa", "Medium": "Média", "High": "Alta", "Unknown": "Desconhecido"},
	"high_blood_pressure": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
        "low_hdl_cholesterol": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
        "high_ldl_cholesterol": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
        "alcohol_consumption": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "stress_level": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "sleep_hours": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "sugar_consumption": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "triglyceride_level": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "fasting_blood_sugar": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "crp_level": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "homocysteine_level": {"High": "Alto", "Low": "Baixo", "Unknown": "Desconhecido"},
        "family_heart_disease": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
        "diabetes": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
        # Novos campos traduzidos
        "gender": {"Male": "Masculino", "Female": "Feminino", "Other": "Outro"},
        "cholesterol_level": {"Low": "Baixo", "Medium": "Médio", "High": "Alto"},
        "exercise_habits": {"Low": "Baixo", "Medium": "Médio", "High": "Alto"},
        "smoking": {"Yes": "Sim", "No": "Não", "Unknown": "Desconhecido"},
    }

    # Traduzir os indicadores
    paciente_traduzido = {}
    for campo, traducoes in TRADUCOES.items():
        valor = getattr(paciente, campo, None)
        paciente_traduzido[campo] = traducoes.get(valor, valor)

    if request.method == 'POST':
        # Atualizar anotações
        if 'anotacoes' in request.POST:
            paciente.anotacoes = request.POST.get('anotacoes', '')
            paciente.save()
            return redirect('resultado', paciente_id=paciente_id)

        # Atualizar status de doença cardiaca
        if 'heart_disease_status' in request.POST:
            paciente.heart_disease_status = request.POST.get('heart_disease_status')
            paciente.save()
            return redirect('resultado', paciente_id=paciente_id)

    return render(
        request,
        'resultado.html',
        {
            'paciente': paciente,
            'paciente_traduzido': paciente_traduzido
        }
    )


def salvar_anotacoes(request, paciente_id):
    if request.method == "POST":
        paciente = get_object_or_404(DadosPacientes, id=paciente_id)
        paciente.anotacoes = request.POST.get("anotacoes", "")
        paciente.save()
    return redirect('resultado', paciente_id=paciente_id)

def salvar_heart_status(request, paciente_id):
    if request.method == "POST":
        paciente = get_object_or_404(DadosPacientes, id=paciente_id)
        paciente.heart_disease_status = request.POST.get("heart_disease_status", "")
        paciente.save()
    return redirect('resultado', paciente_id=paciente_id)
