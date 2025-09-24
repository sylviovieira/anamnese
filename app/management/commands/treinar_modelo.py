import pandas as pd
from django.core.management.base import BaseCommand
from django.apps import apps
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import cross_val_predict, cross_val_score, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class Command(BaseCommand):
    help = "Treina modelo de árvore de decisão usando dados da tabela app_dados_treino"

    def handle(self, *args, **kwargs):
        # carrega modelo do Django
        DadosTreino = apps.get_model("app", "DadosTreino")

        # busca dados do banco
        queryset = DadosTreino.objects.all().values()
        df = pd.DataFrame(list(queryset))

        if df.empty:
            self.stdout.write(self.style.ERROR("Tabela app_dados_treino está vazia!"))
            return

        # encode de variáveis categóricas
        le_dict = {}
        for col in df.columns:
            if df[col].dtype == "object":
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
                le_dict[col] = le

        # separa atributos e alvo
        X = df.drop("heart_disease_status", axis=1)
        y = df["heart_disease_status"]

        # balanceamento com SMOTE
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)

        # modelo
        clf = DecisionTreeClassifier(
            random_state=42,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5
        )

        # validação cruzada
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
        y_pred = cross_val_predict(clf, X_res, y_res, cv=cv)
        scores = cross_val_score(clf, X_res, y_res, cv=cv)

        # saída
        self.stdout.write(self.style.SUCCESS("=== Summary ==="))
        self.stdout.write(f"Correctly Classified Instances  {accuracy_score(y_res, y_pred)*100:.2f} %")
        self.stdout.write(f"Incorrectly Classified Instances {(1-accuracy_score(y_res, y_pred))*100:.2f} %")
        self.stdout.write(f"Mean Accuracy (10-fold CV)       {scores.mean()*100:.2f} % ± {scores.std()*100:.2f}%\n")

        self.stdout.write("=== Confusion Matrix ===")
        self.stdout.write(str(confusion_matrix(y_res, y_pred)))

        self.stdout.write("\n=== Classification Report ===")
        self.stdout.write(classification_report(y_res, y_pred))
