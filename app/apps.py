from django.apps import AppConfig
import pickle
import os




class AppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'app'


    class Classifier:
        """
        Classe auxiliar para carregar e salvar o modelo de machine learning.
        """
        def load_model(self, path):
            """
            Carrega o modelo a partir do caminho especificado.
            Retorna o modelo carregado ou None se o ficheiro não existir.
            """
            if os.path.exists(path):
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"AVISO: Ficheiro do modelo não encontrado em {path}")
                return None

        def save_model(self, model, path):
            """
            Salva o modelo no caminho especificado.
            Cria diretórios se não existirem.
            """
            model_dir = os.path.dirname(path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)

            with open(path, 'wb') as f:
                pickle.dump(model, f)
            print(f"Modelo salvo com sucesso em {path}")
