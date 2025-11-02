from django.contrib import admin
from django.urls import path
from app import views
from app.views import treinar_modelo_view


urlpatterns = [

    path('admin/', admin.site.urls),

    path('', views.interface, name='interface'),

    path('anamnese/', views.anamnese, name='anamnese'),

    path('resultado/', views.resultado, name='resultado'),

    path('resultado_anamnese/', views.resultado_anamnese, name='resultado_anamnese'),

    path('treinar_modelo/', treinar_modelo_view, name='treinar_modelo'),

    path('treinar/', views.treinar_page, name='treinar_page'),

    path('debug-db/', views.debug_db_view, name='debug_db'),

    path('pacientes/', views.pacientes, name='pacientes'),

    path('resultado/<int:paciente_id>/', views.resultado, name='resultado'),

    path("salvar_anotacoes/<int:paciente_id>/", views.salvar_anotacoes, name="salvar_anotacoes"),

    path("salvar_heart_status/<int:paciente_id>/", views.salvar_heart_status, name="salvar_heart_status"),


]

