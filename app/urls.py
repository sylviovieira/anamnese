from django.contrib import admin

from django.urls import path

from app import views

from app.views import treinar_modelo_view

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.interface, name='interface'),
    path('anamnese/', views.anamnese, name='anamnese'),
    path('resultado_anamnese/', views.resultado_anamnese, name='resultado_anamnese'),
    path('treinar_modelo/', treinar_modelo_view, name='treinar_modelo'),
    path('treinar/', views.treinar_page, name='treinar_page'),
    path('pacientes/', views.pacientes, name='pacientes'),
    path('resultado/<int:paciente_id>/', views.resultado, name='resultado'),
]
