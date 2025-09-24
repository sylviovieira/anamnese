from django.shortcuts import render

def interface(request):
    return render(request, 'interface.html')

def anamnese(request):
    return render(request, 'Anamnese Intuitiva Orientada por IA.html')

def resultado(request):
    return render(request, 'Resultado da Análise de Pré-Diagnóstico para o Profissional de Saúde.html')