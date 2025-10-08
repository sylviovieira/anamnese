from django.shortcuts import render

def interface(request):
    return render(request, 'interface.html')

def anamnese(request):
    return render(request, 'anamnese.html')

def resultado(request):
    return render(request, 'resultado.html')