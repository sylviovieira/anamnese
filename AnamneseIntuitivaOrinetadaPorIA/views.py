from django.shortcuts import render

def interface(request):
    return render(request, 'interface')

def anamnese(request):
    return render(request, 'anamnese')

def resultado(request):
    return render(request, 'resultado')