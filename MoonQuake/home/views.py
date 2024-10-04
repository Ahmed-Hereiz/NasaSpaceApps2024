from django.shortcuts import render


def solutionView(request):
    return render(request, 'home/solution.html')

def homeView(request):
    return render(request, 'home/home.html')

def aboutView(request):
    return render(request, 'home/about.html')