from django import forms
from django.shortcuts import render
import requests
from django.http import JsonResponse
from .forms import FileForm
  
api_base_url = 'http://127.0.0.1:8080'


def analyzerView(request):
    if request.method == 'POST':
        form = FileForm(request.POST, request.FILES)
        if form.is_valid():
            uploaded_file = request.FILES.get('file')
            if uploaded_file:
                try:
                    analyzer_endpoint_url = f"{api_base_url}/get-stalta-startpoint"
                    
                    files = {'file': uploaded_file}
                    response = requests.post(
                        analyzer_endpoint_url, files=files)
                    response.raise_for_status()
                    if response.status_code == 200:
                        startPoint = response.json().get("startPoint")
                        img_url = response.json().get("path")
                        # if startPoint :
                        #     api_get_img_endpoint_url = f"{api_base_url}/get-chart-from-quake-startpoint"
                        #     response = requests.post(
                        #         api_get_img_endpoint_url, json={"startPoint": str(startPoint)})
                        #     img = response.json().get("image_url")
                        return render(request, 'analyzer/analyzer.html', {
                            "form": form,
                            'startPoint': startPoint,
                            'image_url': img_url
                        })
                except requests.RequestException as e:
                    return render(request, 'analyzer/analyzer.html', {
                        "form": form,
                        'error': f'Failed to analyze the file: {str(e)}'
                    })
            else:
                return render(request, 'analyzer/analyzer.html', {
                    "form": form,
                    'error': 'No file uploaded'
                })
    else:
        form = FileForm()

    return render(request, 'analyzer/analyzer.html', {
        "form": form,
        'startPoint': "Upload a file to begin analysis"
    })