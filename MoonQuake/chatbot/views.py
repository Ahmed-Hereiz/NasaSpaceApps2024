from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import ensure_csrf_cookie
import json
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
from django.shortcuts import render


def ChatBotView(request):
    return render(request, 'chatbot/chatbot.html')
    

@ensure_csrf_cookie
@require_http_methods(["POST"])
def chat(request):
    try:
        data = json.loads(request.body)
        message = data.get('message', '')

        if not message:
            return JsonResponse({'error': 'No message provided'}, status=400)

        response = f"You said: {message}"  

        return JsonResponse({'message': response})
    except json.JSONDecodeError:
        return JsonResponse({'error': 'Invalid JSON'}, status=400)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
