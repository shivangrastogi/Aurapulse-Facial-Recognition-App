from server import app
from firebase_functions import https

@https.on_request
def api(request):
    return app(request)
