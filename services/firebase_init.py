"""Handle Firebase initialization"""
import os
import firebase_admin
from services.firebase_vision import init_firebase

def init_firebase_app():
    """Initialize Firebase if not already initialized"""
    if not firebase_admin._apps:
        service_account_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                          'keys', 
                                          'attendance-cad1c-firebase-adminsdk-fbsvc-af3ed3b1f6.json')
        if os.path.exists(service_account_path):
            try:
                init_firebase(service_account=service_account_path, 
                            storage_bucket='attendance-cad1c.appspot.com')
                print(f'Firebase initialized with service account from {service_account_path}')
                return True
            except Exception as e:
                print(f"Error initializing Firebase: {e}")
                return False
        else:
            print(f"Warning: Service account file not found at {service_account_path}")
            return False
    return True  # Already initialized