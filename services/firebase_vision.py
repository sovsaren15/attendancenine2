"""
Sample helpers to integrate Firebase Storage + Google Cloud Vision for face detection.
This module does NOT perform face identification (matching a face to a known person).
For identification you can:
 - Use AWS Rekognition or Face++ (external API) which support face search/collections
 - Run a container or Cloud Run function with dlib/face_recognition prebuilt wheel
 - Use a managed ML service that supports face identification

Usage (high-level):
 - Configure GOOGLE_APPLICATION_CREDENTIALS to point to your service account JSON that has
   access to both Cloud Storage and Cloud Vision, or initialize firebase-admin with credentials.
 - Upload images to Firebase Storage (or accept uploads from client which directly uploads to Storage)
 - Call detect_faces() with the Image content or GCS URI

This file shows examples for:
 - Initializing firebase-admin
 - Uploading image bytes to Firebase Storage
 - Sending image to Google Cloud Vision for face detection

"""
from typing import List, Dict, Optional
import os
import io

from google.cloud import vision
import firebase_admin
from firebase_admin import credentials, storage


def init_firebase(service_account: Optional[str] = None, storage_bucket: Optional[str] = None):
    """Initialize firebase-admin SDK.

    service_account can be one of:
    - None: rely on GOOGLE_APPLICATION_CREDENTIALS or default credentials in the environment
    - a filesystem path to a service account JSON file
    - a JSON string containing the service account (contains 'private_key')

    The function will create a temporary file for JSON content if needed. The temp
    file is not deleted immediately because firebase-admin loads it; it's created
    in the system temp directory.
    """
    if not firebase_admin._apps:
        if service_account:
            # If service_account looks like a path to an existing file, use it directly
            if os.path.exists(service_account):
                cred = credentials.Certificate(service_account)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': storage_bucket
                })
                return

            # If it looks like JSON content (contains a private_key field), write it to a temp file
            if isinstance(service_account, str) and 'private_key' in service_account:
                import tempfile
                fd, path = tempfile.mkstemp(prefix='firebase_sa_', suffix='.json')
                os.close(fd)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(service_account)
                cred = credentials.Certificate(path)
                firebase_admin.initialize_app(cred, {
                    'storageBucket': storage_bucket
                })
                return

        # Fallback: use default credentials (GOOGLE_APPLICATION_CREDENTIALS or GCE metadata)
        firebase_admin.initialize_app(options={'storageBucket': storage_bucket})


def upload_image_bytes(image_bytes: bytes, destination_path: str) -> str:
    """Upload image bytes to the configured Firebase Storage bucket.
    Returns the public GCS path (gs://bucket/path) for later use.
    """
    bucket = storage.bucket()
    blob = bucket.blob(destination_path)
    blob.upload_from_string(image_bytes, content_type='image/jpeg')
    return f"gs://{bucket.name}/{destination_path}"


def detect_faces_from_bytes(image_bytes: bytes) -> List[Dict]:
    """Call Google Cloud Vision face detection on raw image bytes. Returns a list
    of face annotation dicts (bounding boxes, detection confidence, landmarks).
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(content=image_bytes)
    response = client.face_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    faces = []
    for face in response.face_annotations:
        bbox = [(v.x, v.y) for v in face.bounding_poly.vertices]
        faces.append({
            'bounding_poly': bbox,
            'detection_confidence': face.detection_confidence,
            'joy_likelihood': face.joy_likelihood,
            'sorrow_likelihood': face.sorrow_likelihood,
            'anger_likelihood': face.anger_likelihood,
            'surprise_likelihood': face.surprise_likelihood,
            'landmarks': [{ 'type': l.type_.name, 'position': (l.position.x, l.position.y, l.position.z) } for l in face.landmarks]
        })
    return faces


def detect_faces_from_gcs_uri(gcs_uri: str) -> List[Dict]:
    """Run face detection on an image stored in GCS (gs://bucket/path).
    """
    client = vision.ImageAnnotatorClient()
    image = vision.Image(source=vision.ImageSource(gcs_image_uri=gcs_uri))
    response = client.face_detection(image=image)
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")

    faces = []
    for face in response.face_annotations:
        bbox = [(v.x, v.y) for v in face.bounding_poly.vertices]
        faces.append({
            'bounding_poly': bbox,
            'detection_confidence': face.detection_confidence,
        })
    return faces


# Simple convenience wrapper
def process_uploaded_image(image_bytes: bytes, destination_path: str) -> Dict:
    """Uploads to Firebase Storage (if configured) and runs face detection.
    Returns a dict with storage_uri and face annotations.
    """
    storage_uri = None
    if firebase_admin._apps:
        storage_uri = upload_image_bytes(image_bytes, destination_path)
    faces = detect_faces_from_bytes(image_bytes)
    return {
        'storage_uri': storage_uri,
        'faces': faces
    }
