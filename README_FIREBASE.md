Firebase integration notes

Goal
----
Replace local dlib/face_recognition-based workflows (which require native builds) with a Firebase-backed approach that uses Google Cloud Vision for face detection and Firebase Storage/Firestore for storing images and metadata.

What this repository changes
---------------------------
- Removed the dlib/face-recognition dependencies from `requirements.txt`.
- Added `firebase-admin` and `google-cloud-vision` to `requirements.txt`.
- Added `services/firebase_vision.py` with helper functions to upload images to Firebase Storage and call Cloud Vision face detection.
- Updated `build.sh` to be robust in environments without `apt-get`.

Identification vs. detection
-----------------------------
- Google Cloud Vision provides face detection (bounding boxes, landmarks, and emotion likelihoods) but does NOT provide face identification (matching a face to a person ID).
- For face identification you can:
  - Use AWS Rekognition (supports face collections and search)
  - Use Face++ (external API)
  - Run a dedicated service (Cloud Run / Docker) with `dlib`/`face_recognition` preinstalled and expose an API

How to configure
----------------
1. Create a Google Cloud project and enable Cloud Vision and Cloud Storage APIs.
2. Create a service account with Storage and Vision permissions and download the JSON key.
3. Set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` to the JSON file path in your deployment environment, or initialize firebase-admin with the JSON file path.
4. Set the Firebase Storage bucket name in your app (see `services/firebase_vision.py`).

Example usage (server-side)
---------------------------
from services.firebase_vision import init_firebase, process_uploaded_image

init_firebase(service_account_json='/path/to/key.json', storage_bucket='your-bucket-name')
with open('some_photo.jpg','rb') as f:
    data = f.read()
result = process_uploaded_image(data, 'uploads/some_photo.jpg')
print(result['faces'])

Next steps
----------
- If identification is required, pick an external API or create a Cloud Run service that runs face identification and call it from your Flask app.
- Add Firestore collections to store known face metadata and search results.
