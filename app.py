"""Attendance app (Firestore-first, SQL fallback).

This is a cleaned, single-version `app.py` that initializes Firebase robustly
(from env vars or a local `keys/` JSON), falls back to SQL when
`USE_FIRESTORE` is disabled, and exposes the main routes used by the UI.
"""

import os
import math
import base64
import pickle
from io import BytesIO
from datetime import datetime
from types import SimpleNamespace

try:
    import cv2
except Exception:
    cv2 = None
try:
    import face_recognition
except Exception:
    face_recognition = None
try:
    import numpy as np
except Exception:
    np = None

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
try:
    from flask_migrate import Migrate
except Exception:
    Migrate = None

import sqlalchemy as sa
import pytz
from PIL import Image

# project services
from services.firebase_vision import init_firebase
from services import firestore_db

# Default to Firestore unless the env var explicitly disables it
env_use_fs = os.environ.get('USE_FIRESTORE')
if env_use_fs is None:
    USE_FIRESTORE = True
else:
    USE_FIRESTORE = str(env_use_fs).lower() in ('1', 'true', 'yes')


def _find_service_account_path():
    """Return a candidate service account path or raw JSON string.

    Order of preference:
      - FIREBASE_SERVICE_ACCOUNT_JSON (raw JSON content)
      - FIREBASE_SERVICE_ACCOUNT_PATH (explicit path)
      - GOOGLE_APPLICATION_CREDENTIALS (path)
      - first .json file in ./keys/
    """
    sa_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT_JSON')
    if sa_json:
        return sa_json

    sa_path = os.environ.get('FIREBASE_SERVICE_ACCOUNT_PATH')
    if sa_path and os.path.exists(sa_path):
        return sa_path

    gac = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
    if gac and os.path.exists(gac):
        return gac

    keys_dir = os.path.join(os.path.dirname(__file__), 'keys')
    if os.path.isdir(keys_dir):
        for fn in os.listdir(keys_dir):
            if fn.lower().endswith('.json'):
                return os.path.join(keys_dir, fn)
    return None


def initialize_firebase():
    """Attempt to initialize firebase-admin. Returns True if initialized."""
    try:
        import firebase_admin
        if firebase_admin._apps:
            return True
    except Exception:
        # firebase_admin may not be installed or importable yet; try to initialize below
        pass

    sa = _find_service_account_path()
    storage_bucket = os.environ.get('FIREBASE_STORAGE_BUCKET')
    try:
        # init_firebase handles None service_account (it may use default credentials)
        init_firebase(service_account=sa, storage_bucket=storage_bucket)
        print('Firebase initialized')
        return True
    except Exception as e:
        print(f'Warning: Firebase initialization failed: {e}')
        return False


app = Flask(__name__)

# Configure DB only if not using Firestore
if not USE_FIRESTORE:
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL', 'sqlite:///attendance')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
else:
    app.config.pop('SQLALCHEMY_DATABASE_URI', None)
    app.config.pop('SQLALCHEMY_TRACK_MODIFICATIONS', None)

app.secret_key = os.environ.get('FLASK_SECRET', 'your_super_secret_key_here')
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# Geolocation and misc config
app.config['COMPANY_LATITUDE'] = float(os.environ.get('COMPANY_LATITUDE', 13.37488193943832))
app.config['COMPANY_LONGITUDE'] = float(os.environ.get('COMPANY_LONGITUDE', 103.842437801041))
app.config['MAX_DISTANCE_METERS'] = int(os.environ.get('MAX_DISTANCE_METERS', 2000))
app.config['DEBUG_USE_COMPANY_LOCATION'] = True


if not USE_FIRESTORE:
    db = SQLAlchemy(app)
    migrate = Migrate(app, db) if Migrate else None
else:
    db = None
    migrate = None


CAMBODIA_TZ = pytz.timezone('Asia/Phnom_Penh')


def get_cambodia_time():
    return datetime.now(CAMBODIA_TZ)


def get_cambodia_date():
    return get_cambodia_time().date()


# Database Models (only when SQLAlchemy is enabled)
if not USE_FIRESTORE:
    class Employee(db.Model):
        __tablename__ = 'employees'
        id = db.Column(db.Integer, primary_key=True)
        name = db.Column(db.String(100), nullable=False, unique=True)
        gender = db.Column(db.String(10), nullable=False)
        date_of_birth = db.Column(db.Date, nullable=False)
        position = db.Column(db.String(100), nullable=False)
        address = db.Column(db.Text, nullable=False)
        face_encoding = db.Column(db.PickleType, nullable=False)


    class Attendance(db.Model):
        __tablename__ = 'attendances'
        id = db.Column(db.Integer, primary_key=True)
        employee_id = db.Column(db.Integer, db.ForeignKey('employees.id'), nullable=False)
        check_in = db.Column(db.DateTime(timezone=True), nullable=False, default=get_cambodia_time)
        check_out = db.Column(db.DateTime(timezone=True), nullable=True)
        check_in_status = db.Column(db.String(10), nullable=True)  # e.g., 'Early', 'Good', 'Late'

        employee = db.relationship('Employee', backref=db.backref('attendances', lazy=True))


with app.app_context():
    try:
        # Initialize Firebase if running in Firestore mode (best-effort)
        if USE_FIRESTORE:
            initialize_firebase()

        # Only create SQL tables when not using Firestore
        if not USE_FIRESTORE and db is not None:
            db.create_all()
            print('Database tables ensured to be created.')
        else:
            print('Firestore mode enabled - skipping SQL table creation.')
    except Exception as e:
        print(f'Error ensuring database tables are created: {e}')


KNOWN_FACES_DIR = 'known_faces'
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


# Global cache for face encodings
known_face_encodings_cache = []
known_face_names_cache = []

def load_known_faces():
    """Load known faces from either SQL DB or Firestore depending on `USE_FIRESTORE` flag."""
    known_face_encodings = []
    known_face_names = []
    try:
        if USE_FIRESTORE:
            try:
                emps = firestore_db.get_all_employees()
            except Exception as e:
                print(f"Unable to fetch employees from Firestore: {e}")
                emps = []
            for e in emps:
                b64 = e.get('face_encoding_b64')
                if not b64:
                    continue
                try:
                    enc_bytes = base64.b64decode(b64)
                    arr = np.array(pickle.loads(enc_bytes))
                except Exception:
                    continue
                known_face_encodings.append(arr)
                known_face_names.append(e.get('name'))
        else:
            employees = Employee.query.all()
            for employee in employees:
                known_face_encodings.append(np.array(pickle.loads(employee.face_encoding)))
                known_face_names.append(employee.name)
    except Exception as exc:
        print(f'Error loading known faces: {exc}')
    return known_face_encodings, known_face_names


def reload_known_faces():
    global known_face_encodings_cache, known_face_names_cache
    known_face_encodings_cache, known_face_names_cache = load_known_faces()


def process_image_for_encoding(image_data_b64):
    try:
        image_data = base64.b64decode(image_data_b64)
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)

        if image_np.ndim == 3 and image_np.shape[2] == 4:  # RGBA
            image_np = image_np[:, :, :3]

        encodings = face_recognition.face_encodings(image_np)
        if not encodings:
            return None
        return encodings[0]
    except Exception as e:
        print(f"Error processing image for encoding: {e}")
        return None


def format_cambodia_datetime(dt):
    """Format datetime for display in Cambodia timezone"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    cambodia_dt = dt.astimezone(CAMBODIA_TZ)
    return cambodia_dt.strftime("%Y-%m-%d %H:%M:%S %Z")


def format_cambodia_time(dt):
    """Format time only for display in Cambodia timezone"""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    cambodia_dt = dt.astimezone(CAMBODIA_TZ)
    return cambodia_dt.strftime("%H:%M:%S")


def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate the Haversine distance in meters."""
    R = 6371000
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        gender = request.form.get('gender', '').strip()
        date_of_birth = request.form.get('date_of_birth')
        position = request.form.get('position', '').strip()
        address = request.form.get('address', '').strip()
        image_data = request.form.get('image')

        if not image_data or ',' not in image_data:
            return jsonify({'error': 'Image is required.'}), 400

        image_b64 = image_data.split(',', 1)[1]

        if not all([name, gender, date_of_birth, position, address, image_b64]):
            return jsonify({'error': 'All fields and an image are required.'}), 400

        if not USE_FIRESTORE and Employee.query.filter_by(name=name).first():
            return jsonify({'error': 'An employee with this name already exists.'}), 400

        try:
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            encoding = process_image_for_encoding(image_b64)
            if encoding is None:
                return jsonify({'error': 'No face detected in the image.'}), 400
            serialized = pickle.dumps(encoding)

            if USE_FIRESTORE:
                initialize_firebase()
                b64 = base64.b64encode(serialized).decode('utf-8')
                firestore_db.create_employee(
                    name=name,
                    gender=gender,
                    date_of_birth=dob.isoformat(),
                    position=position,
                    address=address,
                    face_encoding_b64=b64,
                )
                reload_known_faces() # Reload cache after adding new employee
                return jsonify({'message': f'Employee {name} registered successfully (stored in Firestore)!'}), 200
            else:
                employee = Employee(
                    name=name,
                    gender=gender,
                    date_of_birth=dob,
                    position=position,
                    address=address,
                    face_encoding=serialized,
                )
                db.session.add(employee)
                db.session.commit()
                reload_known_faces() # Reload cache after adding new employee
                return jsonify({'message': f'Employee {name} registered successfully!'}), 200
        except ValueError:
            if db is not None:
                db.session.rollback()
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400
        except Exception as e:
            if db is not None:
                db.session.rollback()
            return jsonify({'error': f'An unexpected error occurred: {e}'}), 500

    return render_template('register.html')


@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        action = request.form.get('action')
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)

        if latitude is None or longitude is None:
            if app.debug and app.config.get('DEBUG_USE_COMPANY_LOCATION'):
                latitude = app.config['COMPANY_LATITUDE']
                longitude = app.config['COMPANY_LONGITUDE']
            else:
                return jsonify({'error': 'Location data is missing.'}), 400

        distance = calculate_distance(latitude, longitude, app.config['COMPANY_LATITUDE'], app.config['COMPANY_LONGITUDE'])
        if distance > app.config['MAX_DISTANCE_METERS']:
            return jsonify({'error': 'You are too far from the company location.'}), 403

        try:
            image_data = request.form.get('image')
            if not image_data or ',' not in image_data:
                return jsonify({'error': 'Image is required.'}), 400
            image_b64 = image_data.split(',', 1)[1]
            image_bytes = base64.b64decode(image_b64)
            image = Image.open(BytesIO(image_bytes))
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) if cv2 is not None else image_np

            face_locations = face_recognition.face_locations(image_cv)
            face_encodings = face_recognition.face_encodings(image_cv, face_locations)

            if not face_encodings:
                return jsonify({'error': 'No face detected.'}), 400

            # Use the global cache instead of loading from DB on every request
            known_face_encodings, known_face_names = known_face_encodings_cache, known_face_names_cache

            matched_employee = None
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    idx = matches.index(True)
                    matched_name = known_face_names[idx]
                    if USE_FIRESTORE:
                        matched_employee = firestore_db.find_employee_by_name(matched_name)
                    else:
                        matched_employee = Employee.query.filter_by(name=matched_name).first()
                    break

            if not matched_employee:
                return jsonify({'error': 'No matching employee found.'}), 400

            today = get_cambodia_date()
            cambodia_now = get_cambodia_time()

            if action == 'check_in':
                check_in_status = 'Good'
                check_in_time_only = cambodia_now.time()
                early_time = datetime.strptime('08:00', '%H:%M').time()
                on_time_limit = datetime.strptime('08:15', '%H:%M').time()
                if check_in_time_only < early_time:
                    check_in_status = 'Early'
                elif check_in_time_only > on_time_limit:
                    check_in_status = 'Late'

                if USE_FIRESTORE:
                    attends = firestore_db.get_attendances_for_employee_on_date(matched_employee['id'], today.isoformat())
                    existing_open = next((a for a in attends if not a.get('check_out')), None)
                    if existing_open:
                        return jsonify({'error': f"{matched_employee['name']} is already checked in for today."}), 400
                    firestore_db.add_attendance(matched_employee['id'], cambodia_now.isoformat(), None, check_in_status)
                    return jsonify({'message': f"Check-in recorded for {matched_employee['name']}"})
                else:
                    existing_open = Attendance.query.filter(
                        Attendance.employee_id == matched_employee.id,
                        sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today,
                        Attendance.check_out == None,
                    ).first()
                    if existing_open:
                        return jsonify({'error': 'Already checked in.'}), 400
                    att = Attendance(employee_id=matched_employee.id, check_in=cambodia_now, check_in_status=check_in_status)
                    db.session.add(att)
                    db.session.commit()
                    return jsonify({'message': f'Check-in recorded for {matched_employee.name}'})

            elif action == 'check_out':
                if USE_FIRESTORE:
                    attends = firestore_db.get_attendances_for_employee_on_date(matched_employee['id'], today.isoformat())
                    open_attends = [a for a in attends if not a.get('check_out')]
                    if not open_attends:
                        return jsonify({'error': 'No active check-in found.'}), 400
                    open_attends.sort(key=lambda x: x.get('check_in'), reverse=True)
                    to_close = open_attends[0]
                    firestore_db.update_attendance(to_close['id'], {'check_out': cambodia_now.isoformat()})
                    return jsonify({'message': 'Check-out recorded.'})
                else:
                    att_to_close = Attendance.query.filter(
                        Attendance.employee_id == matched_employee.id,
                        sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today,
                        Attendance.check_out == None,
                    ).order_by(Attendance.check_in.desc()).first()
                    if not att_to_close:
                        return jsonify({'error': 'No active check-in found.'}), 400
                    att_to_close.check_out = cambodia_now
                    db.session.commit()
                    return jsonify({'message': 'Check-out recorded.'})

            else:
                return jsonify({'error': 'Invalid action.'}), 400

        except Exception as e:
            if db is not None:
                db.session.rollback()
            return jsonify({'error': f'Unexpected error: {e}'}), 500

    return render_template('attendance.html')


@app.route('/records')
def records():
    now_utc = datetime.now(pytz.utc)
    if USE_FIRESTORE:
        try:
            raw_attends = firestore_db.get_all_attendances(limit=1000)
        except Exception as e:
            flash(f'Unable to fetch attendances from Firestore: {e}', 'error')
            raw_attends = []
        
        # Fetch all employees and create a lookup map by name for efficiency
        # Create two lookup maps: one by ID and one by name for backward compatibility.
        try:
            employees_list = firestore_db.get_all_employees()
            employees_by_id = {emp['id']: emp for emp in employees_list}
            employees_by_name = {emp['name']: emp for emp in employees_list}
        except Exception:
            employees_by_id, employees_by_name = {}, {}

        attendances = []
        for a in raw_attends:
            ci = None
            co = None
            try:
                if a.get('check_in'):
                    ci = datetime.fromisoformat(a.get('check_in'))
                    if ci.tzinfo is None:
                        ci = pytz.UTC.localize(ci)
                if a.get('check_out'):
                    co = datetime.fromisoformat(a.get('check_out'))
                    if co.tzinfo is None:
                        co = pytz.UTC.localize(co)
            except Exception:
                pass
            obj = SimpleNamespace()
            
            # Find the employee data. Prioritize ID, but fall back to name for old records.
            employee_id = a.get('employee_id')
            employee_data = employees_by_id.get(employee_id)
            if not employee_data:
                employee_name = a.get('employee_name')
                employee_data = employees_by_name.get(employee_name)

            if not employee_data:
                continue # Skip attendance records for employees not found

            obj.employee = SimpleNamespace(**employee_data) # Attach the employee object
            obj.check_in = ci
            obj.check_out = co
            # The template now uses strftime directly, so these are no longer needed
            # obj.check_in_formatted = format_cambodia_datetime(ci) if isinstance(ci, datetime) else (ci or None)
            # obj.check_out_formatted = format_cambodia_datetime(co) if isinstance(co, datetime) else (co or None)
            obj.check_in_status = a.get('check_in_status')
            attendances.append(obj)
        return render_template('records.html', attendances=attendances, now_utc=now_utc)
    else:
        attendances = Attendance.query.order_by(Attendance.check_in.desc()).all()
        for attendance in attendances:
            attendance.check_in_formatted = format_cambodia_datetime(attendance.check_in)
            attendance.check_out_formatted = format_cambodia_datetime(attendance.check_out)
            attendance.check_in_time = format_cambodia_time(attendance.check_in)
            attendance.check_out_time = format_cambodia_time(attendance.check_out)
        return render_template('records.html', attendances=attendances, now_utc=now_utc)


# Admin routes (list/add/edit/delete)
@app.route('/admin')
def admin_dashboard():
    today = get_cambodia_date()
    now = get_cambodia_time()
    if USE_FIRESTORE or db is None:
        try:
            employees_raw = firestore_db.get_all_employees()
        except Exception as e:
            flash(f'Unable to fetch employees from Firestore: {e}', 'error')
            employees_raw = []
        employees = []
        for e in employees_raw:
            emp = SimpleNamespace()
            # Ensure employee has a name before adding to the list for the template
            if not e.get('name'):
                continue
            emp.id = e.get('id')
            emp.name = e.get('name')
            emp.gender = e.get('gender')
            dob_str = e.get('date_of_birth')
            try:
                emp.date_of_birth = datetime.strptime(dob_str, '%Y-%m-%d').date() if dob_str else None
            except (ValueError, TypeError):
                emp.date_of_birth = None # Handle cases where date is missing or malformed
            emp.position = e.get('position')
            emp.address = e.get('address')
            employees.append(emp)
        total_employees = len(employees)
        first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        first_day_iso = first_day_of_month.isoformat()
        try:
            # Optimize: only fetch attendances from the current month for stats
            attends = firestore_db.get_attendances_since(first_day_iso)
        except Exception:
            attends = []
        checked_in_today_set = set()
        late_today_count = 0

        # Create a lookup map from employee ID to employee name for stats
        employee_id_to_name = {emp.id: emp.name for emp in employees}

        monthly_counts = {}
        for a in attends:
            # Use employee_id, then fall back to employee_name for old records
            emp_id = a.get('employee_id')
            ename = employee_id_to_name.get(emp_id)
            if not ename:
                ename = a.get('employee_name') # Fallback for old data

            ci = a.get('check_in')
            if not ename or not ci:
                continue
            if ci.startswith(today.isoformat()):
                checked_in_today_set.add(ename)
            if ci.startswith(today.isoformat()) and a.get('check_in_status') == 'Late':
                late_today_count += 1
            stats = monthly_counts.setdefault(ename, {'attendance': 0, 'late': 0, 'early': 0})
            stats['attendance'] += 1
            if a.get('check_in_status') == 'Late':
                stats['late'] += 1
            if a.get('check_in_status') == 'Early':
                stats['early'] += 1
        checked_in_today_count = len(checked_in_today_set)
        top_late_employees = sorted([(name, v['late']) for name, v in monthly_counts.items() if v['late'] > 0], key=lambda x: x[1], reverse=True)[:3]
        top_attendance_employees = sorted([(name, v['attendance']) for name, v in monthly_counts.items()], key=lambda x: x[1], reverse=True)[:3]
        top_early_employees = sorted([(name, v['early']) for name, v in monthly_counts.items() if v['early'] > 0], key=lambda x: x[1], reverse=True)[:3]
        return render_template('admin.html', employees=employees, total_employees=total_employees, checked_in_today_count=checked_in_today_count, late_today_count=late_today_count, top_late_employees=top_late_employees, top_attendance_employees=top_attendance_employees, top_early_employees=top_early_employees)

    # SQL branch
    employees = Employee.query.order_by(Employee.name).all()
    total_employees = len(employees)
    checked_in_today_count = db.session.query(Attendance.employee_id).filter(sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today).distinct().count()
    late_today_count = Attendance.query.filter(sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today, Attendance.check_in_status == 'Late').count()
    first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    top_late_employees = db.session.query(Employee.name, sa.func.count(Attendance.id).label('late_count')).join(Employee).filter(Attendance.check_in_status == 'Late', Attendance.check_in >= first_day_of_month).group_by(Employee.name).order_by(sa.desc('late_count')).limit(3).all()
    top_attendance_employees = db.session.query(Employee.name, sa.func.count(Attendance.id).label('attendance_count')).join(Employee).filter(Attendance.check_in >= first_day_of_month).group_by(Employee.name).order_by(sa.desc('attendance_count')).limit(3).all()
    top_early_employees = db.session.query(Employee.name, sa.func.count(Attendance.id).label('early_count')).join(Employee).filter(Attendance.check_in_status == 'Early', Attendance.check_in >= first_day_of_month).group_by(Employee.name).order_by(sa.desc('early_count')).limit(3).all()
    return render_template('admin.html', employees=employees, total_employees=total_employees, checked_in_today_count=checked_in_today_count, late_today_count=late_today_count, top_late_employees=top_late_employees, top_attendance_employees=top_attendance_employees, top_early_employees=top_early_employees)


@app.route('/admin/add_employee', methods=['GET', 'POST'])
def add_employee_manual():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        gender = request.form.get('gender', '').strip()
        date_of_birth = request.form.get('date_of_birth')
        position = request.form.get('position', '').strip()
        address = request.form.get('address', '').strip()
        image_file = request.files.get('image_file')

        if not all([name, gender, date_of_birth, position, address]):
            flash('All text fields are required!', 'error')
            return redirect(url_for('add_employee_manual'))

        if USE_FIRESTORE:
            if firestore_db.find_employee_by_name(name):
                flash(f'Employee with name "{name}" already exists.', 'error')
                return redirect(url_for('add_employee_manual'))
        else:
            if Employee.query.filter_by(name=name).first():
                flash(f'Employee with name "{name}" already exists.', 'error')
                return redirect(url_for('add_employee_manual'))

        try:
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            if image_file and image_file.filename:
                temp_image_path = os.path.join(KNOWN_FACES_DIR, f"temp_upload_{name}_{get_cambodia_time().strftime('%Y%m%d%H%M%S')}.jpg")
                image_file.save(temp_image_path)
                img = face_recognition.load_image_file(temp_image_path)
                encodings = face_recognition.face_encodings(img)
                os.remove(temp_image_path)
                if not encodings:
                    flash('No face detected in the uploaded image.', 'error')
                    return redirect(url_for('add_employee_manual'))
                face_bytes = pickle.dumps(encodings[0])
            else:
                flash('An image file is required for facial recognition data.', 'error')
                return redirect(url_for('add_employee_manual'))

            if USE_FIRESTORE:
                b64 = base64.b64encode(face_bytes).decode('utf-8')
                firestore_db.create_employee(name=name, gender=gender, date_of_birth=dob.isoformat(), position=position, address=address, face_encoding_b64=b64)
                reload_known_faces()
                flash(f'Employee "{name}" added successfully (Firestore)!', 'success')
                return redirect(url_for('admin_dashboard'))
            else:
                employee = Employee(name=name, gender=gender, date_of_birth=dob, position=position, address=address, face_encoding=face_bytes)
                db.session.add(employee)
                db.session.commit()
                reload_known_faces()
                flash(f'Employee "{name}" added successfully!', 'success')
                return redirect(url_for('admin_dashboard'))
        except Exception as e:
            if db is not None:
                db.session.rollback()
            flash(f'An error occurred while adding employee: {e}', 'error')
            return redirect(url_for('add_employee_manual'))

    return render_template('add_edit_employee.html', employee=None, action_url=url_for('add_employee_manual'))


@app.route('/admin/edit_employee/<employee_id>', methods=['GET', 'POST'])
def edit_employee_manual(employee_id):
    if USE_FIRESTORE:
        employee = firestore_db.get_employee_by_id(employee_id)
        if not employee:
            return redirect(url_for('admin_dashboard'))
    else:
        try:
            eid = int(employee_id)
        except Exception:
            return redirect(url_for('admin_dashboard'))
        employee = Employee.query.get_or_404(eid)

    if request.method == 'POST':
        if USE_FIRESTORE:
            update_fields = {
                'name': request.form.get('name', '').strip(),
                'gender': request.form.get('gender', '').strip(),
                'date_of_birth': datetime.strptime(request.form.get('date_of_birth'), '%Y-%m-%d').date().isoformat(),
                'position': request.form.get('position', '').strip(),
                'address': request.form.get('address', '').strip(),
            }
            image_file = request.files.get('image_file')
            if image_file and image_file.filename:
                try:
                    img = face_recognition.load_image_file(image_file)
                    encodings = face_recognition.face_encodings(img)
                    if encodings:
                        face_bytes = pickle.dumps(encodings[0])
                        update_fields['face_encoding_b64'] = base64.b64encode(face_bytes).decode('utf-8')
                        reload_known_faces()
                    else:
                        flash('No face detected in the new image. Face data was not updated.', 'warning')
                except Exception as e:
                    flash(f'Error processing new image: {e}. Face data was not updated.', 'error')

            firestore_db.update_employee(employee_id, update_fields)
            flash(f'Employee "{update_fields["name"]}" updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        else:
            try:
                employee.name = request.form.get('name', '').strip()
                employee.gender = request.form.get('gender', '').strip()
                employee.date_of_birth = datetime.strptime(request.form.get('date_of_birth'), '%Y-%m-%d').date()
                employee.position = request.form.get('position', '').strip()
                employee.address = request.form.get('address', '').strip()
                image_file = request.files.get('image_file')
                if image_file and image_file.filename:
                    temp_image_path = os.path.join(KNOWN_FACES_DIR, f"temp_upload_{employee.name}_{get_cambodia_time().strftime('%Y%m%d%H%M%S')}.jpg")
                    image_file.save(temp_image_path)
                    img = face_recognition.load_image_file(temp_image_path)
                    encodings = face_recognition.face_encodings(img)
                    os.remove(temp_image_path)
                    if encodings:
                        employee.face_encoding = pickle.dumps(encodings[0])
                        reload_known_faces()
                db.session.commit()
                flash(f'Employee "{employee.name}" updated successfully!', 'success')
                return redirect(url_for('admin_dashboard'))
            except Exception as e:
                db.session.rollback()
                flash(f'An error occurred while updating employee: {e}', 'error')
                return redirect(url_for('edit_employee_manual', employee_id=employee_id))

    return render_template('add_edit_employee.html', employee=employee, action_url=url_for('edit_employee_manual', employee_id=employee_id))


@app.route('/admin/employee/<employee_id>')
def view_employee_details(employee_id):
    if USE_FIRESTORE:
        employee = firestore_db.get_employee_by_id(employee_id)
        if not employee:
            flash('Employee not found.', 'error')
            return redirect(url_for('admin_dashboard'))
        raw_attends = firestore_db.get_attendances_for_employee(employee_id)
        # Sort in Python since we removed the order_by from the Firestore query
        raw_attends.sort(key=lambda x: x.get('check_in', ''), reverse=True)
        attendances = []
        for a in raw_attends:
            ci = None
            co = None
            try:
                if a.get('check_in'):
                    ci = datetime.fromisoformat(a.get('check_in'))
                    if ci.tzinfo is None:
                        ci = pytz.UTC.localize(ci)
                if a.get('check_out'):
                    co = datetime.fromisoformat(a.get('check_out'))
                    if co.tzinfo is None:
                        co = pytz.UTC.localize(co)
            except Exception:
                pass
            obj = SimpleNamespace()
            obj.check_in = ci
            obj.check_out = co
            obj.check_in_formatted = format_cambodia_datetime(ci) if isinstance(ci, datetime) else (ci or None)
            obj.check_out_formatted = format_cambodia_datetime(co) if isinstance(co, datetime) else (co or None)
            obj.check_in_time = format_cambodia_time(ci) if isinstance(ci, datetime) else (ci or None)
            obj.check_out_time = format_cambodia_time(co) if isinstance(co, datetime) else (co or None)
            attendances.append(obj)
        emp_obj = SimpleNamespace()
        emp_obj.id = employee.get('id', 'N/A')
        emp_obj.name = employee.get('name', 'Unknown')
        emp_obj.position = employee.get('position', 'N/A')
        emp_obj.gender = employee.get('gender', 'N/A')
        emp_obj.address = employee.get('address', 'N/A')
        dob_str = employee.get('date_of_birth')
        try:
            emp_obj.date_of_birth = datetime.strptime(dob_str, '%Y-%m-%d').date() if dob_str else None
        except (ValueError, TypeError):
            emp_obj.date_of_birth = None
        return render_template('employee_details.html', employee=emp_obj, attendances=attendances, now_utc=datetime.now(pytz.utc))
    else:
        employee = Employee.query.get_or_404(int(employee_id))
        attendances = Attendance.query.filter_by(employee_id=employee.id).order_by(Attendance.check_in.desc()).all()
        for attendance in attendances:
            attendance.check_in_formatted = format_cambodia_datetime(attendance.check_in)
            attendance.check_out_formatted = format_cambodia_datetime(attendance.check_out)
            attendance.check_in_time = format_cambodia_time(attendance.check_in)
            attendance.check_out_time = format_cambodia_time(attendance.check_out)
        return render_template('employee_details.html', employee=employee, attendances=attendances, now_utc=datetime.now(pytz.utc))


@app.route('/admin/delete_employee/<employee_id>', methods=['POST'])
def delete_employee_manual(employee_id):
    if USE_FIRESTORE:
        emp = firestore_db.get_employee_by_id(employee_id)
        if not emp:
            flash('Employee not found.', 'error')
            return redirect(url_for('admin_dashboard'))
        
        emp_name_to_delete = emp.get('name') # Get name before deleting employee
        try:
            firestore_db.delete_attendances_for_employee_name(emp_name_to_delete) # Delete old records by name
            firestore_db.delete_employee(employee_id)
            reload_known_faces()
            flash(f'Employee "{emp.get("name")}" and all associated records deleted successfully!', 'success')
        except Exception as e:
            flash(f'Error deleting employee: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))
    else:
        employee = Employee.query.get_or_404(int(employee_id))
        try:
            Attendance.query.filter_by(employee_id=employee.id).delete()
            db.session.delete(employee)
            db.session.commit()
            image_path = os.path.join(KNOWN_FACES_DIR, f"{employee.name}.jpg")
            if os.path.exists(image_path):
                os.remove(image_path)
            reload_known_faces()
            flash(f'Employee "{employee.name}" and all associated records deleted successfully!', 'success')
        except Exception as e:
            db.session.rollback()
            flash(f'Error deleting employee: {str(e)}', 'error')
        return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    with app.app_context():
        reload_known_faces()
    app.run(debug=True, host='0.0.0.0', port=5000)