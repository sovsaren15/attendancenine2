import cv2
import face_recognition
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session

from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate 
from datetime import datetime
import math
import os
import base64
from io import BytesIO
from PIL import Image
import sqlalchemy as sa
import pickle 
import pytz  

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+psycopg://postgres:123@localhost:5432/smart_attendance'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.secret_key = 'your_super_secret_key_here' 
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

# --- Geolocation Settings ---
app.config['COMPANY_LATITUDE'] = 13.37488193943832
app.config['COMPANY_LONGITUDE'] = 103.842437801041
app.config['MAX_DISTANCE_METERS'] = 2000    # Allow check-in/out within 2000 meters
# --- Debugging Settings ---
# If True, will use company coordinates as a fallback when browser location is not available.
app.config['DEBUG_USE_COMPANY_LOCATION'] = True 
 
db = SQLAlchemy(app)
migrate = Migrate(app, db) 


CAMBODIA_TZ = pytz.timezone('Asia/Phnom_Penh')

def get_cambodia_time():
    """Get current time in Cambodia timezone"""
    return datetime.now(CAMBODIA_TZ)

def get_cambodia_date():
    """Get current date in Cambodia timezone"""
    return get_cambodia_time().date()

# Database Models
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
    check_in_status = db.Column(db.String(10), nullable=True) # e.g., 'Early', 'Good', 'Late'

    employee = db.relationship('Employee', backref=db.backref('attendances', lazy=True))


with app.app_context():
    try:

        db.create_all()
        print("Database tables ensured to be created.")
    except Exception as e:
        print(f"Error ensuring database tables are created: {e}")


KNOWN_FACES_DIR = "known_faces"
if not os.path.exists(KNOWN_FACES_DIR):
    os.makedirs(KNOWN_FACES_DIR)


def load_known_faces():
    employees = Employee.query.all()
    known_face_encodings = [np.array(pickle.loads(employee.face_encoding)) for employee in employees]
    known_face_names = [employee.name for employee in employees]
    return known_face_encodings, known_face_names


def process_image_for_encoding(image_data_b64):
    try:
        image_data = base64.b64decode(image_data_b64)
        image = Image.open(BytesIO(image_data))
        image_np = np.array(image)
        
        if image_np.shape[2] == 4: # RGBA
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
    
    # Convert to Cambodia timezone
    cambodia_dt = dt.astimezone(CAMBODIA_TZ)
    return cambodia_dt.strftime("%Y-%m-%d %H:%M:%S %Z")

def format_cambodia_time(dt):
    """Format time only for display in Cambodia timezone"""
    if dt is None:
        return None
    
    if dt.tzinfo is None:
        dt = pytz.UTC.localize(dt)
    
    # Convert to Cambodia timezone
    cambodia_dt = dt.astimezone(CAMBODIA_TZ)
    return cambodia_dt.strftime("%H:%M:%S")

# --- Main Application Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name'].strip()
        gender = request.form['gender'].strip()
        date_of_birth = request.form['date_of_birth']
        position = request.form['position'].strip()
        address = request.form['address'].strip()
        image_data_b64 = request.form['image'].split(',')[1] # Get base64 data from form

        # Basic validation
        if not all([name, gender, date_of_birth, position, address, image_data_b64]):
            return jsonify({'error': 'All fields and an image are required.'}), 400
        
        # Check if employee name already exists
        if Employee.query.filter_by(name=name).first():
            return jsonify({'error': 'An employee with this name already exists.'}), 400

        try:
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            
            # Get face encoding from the captured image
            encoding = process_image_for_encoding(image_data_b64)
            if encoding is None:
                return jsonify({'error': 'No face detected in the image. Please ensure your face is clearly visible.'}), 400
            
            # Serialize the numpy array encoding into bytes for storage in PostgreSQL PickleType
            serialized_encoding = pickle.dumps(encoding)

            employee = Employee(
                name=name,
                gender=gender,
                date_of_birth=dob,
                position=position,
                address=address,
                face_encoding=serialized_encoding
            )
            db.session.add(employee)
            db.session.commit()
            return jsonify({'message': f'Employee {name} registered successfully!'})
        except ValueError:
            db.session.rollback()
            return jsonify({'error': 'Invalid date format. Please use YYYY-MM-DD.'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'An unexpected error occurred during registration: {str(e)}'}), 500
    return render_template('register.html')

def calculate_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the distance between two GPS coordinates in meters using the Haversine formula.
    """
    R = 6371000  # Radius of Earth in meters
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

@app.route('/attendance', methods=['GET', 'POST'])
def attendance():
    if request.method == 'POST':
        action = request.form.get('action') # 'check_in' or 'check_out'
        latitude = request.form.get('latitude', type=float)
        longitude = request.form.get('longitude', type=float)

        if latitude is None or longitude is None:
            # If debugging, use company location as a fallback
            if app.debug and app.config.get('DEBUG_USE_COMPANY_LOCATION'):
                latitude = app.config['COMPANY_LATITUDE']
                longitude = app.config['COMPANY_LONGITUDE']
                print("DEBUG: Using company location as fallback for attendance check.") # For server log
            else:
                return jsonify({'error': 'Location data is missing. Please enable location services in your browser.'}), 400

        # Validate user's location
        distance = calculate_distance(latitude, longitude, app.config['COMPANY_LATITUDE'], app.config['COMPANY_LONGITUDE'])
        if distance > app.config['MAX_DISTANCE_METERS']:
            return jsonify({'error': f'You are too far from the company location. Distance: {distance:.0f} meters.'}), 403

        try:
            image_data_b64 = request.form['image'].split(',')[1] # Get base64 data from form
            image_data = base64.b64decode(image_data_b64)
            image = Image.open(BytesIO(image_data))
            image_np = np.array(image)
            image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR) # Convert to BGR for face_recognition

            face_locations = face_recognition.face_locations(image_cv)
            face_encodings = face_recognition.face_encodings(image_cv, face_locations)

            if not face_encodings:
                return jsonify({'error': 'No face detected in the image. Please ensure your face is clearly visible.'}), 400

            known_face_encodings, known_face_names = load_known_faces()
            
            matched_employee = None
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                if True in matches:
                    matched_idx = matches.index(True)
                    matched_name = known_face_names[matched_idx]
                    matched_employee = Employee.query.filter_by(name=matched_name).first()
                    break # Found a match, stop looking
            
            if not matched_employee:
                return jsonify({'error': 'No matching employee found. Please ensure you are registered.'}), 400

            # Use Cambodia date for comparison
            today = get_cambodia_date()
            cambodia_now = get_cambodia_time()
            
            if action == 'check_in':
                # Check for an *open* check-in for today (i.e., not checked out yet)
                # Need to handle timezone-aware comparison properly
                existing_open_check_in = Attendance.query.filter(
                    Attendance.employee_id == matched_employee.id,
                    sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today,
                    Attendance.check_out == None
                ).first()

                if existing_open_check_in:
                    return jsonify({'error': f'{matched_employee.name} is already checked in for today.'}), 400
                
                # If there's a record for today but it's already checked out, or no record, create new.
                check_in_status = "Good" # Default status
                check_in_time_only = cambodia_now.time()
                
                # Define your time boundaries
                early_time = datetime.strptime("08:00", "%H:%M").time()
                on_time_limit = datetime.strptime("08:15", "%H:%M").time()

                if check_in_time_only < early_time:
                    check_in_status = "Early"
                elif check_in_time_only > on_time_limit:
                    check_in_status = "Late"

                attendance = Attendance(
                    employee_id=matched_employee.id,
                    check_in=cambodia_now,
                    check_in_status=check_in_status
                )
                db.session.add(attendance)
                db.session.commit()
                
                check_in_time = format_cambodia_time(attendance.check_in)
                return jsonify({'message': f'Check-in recorded for {matched_employee.name} at {check_in_time} ICT. Status: {check_in_status}'})

            elif action == 'check_out':
                # Find the most recent open check-in for today to mark as checked out
                attendance_to_checkout = Attendance.query.filter(
                    Attendance.employee_id == matched_employee.id,
                    sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today,
                    Attendance.check_out == None
                ).order_by(Attendance.check_in.desc()).first()

                if not attendance_to_checkout:
                    return jsonify({'error': f'No active check-in found for {matched_employee.name} today.'}), 400
                
                attendance_to_checkout.check_out = cambodia_now
                db.session.commit()
                
                # Format time for display in Cambodia timezone
                check_out_time = format_cambodia_time(attendance_to_checkout.check_out)
                return jsonify({'message': f'Check-out recorded for {matched_employee.name} at {check_out_time} ICT'})
            else:
                return jsonify({'error': 'Invalid attendance action specified.'}), 400
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': f'An unexpected error occurred: {str(e)}'}), 500
    return render_template('attendance.html')

@app.route('/records')
def records():
    # Order by check_in descending to show most recent first
    now_utc = datetime.now(pytz.utc) # Get current time in UTC for comparison
    attendances = Attendance.query.order_by(Attendance.check_in.desc()).all()
    
    # Format datetime for display in Cambodia timezone
    for attendance in attendances:
        attendance.check_in_formatted = format_cambodia_datetime(attendance.check_in)
        attendance.check_out_formatted = format_cambodia_datetime(attendance.check_out)
        # Also format just the time for easier reading
        attendance.check_in_time = format_cambodia_time(attendance.check_in)
        attendance.check_out_time = format_cambodia_time(attendance.check_out)
    
    return render_template('records.html', attendances=attendances, now_utc=now_utc)

# --- Admin Routes ---

@app.route('/admin')
def admin_dashboard():
    today = get_cambodia_date()
    now = get_cambodia_time()

    # --- Dashboard Stats ---
    employees = Employee.query.order_by(Employee.name).all() # Order employees alphabetically
    total_employees = len(employees)
    
    # Count employees who have checked in today (distinctly)
    checked_in_today_count = db.session.query(Attendance.employee_id).filter(
        sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today
    ).distinct().count()

    # Count employees who were late today
    late_today_count = Attendance.query.filter(
        sa.func.date(Attendance.check_in.op('AT TIME ZONE')('Asia/Phnom_Penh')) == today,
        Attendance.check_in_status == 'Late'
    ).count()

    # --- Monthly Stats ---
    first_day_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

    # Top 3 Late Employees
    top_late_employees = db.session.query(
        Employee.name,
        sa.func.count(Attendance.id).label('late_count')
    ).join(Employee).filter(
        Attendance.check_in_status == 'Late',
        Attendance.check_in >= first_day_of_month
    ).group_by(Employee.name).order_by(sa.desc('late_count')).limit(3).all()

    # Top 3 Attendance Employees
    top_attendance_employees = db.session.query(
        Employee.name,
        sa.func.count(Attendance.id).label('attendance_count')
    ).join(Employee).filter(
        Attendance.check_in >= first_day_of_month
    ).group_by(Employee.name).order_by(sa.desc('attendance_count')).limit(3).all()

    # Top 3 Early Employees
    top_early_employees = db.session.query(
        Employee.name,
        sa.func.count(Attendance.id).label('early_count')
    ).join(Employee).filter(
        Attendance.check_in_status == 'Early',
        Attendance.check_in >= first_day_of_month
    ).group_by(Employee.name).order_by(sa.desc('early_count')).limit(3).all()

    return render_template('admin.html', 
                           employees=employees, 
                           total_employees=total_employees, 
                           checked_in_today_count=checked_in_today_count, 
                           late_today_count=late_today_count,
                           top_late_employees=top_late_employees,
                           top_attendance_employees=top_attendance_employees,
                           top_early_employees=top_early_employees)

@app.route('/admin/add_employee', methods=['GET', 'POST'])
def add_employee_manual():
    if request.method == 'POST':
        name = request.form['name'].strip()
        gender = request.form['gender'].strip()
        date_of_birth = request.form['date_of_birth']
        position = request.form['position'].strip()
        address = request.form['address'].strip()
        image_file = request.files.get('image_file') # Get the uploaded file

        # Server-side validation for required fields
        if not all([name, gender, date_of_birth, position, address]):
            flash('All text fields are required!', 'error')
            return redirect(url_for('add_employee_manual'))
        
        # Check if employee name already exists to prevent duplicates
        if Employee.query.filter_by(name=name).first():
            flash(f'Employee with name "{name}" already exists.', 'error')
            return redirect(url_for('add_employee_manual'))

        try:
            dob = datetime.strptime(date_of_birth, '%Y-%m-%d').date()
            face_encoding = None

            # Process uploaded image for face encoding
            if image_file and image_file.filename != '':
                # It's safer to save the original image if you need it for re-training later
                # For now, we'll just process it in memory or a temp file.
                # If saving, use a secure filename (e.g., UUID or employee ID)
                temp_image_path = os.path.join(KNOWN_FACES_DIR, f"temp_upload_{name}_{get_cambodia_time().strftime('%Y%m%d%H%M%S')}.jpg")
                image_file.save(temp_image_path) # Save to a temporary path

                img = face_recognition.load_image_file(temp_image_path)
                encodings = face_recognition.face_encodings(img)
                os.remove(temp_image_path) # Clean up the temporary file

                if not encodings:
                    flash('No face detected in the uploaded image. Please ensure the face is clear and try again.', 'error')
                    return redirect(url_for('add_employee_manual'))
                face_encoding = pickle.dumps(encodings[0]) # Serialize the numpy array
            else:
                flash('An image file is required for facial recognition data.', 'error')
                return redirect(url_for('add_employee_manual'))

            employee = Employee(
                name=name,
                gender=gender,
                date_of_birth=dob,
                position=position,
                address=address,
                face_encoding=face_encoding
            )
            db.session.add(employee)
            db.session.commit()
            flash(f'Employee "{name}" added successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        except ValueError:
            db.session.rollback()
            flash('Invalid date format. Please use YYYY-MM-DD.', 'error')
            return redirect(url_for('add_employee_manual'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while adding employee: {str(e)}', 'error')
            return redirect(url_for('add_employee_manual'))
    
    # For GET request, render the form for adding
    return render_template('add_edit_employee.html', employee=None, action_url=url_for('add_employee_manual'))

@app.route('/admin/edit_employee/<int:employee_id>', methods=['GET', 'POST'])
def edit_employee_manual(employee_id):
    employee = Employee.query.get_or_404(employee_id) # Get employee or return 404
    
    if request.method == 'POST':
        old_name = employee.name # Store old name for potential file rename

        # Update employee object with new form data
        employee.name = request.form['name'].strip()
        employee.gender = request.form['gender'].strip()
        try:
            employee.date_of_birth = datetime.strptime(request.form['date_of_birth'], '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid date format. Please use YYYY-MM-DD.', 'error')
            return redirect(url_for('edit_employee_manual', employee_id=employee_id))
        employee.position = request.form['position'].strip()
        employee.address = request.form['address'].strip()
        image_file = request.files.get('image_file') # Check if a new image was uploaded

        # Prevent changing name to an existing one (excluding itself)
        if Employee.query.filter(Employee.name == employee.name, Employee.id != employee_id).first():
            flash(f'Employee with name "{employee.name}" already exists. Choose a different name.', 'error')
            return redirect(url_for('edit_employee_manual', employee_id=employee_id))

        try:
            if image_file and image_file.filename != '':
                temp_image_path = os.path.join(KNOWN_FACES_DIR, f"temp_upload_{employee.name}_{get_cambodia_time().strftime('%Y%m%d%H%M%S')}.jpg")
                image_file.save(temp_image_path)
                
                img = face_recognition.load_image_file(temp_image_path)
                encodings = face_recognition.face_encodings(img)
                os.remove(temp_image_path) # Clean up temp file

                if not encodings:
                    flash('No face detected in the new uploaded image. Keeping previous face encoding.', 'warning')
                else:
                    employee.face_encoding = pickle.dumps(encodings[0]) # Update with new encoding
            

            if old_name != employee.name:
                old_image_path = os.path.join(KNOWN_FACES_DIR, f"{old_name}.jpg")
                new_image_path = os.path.join(KNOWN_FACES_DIR, f"{employee.name}.jpg")
                if os.path.exists(old_image_path):
                    try:
                        os.rename(old_image_path, new_image_path)
                    except OSError as e:
                        print(f"Error renaming image file from {old_image_path} to {new_image_path}: {e}")
                        flash(f'Could not rename old image file for {old_name}. Manual intervention may be needed.', 'warning')

            db.session.commit()
            flash(f'Employee "{employee.name}" updated successfully!', 'success')
            return redirect(url_for('admin_dashboard'))
        except Exception as e:
            db.session.rollback()
            flash(f'An error occurred while updating employee: {str(e)}', 'error')
            return redirect(url_for('edit_employee_manual', employee_id=employee_id))
    
    # For GET request, display the edit form with current employee data
    return render_template('add_edit_employee.html', employee=employee, action_url=url_for('edit_employee_manual', employee_id=employee.id))

@app.route('/admin/employee/<int:employee_id>')
def view_employee_details(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    
    # Fetch all attendance records for this employee, most recent first
    attendances = Attendance.query.filter_by(employee_id=employee.id).order_by(Attendance.check_in.desc()).all()
    
    # Format dates and times for display
    for attendance in attendances:
        attendance.check_in_formatted = format_cambodia_datetime(attendance.check_in)
        attendance.check_out_formatted = format_cambodia_datetime(attendance.check_out)
        attendance.check_in_time = format_cambodia_time(attendance.check_in)
        attendance.check_out_time = format_cambodia_time(attendance.check_out)

    now_utc = datetime.now(pytz.utc) # For status calculation
    return render_template('employee_details.html', employee=employee, attendances=attendances, now_utc=now_utc)

@app.route('/admin/delete_employee/<int:employee_id>', methods=['POST'])
def delete_employee_manual(employee_id):
    employee = Employee.query.get_or_404(employee_id)
    try:
        # Delete associated attendance records first due to foreign key constraint
        Attendance.query.filter_by(employee_id=employee.id).delete()
        db.session.delete(employee) # Delete the employee
        db.session.commit()

        # Delete the associated image file from the file system
        image_path = os.path.join(KNOWN_FACES_DIR, f"{employee.name}.jpg")
        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Deleted image file: {image_path}") # For debugging

        flash(f'Employee "{employee.name}" and all associated records deleted successfully!', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Error deleting employee: {str(e)}', 'error')
        print(f"Error during deletion: {e}") # For debugging
    return redirect(url_for('admin_dashboard'))


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)