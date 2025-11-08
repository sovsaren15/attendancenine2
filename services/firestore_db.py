"""
Simple Firestore wrapper for Employees and Attendances.
Stores: employees collection and attendances collection.
Employee document fields:
 - name (string)
 - gender
 - date_of_birth (ISO date string)
 - position
 - address
 - face_encoding_b64 (base64 string of pickled numpy array)

Attendance document fields:
 - employee_name
 - check_in (ISO datetime str)
 - check_out (ISO datetime str or null)
 - check_in_status

Note: This module assumes firebase-admin has already been initialized via init_firebase()
in `services/firebase_vision` or elsewhere.
"""
from typing import List, Dict, Optional
import base64
import pickle
from datetime import datetime

import firebase_admin
from firebase_admin import firestore


def _get_client():
    if not firebase_admin._apps:
        raise RuntimeError("Firebase app is not initialized. Call init_firebase() first.")
    return firestore.client()


def create_employee(name: str, gender: str, date_of_birth: str, position: str, address: str, face_encoding_b64: str) -> str:
    """Create an employee doc. date_of_birth expected as 'YYYY-MM-DD' string. Returns document id."""
    client = _get_client()
    coll = client.collection('employees')
    doc = {
        'name': name,
        'gender': gender,
        'date_of_birth': date_of_birth,
        'position': position,
        'address': address,
        'face_encoding_b64': face_encoding_b64,
        'created_at': datetime.utcnow().isoformat()
    }
    res = coll.add(doc)
    return res[1].id


def get_all_employees() -> List[Dict]:
    """Return list of employee dicts with keys including 'name' and 'face_encoding_b64'"""
    client = _get_client()
    coll = client.collection('employees')
    docs = coll.stream()
    out = []
    for d in docs:
        data = d.to_dict()
        data['id'] = d.id
        out.append(data)
    return out


def find_employee_by_name(name: str) -> Optional[Dict]:
    client = _get_client()
    coll = client.collection('employees')
    q = coll.where('name', '==', name).limit(1).stream()
    for d in q:
        data = d.to_dict(); data['id'] = d.id; return data
    return None


def add_attendance(employee_id: str, check_in_iso: str, check_out_iso: Optional[str], check_in_status: Optional[str]) -> str:
    client = _get_client()
    coll = client.collection('attendances')
    doc = {
        'employee_id': employee_id,
        'check_in': check_in_iso,
        'check_out': check_out_iso,
        'check_in_status': check_in_status,
        'created_at': datetime.utcnow().isoformat()
    }
    res = coll.add(doc)
    return res[1].id


def get_attendances_for_employee_on_date(employee_id: str, date_iso: str) -> List[Dict]:
    """Return attendances (list) filtered by employee ID and date_iso YYYY-MM-DD (check_in date)."""
    client = _get_client()
    coll = client.collection('attendances')
    # Query by employee_id for new records
    docs = coll.where('employee_id', '==', employee_id).stream()
    out = []
    for d in docs:
        data = d.to_dict()
        # filter by date
        if 'check_in' in data and data['check_in'].startswith(date_iso):
            data['id'] = d.id
            out.append(data)

    # Also query by employee_name for backward compatibility with old records
    employee_doc = get_employee_by_id(employee_id)
    if employee_doc and 'name' in employee_doc:
        employee_name = employee_doc['name']
        docs_old = coll.where('employee_name', '==', employee_name).stream()
        for d in docs_old:
            data = d.to_dict()
            if 'check_in' in data and data['check_in'].startswith(date_iso):
                data['id'] = d.id
                if d.id not in [x['id'] for x in out]: # Avoid duplicates
                    out.append(data)
    return out


def get_recent_attendance_for_employee(employee_id: str) -> Optional[Dict]:
    """Gets the most recent attendance record for an employee by their ID."""
    client = _get_client()
    coll = client.collection('attendances')
    docs = coll.where('employee_id', '==', employee_id).order_by('check_in', direction=firestore.Query.DESCENDING).limit(1).stream()
    for d in docs:
        data = d.to_dict(); data['id'] = d.id; return data
    return None


def update_attendance(attendance_id: str, fields: Dict) -> None:
    """Update attendance document fields by id."""
    client = _get_client()
    doc_ref = client.collection('attendances').document(attendance_id)
    doc_ref.update(fields)


def get_all_attendances(limit: int = 1000) -> List[Dict]:
    """Return all attendances ordered by check_in descending. Limit controls how many documents to fetch."""
    client = _get_client()
    coll = client.collection('attendances')
    docs = coll.order_by('check_in', direction=firestore.Query.DESCENDING).limit(limit).stream()
    out = []
    for d in docs:
        data = d.to_dict()
        data['id'] = d.id
        out.append(data)
    return out


def get_attendances_since(iso_date_str: str) -> List[Dict]:
    """Return all attendances since a given ISO date string, ordered by check_in descending."""
    client = _get_client()
    coll = client.collection('attendances')
    docs = coll.where('check_in', '>=', iso_date_str).order_by('check_in', direction=firestore.Query.DESCENDING).stream()
    out = []
    for d in docs:
        data = d.to_dict()
        data['id'] = d.id
        out.append(data)
    return out


def get_employee_by_id(employee_id: str) -> Optional[Dict]:
    """Retrieve a single employee document by document id."""
    client = _get_client()
    doc = client.collection('employees').document(employee_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict()
    data['id'] = doc.id
    return data


def update_employee(employee_id: str, fields: Dict) -> None:
    """Update employee document fields by id."""
    client = _get_client()
    doc_ref = client.collection('employees').document(employee_id)
    doc_ref.update(fields)


def delete_employee(employee_id: str) -> None:
    """Delete an employee document and any attendances linked by employee_name."""
    client = _get_client()
    doc_ref = client.collection('employees').document(employee_id)
    doc = doc_ref.get()
    if doc.exists:
        data = doc.to_dict()
        emp_name = data.get('name')
        # Delete attendances linked to this employee name
        if emp_name:
            att_coll = client.collection('attendances')
            docs = att_coll.where('employee_name', '==', emp_name).stream()
            for d in docs:
                try:
                    d.reference.delete()
                except Exception:
                    pass
        # Delete employee doc
        doc_ref.delete()


def delete_attendances_for_employee_name(employee_name: str) -> None:
    """Deletes all attendance records associated with a given employee name.
    This is useful for cleaning up old records that were linked by name.
    """
    client = _get_client()
    att_coll = client.collection('attendances')
    docs = att_coll.where('employee_name', '==', employee_name).stream()
    for d in docs:
        try:
            d.reference.delete()
        except Exception:
            pass


def get_attendances_for_employee(employee_id: str) -> List[Dict]:
    """Return all attendances for an employee by their ID."""
    client = _get_client()
    coll = client.collection('attendances')
    docs = coll.where('employee_id', '==', employee_id).stream()
    out = []
    for d in docs:
        data = d.to_dict()
        data['id'] = d.id
        out.append(data)
    return out
