import threading

_person_detected = False
_stop_sign_detected = False
_stop_sign_ever_detected = False
_person_ever_detected = False
_arrived = False
_running = True

_lock = threading.Lock()

# -------- PERSON --------
def set_person_detected(value: bool):
    global _person_detected, _person_ever_detected
    with _lock:
        _person_detected = value
        if value:
            _person_ever_detected = True

def is_person_detected() -> bool:
    with _lock:
        return _person_detected

def was_person_ever_detected() -> bool:
    with _lock:
        return _person_ever_detected

# -------- STOP SIGN --------
def set_stop_sign_detected(value: bool):
    global _stop_sign_detected, _stop_sign_ever_detected
    with _lock:
        _stop_sign_detected = value
        if value:
            _stop_sign_ever_detected = True

def is_stop_sign_detected() -> bool:
    with _lock:
        return _stop_sign_detected

def was_stop_sign_ever_detected() -> bool:
    with _lock:
        return _stop_sign_ever_detected

# -------- ARRIVAL --------
def set_arrived(value: bool):
    global _arrived
    with _lock:
        _arrived = value

def is_arrived() -> bool:
    with _lock:
        return _arrived

# -------- GLOBAL RUN FLAG --------
def stop_running():
    global _running
    with _lock:
        _running = False

def is_running() -> bool:
    with _lock:
        return _running