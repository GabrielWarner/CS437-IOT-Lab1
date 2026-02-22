# Copyright 2021 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions to display the pose detection results."""

import time
import numpy as np
from tflite_support.task import processor
from safety_state import set_person_detected, set_stop_sign_detected

PRINT_INTERVAL = 0.5
last_person_time = 0
last_stop_time = 0
COOLDOWN = 2.0

LOG_LEVEL = "INFO"

def log(msg, level="INFO"):
    levels = ["DEBUG", "INFO", "WARNING"]
    if levels.index(level) >= levels.index(LOG_LEVEL):
        print(f"[{level}] {msg}")

def visualize(image: np.ndarray, detection_result: processor.DetectionResult) -> np.ndarray:

    global last_person_time, last_stop_time
    now = time.time()
    person_seen = False
    stop_sign_seen = False

    for detection in detection_result.detections:
        category = detection.categories[0]
        name = category.category_name.lower()
        score = category.score
        if score < 0.5:
            continue
        if name == "person":
            person_seen = True
        if name in ["stop sign", "stop"]:
            stop_sign_seen = True

    # Update shared flags
    set_person_detected(person_seen)
    set_stop_sign_detected(stop_sign_seen)

    # Throttled logging
    if person_seen and now - last_person_time > COOLDOWN:
        log("PERSON detected — stopping car", "INFO")
        last_person_time = now
    if stop_sign_seen and now - last_stop_time > COOLDOWN:
        log("STOP SIGN detected", "INFO")
        last_stop_time = now

    return image