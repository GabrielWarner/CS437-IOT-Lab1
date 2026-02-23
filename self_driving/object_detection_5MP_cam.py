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
"""Main script to run the object detection routine."""

import argparse
import time
import threading
import cv2
from tflite_support.task import core, processor, vision
import utils

# Optimized: Threaded VideoStream class to decouple Camera I/O from Inference.
# This prevents the AI model from "waiting" on the camera hardware.
class VideoStream:
    """Threaded camera reader to prevent blocking inference"""
    def __init__(self, src=0, width=640, height=480):
        # Use default backend (compatible with libcamerify)
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Start the thread to read frames from the video stream
        self.grabbed = False
        self.frame = None
        self.stopped = False

    def start(self):
        # Start the thread to read frames
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            try:
                grabbed, frame = self.cap.read()
                # Ensure we have valid data before updating the class variables
                if grabbed and frame is not None and frame.size > 0:
                    self.grabbed = grabbed
                    self.frame = frame
                else:
                    time.sleep(0.01)
            except cv2.error:
                time.sleep(0.1)
                continue

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()

def run(model, camera_id, width, height, num_threads, enable_edgetpu, stop_event):
    """Continuously run inference on images acquired from the camera.

    Args:
        model: Name of the TFLite object detection model.
        camera_id: The camera id to be passed to OpenCV.
        width: The width of the frame captured from the camera.
        height: The height of the frame captured from the camera.
        num_threads: The number of CPU threads to run the model.
        enable_edgetpu: True/False whether the model is a EdgeTPU model.
    """
    import sys, traceback
    try:
        print(f'[CAMERA] Starting camera thread (model={model}, cam={camera_id})...', flush=True)

        # Optimized: Replaced cap = cv2.VideoCapture with VideoStream class.
        # Start capturing video input from the camera in a separate thread.
        vs = VideoStream(camera_id, width, height).start()
        time.sleep(1.0)

        # Check if camera actually opened
        if not vs.cap.isOpened():
            print('[CAMERA] ERROR: Camera failed to open! Check camera connection.', flush=True)
            return

        # Wait for first frame
        first_frame = vs.read()
        if first_frame is None:
            print('[CAMERA] WARNING: No frame yet after 1s, waiting longer...', flush=True)
            time.sleep(2.0)
            first_frame = vs.read()
            if first_frame is None:
                print('[CAMERA] ERROR: Still no frames from camera. Giving up.', flush=True)
                vs.stop()
                return

        print(f'[CAMERA] Camera ready, frame shape: {first_frame.shape}', flush=True)

        base_options = core.BaseOptions(file_name=model, use_coral=enable_edgetpu, num_threads=num_threads)
        detection_options = processor.DetectionOptions(max_results=3, score_threshold=0.3)
        options = vision.ObjectDetectorOptions(base_options=base_options, detection_options=detection_options)
        detector = vision.ObjectDetector.create_from_options(options)

        print('[CAMERA] Object detection model loaded successfully.', flush=True)

        last_stop_sign_print_time = 0.0
        last_person_print_time = 0.0
        print_cooldown_sec = 0.5
        frame_count = 0

        try:
            while not stop_event.is_set():
                frame = vs.read()
                if frame is None:
                    continue
                frame_count += 1
                if frame_count == 1:
                    print('[CAMERA] Processing first frame...', flush=True)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                tensor = vision.TensorImage.create_from_array(rgb)
                result = detector.detect(tensor)
                utils.visualize(frame, result)

                if frame_count == 1:
                    print(f'[CAMERA] First detection done. {len(result.detections)} object(s) found.', flush=True)

                # Direct terminal output for detections (like detect.py)
                now = time.time()
                for detection in result.detections:
                    if not detection.categories:
                        continue
                    category = detection.categories[0]
                    label = (category.category_name or '').strip().lower().replace('_', ' ').replace('-', ' ')
                    score = category.score
                    if label == 'stop sign' and now - last_stop_sign_print_time >= print_cooldown_sec:
                        print(f'[CAMERA] stop sign detected (confidence={score:.4f})', flush=True)
                        last_stop_sign_print_time = now
                    if label == 'person' and now - last_person_print_time >= print_cooldown_sec:
                        print(f'[CAMERA] person detected (confidence={score:.4f})', flush=True)
                        last_person_print_time = now

                # Optimized: sleep allows the OS to switch between the camera thread
                # and the inference thread efficiently, reducing CPU spikes.
                time.sleep(0.01)
        finally:
            vs.stop()
            print('[CAMERA] Detector stopped.', flush=True)
    except Exception as e:
        print(f'[CAMERA] FATAL ERROR in camera thread: {e}', flush=True)
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='efficientdet_lite0.tflite')
    parser.add_argument('--cameraId', type=int, default=0)
    parser.add_argument('--frameWidth', type=int, default=640)
    parser.add_argument('--frameHeight', type=int, default=480)
    parser.add_argument('--numThreads', type=int, default=4)
    parser.add_argument('--enableEdgeTPU', action='store_true')
    args = parser.parse_args()

    # Optimized: Using a Event() allows for shutdown from other parts 
    # of the program, unlike the hard 'sys.exit()' in the original code.
    import threading
    stop_event = threading.Event()
    run(args.model, args.cameraId, args.frameWidth, args.frameHeight, args.numThreads, args.enableEdgeTPU, stop_event)

if __name__ == "__main__":
    main()