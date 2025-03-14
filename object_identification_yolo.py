import cv2
from picamera2 import Picamera2
from ultralytics import YOLO
import pyttsx3
import time

# Set up the camera with Picam
picam2 = Picamera2()
picam2.preview_configuration.main.size = (1080, 720)
picam2.preview_configuration.main.format = "RGB888"
picam2.preview_configuration.align()
picam2.configure("preview")
picam2.start()

# Load YOLOv8
model = YOLO("yolov8n")

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate", 125)  # Adjust reading speed

while True:
    # Capture a frame from the camera
    frame = picam2.capture_array()
    
    # Run YOLO model on the captured frame
    results = model(frame)

    # Get detected object names
    detected_objects = [model.names[int(box.cls)] for box in results[0].boxes]

    # Annotate the frame with detection results
    annotated_frame = results[0].plot()

    # Get inference time
    inference_time = results[0].speed['inference']
    fps = 1000 / inference_time  # Convert to milliseconds
    text = f'FPS: {fps:.1f}'

    # Define font and position
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, 1, 2)[0]
    text_x = annotated_frame.shape[1] - text_size[0] - 10  # 10 pixels from the right
    text_y = text_size[1] + 10  # 10 pixels from the top

    # Draw the FPS text on the annotated frame
    cv2.putText(annotated_frame, text, (text_x, text_y), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Display the frame
    cv2.imshow("Camera", annotated_frame)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("o"):  # Start collecting objects over 3 seconds
        print("Capturing objects for 3 seconds...")
        start_time = time.time()
        collected_objects = set()  # Store unique detected objects

        while time.time() - start_time < 3:  # Collect objects for 3 seconds
            frame = picam2.capture_array()
            results = model(frame)
            for box in results[0].boxes:
                collected_objects.add(model.names[int(box.cls)])

        # Generate speech output
        if collected_objects:
            speech_text = f"Detected objects: {', '.join(collected_objects)}"
        else:
            speech_text = "No objects detected"

        print(speech_text)
        tts_engine.say(speech_text)
        tts_engine.runAndWait()

    elif key == ord("q"):  # Quit on "q"
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()