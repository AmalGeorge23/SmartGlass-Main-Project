reading assistance import cv2
import pytesseract
from picamera2 import Picamera2
import time
import pyttsx3

# Initialize PiCamera2
picam2 = Picamera2()

# Create a still configuration with better exposure control
config = picam2.create_still_configuration(
    main={"size": (800, 600), "format": "RGB888"},
    controls={"ExposureTime": 50000, "AnalogueGain": 1.0}  # Adjust exposure time & gain
)
picam2.configure(config)
picam2.start()

# Initialize Text-to-Speech Engine
tts_engine = pyttsx3.init()
tts_engine.setProperty("rate",125)

while True:
    # Capture image
    frame = picam2.capture_array()

    # Adjust contrast & brightness (if needed)
    alpha = 1.2  # Contrast control (1.0-3.0)
    beta = 10    # Brightness control (0-100)
    #frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

    # Display the frame
    cv2.imshow("Frame", frame)

    # OCR Processing
    text = pytesseract.image_to_string(frame)
    print(text)

    # Wait for key press
    key = cv2.waitKey(1) & 0xFF
    if key == ord("s"):  # Save frame on 's' key press
        cv2.imwrite("captured_frame.jpg", frame)
        print("Image saved. Audio output will play in 5 seconds...")
        time.sleep(5)  # Delay for 5 seconds
        tts_engine.say(text)  # Speak the extracted text
        tts_engine.runAndWait()
    elif key == ord("q"):  # Quit on 'q' key press
        break

# Cleanup
cv2.destroyAllWindows()
picam2.stop()