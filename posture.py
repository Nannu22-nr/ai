import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands and Face modules
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Initialize the Hand and Face detectors
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)

# Open the webcam
cap = cv2.VideoCapture(0)

# Create a virtual canvas
canvas = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Get frame dimensions
    h, w, _ = frame.shape

    # Initialize the canvas on the first frame
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert the frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hands
    hand_results = hands.process(rgb_frame)

    # Detect face
    face_results = face_detection.process(rgb_frame)

    # Draw hand landmarks
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            # Draw landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the index finger tip position (landmark 8)
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            x, y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)

            # Draw a circle at the index finger tip on the frame
            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            # Draw on the canvas
            cv2.circle(canvas, (x, y), 5, (255, 255, 255), -1)

            # Map finger position to a letter (example: A, B, C)
            if x < w // 3:
                letter = 'A'
            elif x < 2 * w // 3:
                letter = 'B'
            else:
                letter = 'C'

            # Print the letter in the terminal
            print(letter, end='', flush=True)

    # Draw face detection
    if face_results.detections:
        for detection in face_results.detections:
            mp_drawing.draw_detection(frame, detection)

    # Show the frame and canvas
    cv2.imshow('Frame', frame)
    cv2.imshow('Canvas', canvas)

    # Exit on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()