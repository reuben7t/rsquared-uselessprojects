import cv2
import mediapipe as mp
import pygame
import os

# Init pygame mixer
pygame.mixer.init()

# Define notes
notes = ["C4", "D4", "E4", "F4", "G4", "A4", "B4", "C5"]

# Get absolute path for sounds folder
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sounds = {note: pygame.mixer.Sound(os.path.join(BASE_DIR, "sounds", f"{note}.wav")) for note in notes}

# Setup mediapipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Open webcam
cap = cv2.VideoCapture(0)

frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

key_width = frame_width // len(notes)
key_height = 150  # height of piano keys

last_note = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Create overlay for translucent keys
    overlay = frame.copy()

    for i, note in enumerate(notes):
        x1 = i * key_width
        y1 = frame_height - key_height
        x2 = x1 + key_width
        y2 = frame_height

        # Key color and transparency
        if last_note == note:
            color = (0, 255, 255)  # cyan highlight
            alpha = 0.6
        else:
            color = (255, 255, 255)  # white key
            alpha = 0.4

        # Draw filled rectangle on overlay
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness=-1)

        # Draw black outline on the main frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), thickness=2)

        # Draw note text on the main frame
        cv2.putText(frame, note, (x1 + 10, y2 - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

    # Blend the overlay with the frame
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x = int(hand_landmarks.landmark[8].x * frame_width)
            y = int(hand_landmarks.landmark[8].y * frame_height)

            cv2.circle(frame, (x, y), 10, (0, 255, 0), -1)

            note_index = x // key_width
            if note_index < len(notes):
                note = notes[note_index]
                cv2.putText(frame, note, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

                if note != last_note:
                    sounds[note].play()
                    last_note = note
            else:
                last_note = None
    else:
        last_note = None

    cv2.imshow('Virtual Piano', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
