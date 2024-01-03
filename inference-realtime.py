import cv2
import numpy as np
import tensorflow as tf
from button_detection import ButtonDetector
from character_recognition import CharacterRecognizer


def button_candidates(boxes, scores, image):
    img_height = image.shape[0]
    img_width = image.shape[1]

    button_scores = []
    button_patches = []
    button_positions = []

    for box, score in zip(boxes, scores):
        if score < 0.5: continue

        y_min = int(box[0] * img_height)
        x_min = int(box[1] * img_width)
        y_max = int(box[2] * img_height)
        x_max = int(box[3] * img_width)

        button_patch = image[y_min: y_max, x_min: x_max]
        button_patch = cv2.resize(button_patch, (180, 180))

        button_scores.append(score)
        button_patches.append(button_patch)
        button_positions.append([x_min, y_min, x_max, y_max])
    return button_patches, button_positions, button_scores


# Initialize the detector and recognizer
detector = ButtonDetector()
recognizer = CharacterRecognizer()

# Start capturing video from the first camera device
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the captured frame to the format expected by your model
    img_np = np.asarray(frame)

    # Detection and Recognition
    boxes, scores, _ = detector.predict(img_np, True)
    button_patches, button_positions, _ = button_candidates(boxes, scores, img_np)

    for button_img, button_pos in zip(button_patches, button_positions):
        button_text, button_score, button_draw = recognizer.predict(button_img, draw=True)
        x_min, y_min, x_max, y_max = button_pos
        button_rec = cv2.resize(button_draw, (x_max-x_min, y_max-y_min))
        img_np[y_min+6:y_max-6, x_min+6:x_max-6] = button_rec[6:-6, 6:-6]

    # Display the resulting frame
    cv2.imshow('Real-Time Detection', img_np)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
detector.clear_session()
recognizer.clear_session()
