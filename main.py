import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def display_tutorial():
    # Create a black canvas for the tutorial
    tutorial_canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Write instructions on the canvas
    instructions = [
        "Welcome to Hand Drawing Tutorial!",
        "Press 'c' to clear the canvas.",
        "Press numbers 0-4 to change colors.",
        "Draw by pinching your thumb and index finger.Press any key to continue"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)
    line_height = 40
    y = 40
    for instruction in instructions:
        cv2.putText(tutorial_canvas, instruction, (20, y), font, font_scale, font_color, 2)
        y += line_height
    
    # Display the tutorial
    cv2.imshow('Tutorial', tutorial_canvas)
    cv2.waitKey(0)
    cv2.destroyWindow('Tutorial')

# Initialize MediaPipe Hands model
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.1)

# Initialize the webcamw
cap = cv2.VideoCapture(0)

# Call the function to display the tutorial
display_tutorial()

# Variables to track pinching state and palm state
pinch_threshold = 0.1
pinched = True
prev_x, prev_y = 0, 0
drawing_color = (0, 255, 0)  # Initial color is green

# Create a black canvas for drawing
canvas = np.zeros((480, 640, 3), dtype=np.uint8)

# Dictionary to store color names and corresponding BGR values
colors = {
    '0': (0, 255, 0),  # Green
    '1': (0, 0, 255),  # Red
    '2': (255, 0, 0),  # Blue
    '3': (0, 255, 255),  # Yellow
    '4': (255, 255, 0),  # Cyan
}

# Function to draw selected color indicator
def draw_color_indicator(image, color):
    indicator_size = 50
    indicator_pos = (image.shape[1] - indicator_size - 10, image.shape[0] - indicator_size - 10)
    cv2.rectangle(image, indicator_pos, (image.shape[1] - 10, image.shape[0] - 10), color, -1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Hands
    results = hands.process(rgb_frame)

    # Draw hand landmarks on the frame
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get the coordinates of thumb and index finger
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

            # Calculate the distance between thumb and index finger
            distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2 +
                               (thumb_tip.z - index_tip.z) ** 2)

            # Check if the distance is less than the threshold to determine pinching
            if distance < pinch_threshold:
                pinched = True
            else:
                pinched = False

            # Check if palm is open and thumb and index finger are not touching
            if not pinched and hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y:
                pinched = True

            # Draw on the canvas if pinched
            if pinched:
                curr_x, curr_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
                if prev_x != 0 and prev_y != 0:
                    cv2.line(canvas, (prev_x, prev_y), (curr_x, curr_y), drawing_color, 5)
                prev_x, prev_y = curr_x, curr_y
            else:
                prev_x, prev_y = 0, 0

    # Combine the frame with the canvas
    frame = cv2.add(frame, canvas)

    # Draw color indicator
    draw_color_indicator(frame, drawing_color)

    # Show the combined frame
    cv2.imshow('Hand Drawing', frame)

    # Check for keypresses
    key = cv2.waitKey(1) & 0xFF

    # Clear canvas when 'c' is pressed
    if key == ord('c'):
        canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    # Change drawing color using keyboard numbers
    if chr(key) in colors:
        drawing_color = colors[chr(key)]

    # Break the loop when 'q' is pressed
    if key == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
