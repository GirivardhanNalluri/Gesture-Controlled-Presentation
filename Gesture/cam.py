import os
import cv2
import numpy as np
import mediapipe as mp
import aspose.slides as slides
import shutil

# Specify the folder path
folder_path = r"C:\Users\DELL\gesture controlled presentation\ngv\converted_images"

# Create the folder if it doesn't exist
os.makedirs(folder_path, exist_ok=True)

def ppt_to_images(ppt_path, output_dir):
    presentation = slides.Presentation(ppt_path)
    for i, slide in enumerate(presentation.slides):
        slide_image_path = f"{output_dir}/slide_{i + 1}.png"
        slide_image = slide.get_thumbnail(1.0, 1.0)  # Adjust scale as needed
        slide_image.save(slide_image_path)

class SlideShow:
    def __init__(self, folder_path, window_width=800, window_height=600):
        self.folder_path = folder_path
        self.image_files = self._get_image_files()
        self.current_index = 0
        self.window_width = window_width
        self.window_height = window_height
        if not self.image_files:
            print("No images found in the folder.")
            return
        self.camera_running = True
        self.cap = cv2.VideoCapture(0)  # Initialize the camera
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)
        self.mp_drawing = mp.solutions.drawing_utils
        if not self.cap.isOpened():
            print("Error: Cannot access the camera.")
            self.camera_running = False
        # Initialize zoom factor and zoom center
        self.zoom_factor = 1.0
        self.zoom_center = (self.window_width // 2, self.window_height // 2)
        # Initialize drawing mode and annotations
        self.drawing_mode = False
        self.annotations = []  # List to store drawn lines
        self.current_line = []  # List to store points of the current line
        # Display the first image
        self.show_image()

    def _get_image_files(self):
        files = os.listdir(self.folder_path)
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff']
        return [f for f in files if os.path.splitext(f)[1].lower() in image_extensions]

    def show_image(self):
        control_mode = False  # Start in non-control mode
        gesture_detected = False  # Flag to prevent continuous slide change on gesture hold
        while True:
            image_path = os.path.join(self.folder_path, self.image_files[self.current_index])
            img = cv2.imread(image_path)
            if img is None:
                print(f"Error opening image: {image_path}")
                return
            # Resize the image to fit the window
            img = self.resize_image(img, self.window_width, self.window_height)
            if self.camera_running:
                ret, frame = self.cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = self.hands.process(frame_rgb)
                    fingers = None  # Initialize fingers variable
                    if results.multi_hand_landmarks:
                        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                            # Get hand name (Left or Right)
                            hand_name = handedness.classification[0].label
                            # Draw landmarks and connections
                            self.mp_drawing.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                            # Get bounding box coordinates
                            x_min, y_min, x_max, y_max = self.get_hand_bounding_box(hand_landmarks, frame.shape)
                            # Draw bounding box
                            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                            # Display hand name
                            cv2.putText(frame, hand_name, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                            fingers = self.get_finger_gesture(hand_landmarks)
                            # Check if all fingers are closed ([0, 1, 0, 0, 1]) to start control mode
                            if fingers == [1, 1, 1, 1, 1]:
                                control_mode = True  # Enable control mode when all fingers are closed
                                gesture_detected = True
                            # Exit the program if all fingers are open ([1, 1, 1, 1, 1])
                            if fingers == [0, 1, 0, 0, 1]:
                                print("Exiting...")
                                self.cleanup_images()
                                return
                            # Call handle_gestures function based on the detected hand
                            control_mode, gesture_detected = self.handle_gestures(fingers, control_mode, gesture_detected, hand_landmarks, hand_name)
                    # Resize the camera frame to fit in the top-right corner
                    cam_width, cam_height = 200, 150
                    frame_resized = cv2.resize(frame, (cam_width, cam_height))
                    # Overlay the camera feed onto the image (top-right corner)
                    img[0:cam_height, img.shape[1]-cam_width:img.shape[1]] = frame_resized
            # Apply zoom to the image
            if self.zoom_factor != 1.0:
                img = self.apply_zoom(img, self.zoom_factor)
            # Draw annotations on the image
            self.draw_annotations(img)
            # Display the image with the current index text
            text = f"Slide {self.current_index + 1}/{len(self.image_files)}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (255, 105, 180)
            font_thickness = 2
            text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
            text_x = (img.shape[1] - text_size[0]) // 2
            text_y = img.shape[0] - 20
            cv2.putText(img, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
            cv2.imshow("Slide Show", img)
            key = cv2.waitKey(1) & 0xFF
            if key == 83 or key == 2555904:  # Right arrow key or 'd' key
                if self.current_index < len(self.image_files) - 1:  # Check if not the last slide
                    self.current_index += 1
            elif key == 81 or key == 2424832:  # Left arrow key or 'a' key
                if self.current_index > 0:  # Check if not the first slide
                    self.current_index -= 1
            elif key == 27:  # ESC key to exit
                break
            # Reset the gesture detection flag if no gesture is detected
            if not results.multi_hand_landmarks or not fingers:
                gesture_detected = False
            # Exit the slideshow if the last slide is reached
            if self.current_index == len(self.image_files) - 1:
                print("Reached the last slide. Exiting...")
                break
        cv2.destroyAllWindows()
        self.cap.release()

    def handle_gestures(self, fingers, control_mode, gesture_detected, hand_landmarks, hand_name):
        """Handle gestures for both left and right hands."""
        if control_mode:
            # Perform slide changes based on gesture
            if fingers == [1, 0, 0, 0, 0] and not gesture_detected:  # [1, 0, 0, 0, 0] moves to the previous slide
                if self.current_index > 0:  # Check if not the first slide
                    self.current_index -= 1
                gesture_detected = True  # Set the flag to True to avoid continuous slide change
            elif fingers == [0, 0, 0, 0, 1] and not gesture_detected:  # [0, 0, 0, 0, 1] moves to the next slide
                if self.current_index < len(self.image_files) - 1:  # Check if not the last slide
                    self.current_index += 1
                gesture_detected = True  # Set the flag to True to avoid continuous slide change
            # Zoom functionality for right hand
            if hand_name == "Right" and fingers == [1, 1, 0, 0, 0]:  # Both index and thumb extended
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                horizontal_distance = abs(thumb_tip.x - index_tip.x)
                if horizontal_distance > 0.1:  # Increase zoom if distance is large
                    self.zoom_factor = min(2.0, self.zoom_factor + 0.05)  # Maximum zoom out factor
                elif horizontal_distance < 0.05:  # Decrease zoom if distance is small
                    self.zoom_factor = max(1.0, self.zoom_factor - 0.05)  # Minimum zoom in factor
            # Draw mode for right hand
            if hand_name == "Right" and fingers == [0, 1, 0, 0, 0]:  # Only index finger extended
                self.drawing_mode = True
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                x, y = int(index_tip.x * self.window_width), int(index_tip.y * self.window_height)
                self.current_line.append((x, y))  # Add the point to the current line
            else:
                if self.drawing_mode and hand_name == "Right":
                    if self.current_line:
                        self.annotations.append(self.current_line)
                        self.current_line = []
                    self.drawing_mode = False
            # Erase mode for right hand
            if hand_name == "Right" and fingers == [0, 1, 1, 1, 0]:  # Index, middle, and ring fingers extended
                self.annotations = []  # Clear all annotations
            if hand_name == "Left" and fingers == [1, 1, 0, 0, 0]:  # Both index and thumb extended
                thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # Update zoom center dynamically
                self.zoom_center = (int((thumb_tip.x + index_tip.x) / 2 * self.window_width),int((thumb_tip.y + index_tip.y) / 2 * self.window_height))
                horizontal_distance = abs(thumb_tip.x - index_tip.x)
                if horizontal_distance > 0.1:
                    self.zoom_factor = min(2.0, self.zoom_factor + 0.05)  # Zoom in
                elif horizontal_distance < 0.05:
                    self.zoom_factor = max(1.0, self.zoom_factor - 0.05)  # Zoom out
            if hand_name == "Left" and fingers == [0, 1, 0, 0, 0]:  # Only index finger extended
                self.drawing_mode = True
                index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
                # Convert normalized hand coordinates to pixel coordinates
                x, y = int(index_tip.x * self.window_width), int(index_tip.y * self.window_height)
                if 0 <= x < self.window_width and 0 <= y < self.window_height:  # Ensure within bounds
                    self.current_line.append((x, y))  # Add to current line
                else:
                    if self.drawing_mode and hand_name == "Left":
                        if self.current_line:
                            self.annotations.append(self.current_line)
                            self.current_line = []
                        self.drawing_mode = False
            # Erase mode for left hand
            if hand_name == "Left" and fingers == [0, 1, 1, 1, 0]:  # Index, middle, and ring fingers extended
                self.annotations = []  # Clear all annotations
        return control_mode, gesture_detected

    def draw_annotations(self, img):
        """Draw all annotations on the image."""
        for line in self.annotations:
            for i in range(1, len(line)):
                cv2.line(img, line[i - 1], line[i], (0, 0, 255), 10)  # Draw lines in red

    def apply_zoom(self, img, zoom_factor):
        """Apply zoom to the image."""
        h, w = img.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        x1 = max(0, self.zoom_center[0] - new_w // 2)
        y1 = max(0, self.zoom_center[1] - new_h // 2)
        x2 = min(w, x1 + new_w)
        y2 = min(h, y1 + new_h)
        zoomed_img = img[y1:y2, x1:x2]
        return cv2.resize(zoomed_img, (w, h))

    def get_hand_bounding_box(self, hand_landmarks, frame_shape):
        """Get the bounding box coordinates for the hand."""
        x_coords = [int(lm.x * frame_shape[1]) for lm in hand_landmarks.landmark]
        y_coords = [int(lm.y * frame_shape[0]) for lm in hand_landmarks.landmark]
        return min(x_coords), min(y_coords), max(x_coords), max(y_coords)

    def get_finger_gesture(self, hand_landmarks):
        fingers = [0, 0, 0, 0, 0]
        landmarks = hand_landmarks.landmark
        # Check the thumb (0th finger)
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        thumb_mcp = landmarks[self.mp_hands.HandLandmark.THUMB_MCP]

        # Calculate the horizontal distance between the thumb tip and the thumb MCP
        if thumb_tip.x < thumb_mcp.x:
            fingers[0] = 1  # Thumb is extended
        # Check the index finger (1st finger)
        if landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP].y:
            fingers[1] = 1
        # Check the middle finger (2nd finger)
        if landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y:
            fingers[2] = 1
        # Check the ring finger (3rd finger)
        if landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP].y < landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP].y:
            fingers[3] = 1
        # Check the pinky (4th finger)
        if landmarks[self.mp_hands.HandLandmark.PINKY_TIP].y < landmarks[self.mp_hands.HandLandmark.PINKY_PIP].y:
            fingers[4] = 1
        return fingers

    def cleanup_images(self):
        """Remove all converted images from the directory."""
        for image_file in self.image_files:
            file_path = os.path.join(self.folder_path, image_file)
            try:
                os.remove(file_path)
            except OSError as e:
                print(f"Error deleting file {file_path}: {e}")

    def resize_image(self, img, target_width, target_height):
        original_height, original_width = img.shape[:2]
        aspect_ratio = original_width / original_height
        if aspect_ratio > 1:
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:
            new_height = target_height
            new_width = int(target_height * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, new_height))
        return resized_img

if __name__ == "__main__":
    ppt_path = r"C:\Users\DELL\gesture controlled presentation\ngv\RS&GIS UNIT-I.ppt"
    output_dir = r"C:\Users\DELL\gesture controlled presentation\ngv\converted_images"
    ppt_to_images(ppt_path, output_dir)  # Convert PPT slides to images first
    folder_path = output_dir  # Use the folder where PPT images are saved
    image_viewer = SlideShow(folder_path)
    shutil.rmtree(folder_path)  # Clean up converted images after viewing