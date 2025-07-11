from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os
import mediapipe as mp
import math
import win32com.client
import cv2
import numpy as np
import pythoncom
import shutil
app = Flask(__name__)
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'gesture'
mysql = MySQL(app)
app.secret_key = 'your_secret_key'

# Define upload settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
SLIDES_FOLDER = os.path.join(BASE_DIR, 'static', 'slides')
ALLOWED_EXTENSIONS = {'ppt', 'pptx', 'ppsx'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SLIDES_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """Check if the uploaded file is a valid format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

class HandTracker:
    def __init__(self, static_mode=False, max_hands=2, detection_confidence=0.5, tracking_confidence=0.5):
        self.static_mode = static_mode
        self.max_hands = max_hands
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.static_mode,
            max_num_hands=self.max_hands,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )
        self.mpDraw = mp.solutions.drawing_utils
        self.finger_tips = [4, 8, 12, 16, 20]

    def detect_hands(self, frame, draw_landmarks=True, draw_bounds=True, mirror=True):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb_frame)
        detected_hands = []

        if self.results.multi_hand_landmarks:
            for hand_label, hand_points in zip(self.results.multi_handedness, self.results.multi_hand_landmarks):
                hand_info = self._process_hand_landmarks(frame, hand_points, hand_label, mirror)
                detected_hands.append(hand_info)

                if draw_landmarks:
                    self.mpDraw.draw_landmarks(frame, hand_points, self.mpHands.HAND_CONNECTIONS)
                if draw_bounds:
                    bounds = hand_info["bounds"]
                    cv2.rectangle(frame, (bounds[0] - 20, bounds[1] - 20),
                                  (bounds[0] + bounds[2] + 20, bounds[1] + bounds[3] + 20),
                                  (255, 0, 255), 2)

        return (detected_hands, frame) if (draw_landmarks or draw_bounds) else detected_hands

    def _process_hand_landmarks(self, frame, hand_points, hand_label, mirror):
        height, width, _ = frame.shape
        point_coords = []
        x_coords = []
        y_coords = []

        for landmark in hand_points.landmark:
            x, y, z = int(landmark.x * width), int(landmark.y * height), int(landmark.z * width)
            point_coords.append([x, y, z])
            x_coords.append(x)
            y_coords.append(y)

        bound_x_min, bound_x_max = min(x_coords), max(x_coords)
        bound_y_min, bound_y_max = min(y_coords), max(y_coords)
        bound_width = bound_x_max - bound_x_min
        bound_height = bound_y_max - bound_y_min
        center_x = bound_x_min + (bound_width // 2)
        center_y = bound_y_min + (bound_height // 2)

        return {
            "landmarks": point_coords,
            "bounds": (bound_x_min, bound_y_min, bound_width, bound_height),
            "center": (center_x, center_y),
            "type": "Left" if hand_label.classification[0].label == "Right" else "Right" if mirror else
            hand_label.classification[0].label
        }

    def check_fingers(self, hand_info):
        landmarks = hand_info["landmarks"]
        hand_type = hand_info["type"]
        finger_states = []

        # Thumb check
        if hand_type == "Right":
            finger_states.append(1 if landmarks[self.finger_tips[0]][0] > landmarks[self.finger_tips[0] - 1][0] else 0)
        else:
            finger_states.append(1 if landmarks[self.finger_tips[0]][0] < landmarks[self.finger_tips[0] - 1][0] else 0)

        # Other fingers
        for finger in range(1, 5):
            finger_states.append(
                1 if landmarks[self.finger_tips[finger]][1] < landmarks[self.finger_tips[finger] - 2][1] else 0)

        return finger_states
def draw_dashed_line(frame, start, end, color, thickness=1, style='dotted', gap=20):
    distance = ((start[0] - end[0]) ** 2 + (start[1] - end[1]) ** 2) ** .5
    points = []

    for i in np.arange(0, distance, gap):
        ratio = i / distance
        x = int((start[0] * (1 - ratio) + end[0] * ratio) + .5)
        y = int((start[1] * (1 - ratio) + end[1] * ratio) + .5)
        points.append((x, y))

    if style == 'dotted':
        for point in points:
            cv2.circle(frame, point, thickness, color, -1)
    else:
        point1 = points[0]
        point2 = points[0]
        for idx, point in enumerate(points):
            point1 = point2
            point2 = point
            if idx % 2:
                cv2.line(frame, point1, point2, color, thickness)


def draw_dashed_polygon(frame, points, color, thickness=1, style='dotted'):
    start = points[0]
    end = points[0]
    points.append(points.pop(0))
    for point in points:
        start = end
        end = point
        draw_dashed_line(frame, start, end, color, thickness, style)


def draw_dashed_rectangle(frame, point1, point2, color, thickness=1, style='dotted'):
    points = [point1, (point2[0], point1[1]), point2, (point1[0], point2[1])]
    draw_dashed_polygon(frame, points, color, thickness, style)


def convert_ppt_to_images(ppt_path):
    Application = None
    Presentation = None

    try:
        # Initialize COM
        pythoncom.CoInitialize()

        # Get absolute paths
        current_dir = os.path.dirname(os.path.abspath(__file__))
        ppt_absolute_path = os.path.abspath(ppt_path)

        # Create PowerPoint application object
        Application = win32com.client.Dispatch("PowerPoint.Application")

        # Open presentation
        Presentation = Application.Presentations.Open(ppt_absolute_path, WithWindow=False)

        # Create slides folder path with absolute path
        presentation_name = os.path.splitext(os.path.basename(ppt_path))[0]
        slides_folder = os.path.join(current_dir, 'static', 'slides', presentation_name)

        # Ensure the directory exists with proper permissions
        if os.path.exists(slides_folder):
            # Remove existing folder and its contents
            import shutil
            shutil.rmtree(slides_folder)

        # Create new directory
        os.makedirs(slides_folder, exist_ok=True)

        # Export slides as images with absolute paths
        for i, slide in enumerate(Presentation.Slides):
            image_path = os.path.join(slides_folder, f"{i + 1}.png")
            # Ensure the path is absolute and exists
            abs_image_path = os.path.abspath(image_path)
            print(f"Exporting slide {i + 1} to: {abs_image_path}")
            slide.Export(abs_image_path, "PNG")

        return slides_folder

    except Exception as e:
        print(f"Error converting PPT to images: {e}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Presentation path: {ppt_absolute_path}")
        raise

    finally:
        try:
            if Presentation:
                Presentation.Close()
            if Application:
                Application.Quit()
        except:
            pass

        # Uninitialize COM
        pythoncom.CoUninitialize()


@app.route('/')
def home():
    """Render landing page."""
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handle user registration."""
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']

        if password != confirm_password:
            flash('Passwords do not match.', 'danger')
            return redirect(url_for('register'))

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()

        if user:
            flash('Email already exists.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password)
        cursor.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", 
                       (name, email, hashed_password))
        mysql.connection.commit()
        cursor.close()
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handle user login."""
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        cursor = mysql.connection.cursor()
        cursor.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cursor.fetchone()
        print(user)
        cursor.close()
        if user and check_password_hash(user[3], password):
            session['logged_in'] = True
            session['user_id'] = user[0]
            session['name'] = user[1]
            session['email'] = user[2]
            flash('Login successful!', 'success')
            return redirect(url_for('dashboard'))
        else:
            flash('Invalid email or password.', 'danger')

    return render_template('login.html')


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'success': False, 'message': 'No file part'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'success': False, 'message': 'No selected file'})

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                # Convert PPT to images
                slides_folder = convert_ppt_to_images(file_path)

                # Store the absolute path in session
                session['slides_folder'] = os.path.abspath(slides_folder)

                return jsonify({
                    'success': True,
                    'filename': filename,
                    'file_path': slides_folder
                })
            except Exception as e:
                print(f"Error processing file: {e}")
                return jsonify({'success': False, 'message': str(e)})

        return jsonify({'success': False, 'message': 'Invalid file type'})

    return render_template('dashboard.html')


@app.route('/start_control', methods=['POST'])
def start_control():
    if 'slides_folder' not in session:
        return jsonify({'success': False, 'message': 'No presentation loaded'})

    try:
        slides_folder = session.get('slides_folder')
        print(f"Starting gesture control with slides folder: {slides_folder}")

        if not os.path.exists(slides_folder):
            return jsonify({'success': False,
                            'message': f'Slides folder not found: {slides_folder}'})

        start_gesture_control(slides_folder)
        return jsonify({'success': True, 'message': 'Gesture control started successfully'})

    except Exception as e:
        print(f"Error starting gesture control: {e}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/profile')
def profile():
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    cursor = mysql.connection.cursor()
    cursor.execute("SELECT * FROM users WHERE id = %s", [session['user_id']])
    user = cursor.fetchone()
    cursor.close()
    print(user)
    return render_template('profile.html', user=user)



@app.route('/logout')
def logout():
    """Logout user."""
    session.clear()
    flash('You have been logged out!', 'success')
    return redirect(url_for('home'))


def start_gesture_control(slides_folder):
    # Initialize variables
    screen_width = 1920
    screen_height = 1080
    current_slide = 0
    camera_height, camera_width = int(120 * 1.0), int(213 * 1.0)  # Slightly reduced camera size
    gesture_boundary_y = 400
    gesture_boundary_x = 750
    gesture_active = False
    gesture_timer = 0
    gesture_timeout = 15
    drawing_strokes = [[]]
    stroke_count = 0
    drawing_active = False
    # Adjust zoom settings
    zoom_scale = 1.0
    zoom_max = 3.0
    zoom_min = 1.0
    zoom_speed = 0.005  # Reduced zoom sensitivity
    previous_distance = None
    if not os.path.exists(slides_folder):
        raise FileNotFoundError(f"Slides folder not found: {slides_folder}")

    slide_files = sorted(os.listdir(slides_folder), key=len)
    print(slide_files)

    # Initialize webcam
    video_input = cv2.VideoCapture(0)
    video_input.set(3, screen_width)
    video_input.set(4, screen_height)
    tracker = HandTracker(detection_confidence=0.8, max_hands=1)

    # Create a named window and set it to fullscreen
    cv2.namedWindow("Presentation", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Presentation", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        ret, frame = video_input.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        current_slide_path = os.path.join(slides_folder, slide_files[current_slide])
        print(f"Loading slide: {current_slide_path}")

        slide_image = cv2.imread(current_slide_path)
        if slide_image is None:
            raise FileNotFoundError(f"Failed to load slide: {current_slide_path}")

        # Resize slide to match screen
        slide_image = cv2.resize(slide_image, (screen_width, screen_height))

        # Apply zoom to the slide
        if zoom_scale != 1.0:
            zoomed_width = int(screen_width * zoom_scale)
            zoomed_height = int(screen_height * zoom_scale)
            zoomed_slide = cv2.resize(slide_image, (zoomed_width, zoomed_height))
            start_x = (zoomed_width - screen_width) // 2
            start_y = (zoomed_height - screen_height) // 2
            slide_image = zoomed_slide[start_y:start_y + screen_height,
                          start_x:start_x + screen_width]
        else:
            slide_image = cv2.resize(slide_image, (screen_width, screen_height))

        # Process hand tracking
        # Process hand tracking and gesture control
        hands, frame = tracker.detect_hands(frame)

        draw_dashed_rectangle(frame, (screen_width, 0),
                              (gesture_boundary_x, gesture_boundary_y),
                              (0, 255, 0), 5, 'dotted')

        if hands and not gesture_active:
            hand = hands[0]
            center_x, center_y = hand["center"]
            landmarks = hand["landmarks"]
            finger_states = tracker.check_fingers(hand)

            # Map drawing coordinates
            x_pos = int(np.interp(landmarks[8][0],
                                  [screen_width // 2, screen_width],
                                  [0, screen_width]))
            y_pos = int(np.interp(landmarks[8][1],
                                  [150, screen_height - 150],
                                  [0, screen_height]))
            draw_point = x_pos, y_pos

            # Zoom gesture detection
            if finger_states == [1, 1, 0, 0, 0]:  # Thumb and index finger extended
                # Get the thumb and index finger positions
                thumb_x, thumb_y = landmarks[4][0], landmarks[4][1]
                index_x, index_y = landmarks[8][0], landmarks[8][1]

                # Calculate current distance between fingers
                current_distance = math.hypot(index_x - thumb_x, index_y - thumb_y)

                # Update zoom based on finger movement
                if previous_distance:
                    distance_diff = current_distance - previous_distance

                    # Zoom in/out based on finger movement
                    zoom_scale += distance_diff * zoom_speed

                    # Constrain zoom scale
                    zoom_scale = max(zoom_min, min(zoom_max, zoom_scale))

                previous_distance = current_distance
                drawing_active = False
            else:
                previous_distance = None

            if center_y < gesture_boundary_y and center_x > gesture_boundary_x:
                drawing_active = False

                # Previous slide gesture
                if finger_states == [1, 0, 0, 0, 0]:
                    if current_slide > 0:
                        gesture_active = True
                        current_slide -= 1
                        drawing_strokes = [[]]
                        stroke_count = 0
                        zoom_scale = 1.0  # Reset zoom when changing slides

                # Next slide gesture
                if finger_states == [0, 0, 0, 0, 1]:
                    if current_slide < len(slide_files) - 1:
                        gesture_active = True
                        current_slide += 1
                        drawing_strokes = [[]]
                        stroke_count = 0
                        zoom_scale = 1.0  # Reset zoom when changing slides

                # Clear drawings gesture
                if finger_states == [1, 1, 1, 1, 1]:
                    if drawing_strokes:
                        if stroke_count >= 0:
                            drawing_strokes.clear()
                            stroke_count = 0
                            gesture_active = True
                            drawing_strokes = [[]]

            # Pointer gesture
            if finger_states == [0, 1, 1, 0, 0]:
                cv2.circle(slide_image, draw_point, 4, (0, 0, 255), cv2.FILLED)
                drawing_active = False

            # Drawing gesture
            if finger_states == [0, 1, 0, 0, 0]:
                if not drawing_active:
                    drawing_active = True
                    stroke_count += 1
                    drawing_strokes.append([])
                drawing_strokes[stroke_count].append(draw_point)
                cv2.circle(slide_image, draw_point, 4, (0, 0, 255), cv2.FILLED)
            else:
                drawing_active = False

            # Erase gesture
            if finger_states == [0, 1, 1, 1, 0]:
                if drawing_strokes:
                    if stroke_count >= 0:
                        drawing_strokes.pop(-1)
                        stroke_count -= 1
                        gesture_active = True

        else:
            drawing_active = False

        # Handle gesture cooldown
        if gesture_active:
            gesture_timer += 1
            if gesture_timer > gesture_timeout:
                gesture_timer = 0
                gesture_active = False

        # Render drawings
        for stroke_idx, stroke in enumerate(drawing_strokes):
            for point_idx in range(len(stroke)):
                if point_idx != 0:
                    cv2.line(slide_image, stroke[point_idx - 1],
                             stroke[point_idx], (0, 0, 255), 6)

        preview = cv2.resize(frame, (camera_width, camera_height))
        slide_image[10:10 + camera_height, screen_width - camera_width - 10:screen_width - 10] = preview

        # Add zoom level indicator (adjusted position for fullscreen)
        cv2.putText(slide_image, f"Zoom: {zoom_scale:.2f}x", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Add slide counter at bottom center
        slide_text = f"Slide {current_slide + 1}/{len(slide_files)}"
        # Get text size for centering
        text_size = cv2.getTextSize(slide_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (screen_width - text_size[0]) // 2
        text_y = screen_height - 30  # 30 pixels from bottom
        # Add dark background for better visibility
        cv2.rectangle(slide_image,
                      (text_x - 10, text_y - text_size[1] - 10),
                      (text_x + text_size[0] + 10, text_y + 10),
                      (0, 0, 0), -1)
        # Add text
        cv2.putText(slide_image, slide_text, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow("Presentation", slide_image)

        # Check for quit condition
        if cv2.waitKey(1) == ord('q'):
            break

    video_input.release()
    cv2.destroyAllWindows()
    shutil.rmtree(SLIDES_FOLDER)
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.isfile(file_path):
            os.remove(file_path)
            print(f"Deleted: {file_path}")


if __name__ == '__main__':
    app.run(debug=True, port=5008)
    
