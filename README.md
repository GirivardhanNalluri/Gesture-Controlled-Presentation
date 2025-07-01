# ✋ Gesture Controlled Presentation

This project is a **gesture recognition-based presentation controller** developed by me. It uses a webcam to detect hand gestures in real time and allows users to control presentation slides without touching the keyboard or mouse. Built with Python, OpenCV, and MediaPipe, the application runs through a web-based interface using Flask.

## 📌 Features

- 👋 Real-time hand gesture recognition using webcam
- 🖼️ Slide navigation using natural gestures
- 🔧 OpenCV and MediaPipe for hand tracking and gesture analysis
- 🌐 Flask-based web interface with file upload support
- 🎨 Clean, responsive UI with CSS and HTML

## 🗂️ Project Structure

```
Gesture/
├── app.py                   # Flask application
├── cam.py                   # Gesture recognition using webcam
├── static/                  # CSS and image assets
│   ├── css/
│   └── images/
├── templates/               # HTML templates
│   ├── index.html
│   ├── dashboard.html
│   └── base.html
├── uploaded_files/          # Stores uploaded files
├── __pycache__/             # Compiled Python cache
└── .idea/                   # IDE-specific configs (can be ignored)
```

## ⚙️ How to Run

1. **Clone the Repository**
   ```bash
   git clone https://github.com/your-username/gesture-controlled-presentation.git
   cd gesture-controlled-presentation/Gesture
   ```

2. **Create a Virtual Environment (optional but recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate        # On Windows: venv\Scripts\activate
   ```

3. **Install Required Libraries**
   ```bash
   pip install flask opencv-python mediapipe
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Open in Browser**
   Navigate to `http://127.0.0.1:5000` in your web browser.

## 🛠️ Technologies Used

- **Python**
- **OpenCV**
- **MediaPipe**
- **Flask**
- **HTML5/CSS3**

## 👨‍💻 Developed By

- **Nalluri Girivardhan**

## 📬 Contact

For questions, suggestions, or collaborations, feel free to reach out:  
📧 girivardhan2301@gmail.com

---
