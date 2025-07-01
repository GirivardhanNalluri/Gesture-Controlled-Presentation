from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_mysqldb import MySQL
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import os


app = Flask(__name__)

# Configure MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_PORT'] = 3306
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = 'GirI!2329'
app.config['MYSQL_DB'] = 'hand'
mysql = MySQL(app)
app.secret_key = 'your_secret_key'

# Define upload settings
UPLOAD_FOLDER = 'uploaded_files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'ppt', 'pptx', 'ppsx'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


def allowed_file(filename):
    """Check if the uploaded file is a valid format."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



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
    """Handle file upload and storage."""
    if 'logged_in' not in session:
        return redirect(url_for('login'))

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded.', 'danger')
            return jsonify({"success": False, "message": "No file uploaded."})

        file = request.files['file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return jsonify({"success": False, "message": "No file selected."})

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            cursor = mysql.connection.cursor()
            cursor.execute("INSERT INTO presentations (filename, file_path, uploaded_by) VALUES (%s, %s, %s)", 
                           (filename, filepath, session['user_id']))
            mysql.connection.commit()
            cursor.close()
            os.remove(filepath)
	    

            return jsonify({'success': True, 'file_path': filepath, 'filename': filename})

    return render_template('dashboard.html', name=session['name'])

@app.route('/start_control', methods=['POST'])
def start_control():
    """Handle Start Control action."""
    if 'logged_in' not in session:
        return jsonify({'success': False, 'message': 'User not authenticated.'})

    # You can define the logic to trigger gestures or specific actions here.
    # For example:
    response_message = "Gesture control activated successfully!"

    return jsonify({'success': True, 'message': response_message})

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

if __name__ == '__main__':
    app.run(debug=True, port=5008)
    