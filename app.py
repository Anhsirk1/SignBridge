from flask import Flask, render_template, jsonify, session, redirect, url_for
from models.models import db, User
from controllers.controllers import controllers
from werkzeug.security import generate_password_hash
import os
import subprocess
import threading
import time

app = Flask(__name__)

# Database Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///anhsirk20.sqlite'  # Will create in the root folder
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SECRET_KEY'] = 'password'

# Initialize Database
db.init_app(app)

# Create tables if they don't exist
with app.app_context():
    db.create_all()
    print("Database and tables created successfully.")

# Register the blueprint
app.register_blueprint(controllers)


@app.route('/')
def home():
    session['visited_home'] = True
    return render_template('index.html')

@app.route('/login')
def login():
    if 'visited_home' not in session:
        return redirect(url_for('home'))
    return render_template('login_register.html')

def run_gesture_script():
    subprocess.run(["python", "classify_webcam.py"])

@controllers.route('/dashboard')
def dashboard():
    prediction = "Waiting for prediction..."
    if os.path.exists("static/prediction.txt"):
        with open("static/prediction.txt", "r") as f:
            prediction = f.read()
    return render_template('dashboard.html', prediction=prediction)

@app.route('/contact')
def contact():
    if 'visited_home' not in session:
        return redirect(url_for('home'))
    return render_template('contact.html')

if __name__ == "__main__":
    app.run(debug=True)