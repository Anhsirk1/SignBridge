# classify_webcam.py
import os
from urllib.parse import urlparse, parse_qs
import numpy as np
import cv2
import tensorflow as tf
import threading
import time
import http.server
import socketserver

# Suppress TensorFlow warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ───────────────────────────── HTML & JS ──────────────────────────────
HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Sign Language Suggestions</title>
  <link href='https://unpkg.com/boxicons@2.1.4/css/boxicons.min.css' rel='stylesheet'>
  <style>
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }

    body {
        background: linear-gradient(135deg, #ff7b54 0%, #ff9068 100%);
        color: #333;
        min-height: 100vh;
    }

    .admin-jpg {
        width: 50px;
        border-radius: 100%;
        border: 2px solid #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .sidebar {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        width: 80px;
        background: linear-gradient(135deg, #12171e 0%, #12171e 100%);
        padding: 0.4rem 0.8rem;
        transition: all 0.5s ease;
        z-index: 1000;
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }

    .sidebar.active {
        width: 250px;
    }

    .sidebar #btn {
        position: absolute;
        color: #fff;
        top: .4rem;
        left: 50%;
        font-size: 1.2rem;
        line-height: 50px;
        transform: translateX(-50%);
        cursor: pointer;
        transition: all 0.3s ease;
    }

    .sidebar.active #btn {
        left: 90%;
    }

    .sidebar .top .logo {
        color: #fff;
        display: flex;
        height: 50px;
        width: 100%;
        align-items: center;
        pointer-events: none;
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .sidebar.active .top .logo {
        opacity: 1;
    }

    .top .logo i {
        font-size: 2rem;
        margin-right: 5px;
        background: linear-gradient(45deg, #ff7b54, #ff9068);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .user {
        display: flex;
        align-items: center;
        margin: 1rem 0;
    }

    .user p {
        color: #fff;
        opacity: 1;
        margin-left: 1rem;
    }

    .bold {
        font-weight: 600;
    }

    .sidebar p {
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .sidebar.active p {
        opacity: 1;
    }

    .sidebar ul li {
        position: relative;
        list-style-type: none;
        height: 50px;
        width: 90%;
        margin: 0.8rem auto;
        line-height: 50px;
    }

    .sidebar ul li a {
        color: #fff;
        display: flex;
        align-items: center;
        text-decoration: none;
        border-radius: 0.8rem;
        transition: all 0.3s ease;
    }

    .sidebar ul li a:hover {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: #fff;
        transform: translateX(5px);
    }

    .sidebar ul li a i {
        min-width: 50px;
        text-align: center;
        height: 50px;
        border-radius: 12px;
        line-height: 50px;
    }

    .sidebar .nav-item {
        opacity: 0;
        transition: opacity 0.3s ease;
    }

    .sidebar.active .nav-item {
        opacity: 1;
    }

    .sidebar ul li .tooltip {
        position: absolute;
        left: 125px;
        top: 50%;
        transform: translate(-50%, -50%);
        box-shadow: 0 0.5rem 0.8rem rgba(0, 0, 0, 0.2);
        border-radius: .6rem;
        padding: .4rem 1.2rem;
        line-height: 1.8rem;
        z-index: 20;
        opacity: 0;
        background: #fff;
        color: #333;
        transition: opacity 0.3s ease;
    }

    .sidebar ul li:hover .tooltip {
        opacity: 1;
    }

    .sidebar.active ul li .tooltip {
        display: none;
    }

    .main-container {
        margin-left: 80px;
        min-height: 100vh;
        display: flex;
        transition: margin-left 0.5s ease;
    }

    .sidebar.active ~ .main-container {
        margin-left: 250px;
    }

    .left-panel {
        flex: 1;
        padding: 30px;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        background: #e6edf8;
        backdrop-filter: blur(10px);
        border-right: 1px solid rgba(255,255,255,0.2);
    }

    .right-panel {
        flex: 1;
        padding: 15px;
        background: rgba(255,255,255,0.95);
        backdrop-filter: blur(10px);
        overflow-y: auto;
    }

    .app-title {
        text-align: center;
        margin-bottom: 30px;
        color: #000;
        font-size: 2.5em;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }

    .webcam-section {
        background: rgba(255,255,255,0.15);
        border-radius: 20px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0,0,0,0.1);
        border: 1px solid rgba(255,255,255,0.2);
        text-align: center;
    }

    .webcam-section h2 {
        color: #000;
        margin-bottom: 20px;
        font-size: 1.5em;
    }

    #webcam-container img {
        border: 3px solid rgba(255,255,255,0.3);
        border-radius: 15px;
        width: 100%;
        max-width: 400px;
        height: 300px;
        background: rgba(255,255,255,0.1);
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    .sequence-display {
        margin-top: 20px;
        padding: 20px;
        background: rgba(255,255,255,0.2);
        border-radius: 15px;
        color: #000;
        font-size: 1.2em;
        border: 1px solid rgba(255,255,255,0.3);
    }

    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-left: 10px;
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0% { transform: scale(1); opacity: 1; }
        50% { transform: scale(1.1); opacity: 0.7; }
        100% { transform: scale(1); opacity: 1; }
    }

    .status-listening { background-color: #ffa500; }
    .status-connected { background-color: #4caf50; }
    .status-error { background-color: #f44336; }

    .content-section {
        background: #fff;
        border-radius: 10px;
        padding: 14px;
        margin-bottom: 14px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
    }

    .content-section h3 {
        color: #333;
        margin-bottom: 8px;
        font-size: 1.3em;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .content-section h3 i {
        color: #667eea;
    }

    .language-buttons {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        gap: 10px;
        margin-top: 15px;
    }

    .lang-btn {
        padding: 12px 16px;
        background: linear-gradient(45deg, #f8f9ff, #e8f0ff);
        color: #2196f3;
        border: 2px solid #e3f2fd;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-weight: 500;
        font-size: 0.9em;
        text-align: center;
    }

    .lang-btn:hover {
        background: linear-gradient(45deg, #2196f3, #1976d2);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }

    .lang-btn.active {
        background: linear-gradient(45deg, #2196f3, #1976d2);
        color: white;
        box-shadow: 0 4px 15px rgba(33, 150, 243, 0.4);
    }

    .suggestions-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 10px;
        margin-top: 15px;
    }

    .suggestion {
        padding: 15px 20px;
        background: linear-gradient(45deg, #e0f7fa, #b2ebf2);
        color: #00796b;
        border-radius: 12px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
        font-weight: 500;
        border: 2px solid transparent;
    }

    .suggestion:hover {
        background: linear-gradient(45deg, #00796b, #004d40);
        color: white;
        transform: translateY(-3px);
        box-shadow: 0 6px 20px rgba(0, 121, 107, 0.3);
        border-color: #00796b;
    }

    .translation-container {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        border-radius: 15px;
        padding: 8px;
        margin-top: 8px;
    }

    .text-row {
        display: flex;
        align-items: center;
        gap: 15px;
        margin-bottom: 15px;
        padding: 15px;
        border-radius: 10px;
        background: #fff;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .text-row:last-child {
        margin-bottom: 0;
    }

    .text-label {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        padding: 8px 15px;
        border-radius: 20px;
        font-weight: 600;
        min-width: 100px;
        text-align: center;
        font-size: 0.9em;
    }

    .text-content {
        flex: 1;
        font-size: 1.1em;
        color: #333;
        font-weight: 500;
    }

    .audio-controls {
        display: flex;
        gap: 10px;
        justify-content: center;
        margin-top: 20px;
    }

    .audio-btn {
        padding: 12px 20px;
        background: linear-gradient(45deg, #ff9800, #f57c00);
        color: white;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        transition: all 0.3s ease;
        font-size: 0.9em;
        display: flex;
        align-items: center;
        gap: 8px;
        font-weight: 500;
    }

    .audio-btn:hover {
        background: linear-gradient(45deg, #f57c00, #ef6c00);
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(245, 124, 0, 0.4);
    }

    .loading {
        color: #666;
        font-style: italic;
        animation: loading-dots 1.5s infinite;
    }

    @keyframes loading-dots {
        0%, 20% { content: ''; }
        40% { content: '.'; }
        60% { content: '..'; }
        80%, 100% { content: '...'; }
    }

    /* Responsive Design */
    @media (max-width: 1024px) {
        .main-container {
            flex-direction: column;
        }
        
        .left-panel, .right-panel {
            flex: none;
        }
        
        .left-panel {
            border-right: none;
            border-bottom: 1px solid rgba(255,255,255,0.2);
        }
        
        .language-buttons {
            grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        }
        
        .suggestions-grid {
            grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
        }
    }

    @media (max-width: 768px) {
        .text-row {
            flex-direction: column;
            align-items: flex-start;
            gap: 10px;
        }
        
        .text-label {
            min-width: auto;
        }
        
        .audio-controls {
            flex-direction: column;
            align-items: center;
        }
    }
  </style>
</head>
<body>
    <div class="sidebar">
        <div class="top">
            <div class="logo">
                <i class="bx bxl-codepen"></i>
                <span>Sign Bridge</span>
            </div>
            <i class="bx bx-menu" id="btn"></i>
        </div>
        <div class="user">
            <img src="https://cdn-icons-png.flaticon.com/512/2206/2206368.png" alt="admin" class="admin-jpg">
            <div>
                <p class="bold">Anhsirk</p>
                <p>User</p>
            </div>
        </div>
        <ul>
            <li>
                <a href="#" class="active">
                    <i class="bx bxs-grid-alt"></i>
                    <span class="nav-item">Dashboard</span>
                </a>
                <span class="tooltip">Dashboard</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bx-home"></i>
                    <span class="nav-item">Home</span>
                </a>
                <span class="tooltip">Home</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bxs-user-detail"></i>
                    <span class="nav-item">Users</span>
                </a>
                <span class="tooltip">Users</span>
            </li>
            <li>
                <a href="dictionary.html">
                    <i class="bx bx-file"></i>
                    <span class="nav-item">Dictionary</span>
                </a>
                <span class="tooltip">Dictionary</span>
            </li>
            <li>
                <a href="community.html">
                    <i class="bx bx-group"></i>
                    <span class="nav-item">Community</span>
                </a>
                <span class="tooltip">Community</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bx-cog"></i>
                    <span class="nav-item">Settings</span>
                </a>
                <span class="tooltip">Settings</span>
            </li>
            <li>
                <a href="#">
                    <i class="bx bx-log-out"></i>
                    <span class="nav-item">Logout</span>
                </a>
                <span class="tooltip">Logout</span>
            </li>
        </ul>
    </div>

    <div class="main-container">
        <!-- Left Panel - Webcam and Sequence -->
        <div class="left-panel">
            <h1 class="app-title">Sign Language Translator</h1>
            
            <div class="webcam-section">
                <h2><i class="bx bx-video"></i> Live Camera Feed</h2>
                <div id="webcam-container">
                    <img src="/video_feed" alt="Webcam stream" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y4ZjlmYSIvPjx0ZXh0IHg9IjIwMCIgeT0iMTUwIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2Yzc1N2QiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkNhbWVyYSBTdHJlYW0gVW5hdmFpbGFibGU8L3RleHQ+PC9zdmc+'">
                </div>
                
                <div class="sequence-display">
                    <strong>Detected Sequence:</strong> <span id="signed-seq">[Listening...]</span>
                    <span id="connection-status" class="status-indicator status-listening"></span>
                </div>
            </div>
        </div>

        <!-- Right Panel - Controls and Results -->
        <div class="right-panel">
            <!-- Language Selection -->
            <div class="content-section">
                <h3><i class="bx bx-world"></i> Choose Translation Language</h3>
                <div class="language-buttons">
                    <button class="lang-btn active" data-lang="en" data-lang-name="English">English</button>
                    <button class="lang-btn" data-lang="hi" data-lang-name="हिंदी">Hindi</button>
                    <button class="lang-btn" data-lang="te" data-lang-name="తెలుగు">Telugu</button>
                    <button class="lang-btn" data-lang="ta" data-lang-name="தமிழ்">Tamil</button>
                    <button class="lang-btn" data-lang="kn" data-lang-name="ಕನ್ನಡ">Kannada</button>
                    <button class="lang-btn" data-lang="ml" data-lang-name="മലയാളം">Malayalam</button>
                    <button class="lang-btn" data-lang="mr" data-lang-name="मराठी">Marathi</button>
                    <button class="lang-btn" data-lang="gu" data-lang-name="ગુજરાતી">Gujarati</button>
                    <button class="lang-btn" data-lang="bn" data-lang-name="বাংলা">Bengali</button>
                    <button class="lang-btn" data-lang="pa" data-lang-name="ਪੰਜਾਬੀ">Punjabi</button>
                </div>
            </div>

            <!-- Suggestions -->
            <div class="content-section">
                <h3><i class="bx bx-bulb"></i> Suggested Phrases</h3>
                <div id="suggestions" class="suggestions-grid">
                    <!-- Suggestions will be populated here -->
                </div>
            </div>

            <!-- Translation Results -->
            <div class="content-section">
                <h3><i class="bx bx-transfer-alt"></i> Translation Results</h3>
                <div class="translation-container">
                    <div class="text-row">
                        <div class="text-label">Original</div>
                        <div class="text-content" id="original-sentence">[Click a suggestion above]</div>
                    </div>
                    <div class="text-row">
                        <div class="text-label">Translation</div>
                        <div class="text-content" id="translated-sentence">[Select language and click suggestion]</div>
                    </div>
                    <div class="audio-controls">
                        <button class="audio-btn" id="play-original">
                            <i class="bx bx-play"></i> Play Original
                        </button>
                        <button class="audio-btn" id="play-translation">
                            <i class="bx bx-play"></i> Play Translation
                        </button>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
/* ────────────────────────────────────────────────
   0.  FULL SUGGESTIONS LIST  (A‑Z)
────────────────────────────────────────────────── */
const suggestionsData = [
  /*  A  */ "A","AM","AND","ARE","ALL","ABOUT","AFTER","AGAIN","ASK","AWESOME",
  "AM FINE","AM GOOD","AM HAPPY","AND YOU","ARE YOU","ASK YOU",
  "ARE YOU FREE TODAY","AM GOING TO WORK","AND HOW ABOUT YOU","AWESOME TO MEET YOU",
  "ALL IS WELL","AFTER WORK I GO HOME","AGAIN NICE TO SEE YOU","ASK ME ANYTHING",
  /*  B  */ "B","BE","BY","BUT","BYE","BECAUSE","BEFORE","BEAUTIFUL","BEST","BUSY",
  "BE GOOD","BYE BYE","BE HAPPY","BE SAFE","BECAUSE I","BEFORE YOU",
  "BYE SEE YOU LATER","BE CAREFUL DRIVING HOME","BECAUSE I HAVE WORK TODAY",
  "BEAUTIFUL DAY TODAY","BEST OF LUCK TO YOU","BUSY WITH FAMILY RIGHT NOW",
  /*  C  */ "C","CAN","COME","COOL","CALL","CHANGE","CHOOSE","CONGRATULATIONS","CLEAN","CAREFUL",
  "CAN YOU","COME HERE","CALL ME","CAN I","COME ON","COOL DOWN",
  "CAN YOU HELP ME PLEASE","COME VISIT US SOMETIME","CALL ME WHEN YOU ARRIVE",
  "CAN I ASK YOU SOMETHING","CONGRATULATIONS ON YOUR SUCCESS","COOL TO MEET YOU TODAY",
  /*  D  */ "D","DO","DAY","DONE","DRIVE","DRINK","DIFFERENT","DOCTOR","DANCE","DIFFICULT",
  "DO YOU","GOOD DAY","DONE WORK","DRIVE SAFE","DO IT","DAY GOOD",
  "DO YOU HAVE TIME TO TALK","DONE WITH WORK FOR TODAY","DRIVE SAFE AND CALL WHEN HOME",
  "DAY WENT REALLY WELL","DOCTOR APPOINTMENT WENT GOOD","DIFFERENT FROM WHAT I EXPECTED",
  /*  E  */ "E","EAT","EVERY","EXCELLENT","EXCITED","EMERGENCY","EASY","EARLY","EVENING","EXPLAIN",
  "EAT FOOD","EVERY DAY","EXCITED TO","EASY TO","EVENING GOOD","EXPLAIN TO",
  "EAT DINNER WITH FAMILY TONIGHT","EVERY DAY IS A BLESSING","EXCITED TO SEE YOU SOON",
  "EXCELLENT WORK YOU DID","EARLY MORNING WORKOUT FEELS GOOD","EVENING PLANS WITH FRIENDS",
  /*  F  */ "F","FINE","FROM","FRIEND","FAMILY","FOOD","FUN","FEEL","FINISH","FORGET",
  "FINE THANK","FROM WHERE","FRIEND GOOD","FAMILY GOOD","FEEL GOOD","FINISH WORK",
  "FINE THANK YOU FOR ASKING","FAMILY IS DOING WELL","FRIEND COMING TO VISIT TODAY",
  "FOOD WAS DELICIOUS THANK YOU","FUN TIME AT THE PARTY","FEEL MUCH BETTER NOW",
  /*  G  */ "GO","GOOD","GOOD MORNING","GOOD NIGHT","GET","GIVE","GREAT","GIRL","GAME","GREEN","GROW",
  "GO HOME","GET UP","GIVE ME","GREAT JOB","GOOD BYE","GOOD DAY","GOOD LUCK","GO AHEAD",
  "GET READY","GOOD WORK","GOOD MORNING HOW ARE YOU","GOOD NIGHT SLEEP WELL",
  "GREAT TO SEE YOU AGAIN","GO HOME AND REST WELL","GET READY FOR THE MEETING",
  "GOOD LUCK WITH YOUR INTERVIEW",
  /*  H  */ "H","HI","HELLO","HOW","HELP","HAPPY","HOME","HOSPITAL","HEAR","HOPE",
  "HI THERE","HELLO FRIEND","HOW ARE","HOW ARE YOU","HELP ME","HAPPY TO",
  "HELLO I","HELLO I AM","HELLO HOW","HELP YOU","HOME SAFE","HOPE YOU",
  "HI THERE HOW ARE YOU DOING","HELLO NICE TO MEET YOU","HOW WAS YOUR DAY TODAY",
  "HELP ME WITH THIS PLEASE","HAPPY TO SEE YOU AGAIN","HOME SAFE AND SOUND",
  "HOPE YOU HAVE A GREAT DAY","HOW IS YOUR FAMILY DOING","HELLO MY NAME IS",
  /*  I  */ "I","I AM","IS","IN","IT","IF","INTERESTING","IMPORTANT","INVITE","INSIDE",
  "I AM FINE","I AM GOOD","I AM HAPPY","I LOVE","I NEED","I WANT",
  "I CAN","I WILL","I HOPE","I THANK","I UNDERSTAND","I LIKE",
  "I AM FINE THANK YOU","I AM GOING TO THE STORE","I LOVE SPENDING TIME WITH FAMILY",
  "I NEED YOUR HELP WITH SOMETHING","I WANT TO LEARN MORE SIGN","I CAN MEET YOU TOMORROW",
  "I WILL CALL YOU LATER","I HOPE YOU FEEL BETTER SOON","I UNDERSTAND WHAT YOU MEAN",
  /*  J  */ "J","JOB","JUST","JUMP","JOURNEY","JOIN","JUICE","JACKET","JEALOUS","JOKE",
  "GOOD JOB","JUST FINE","JOIN US","JOURNEY SAFE","JUICE GOOD","JUST OKAY",
  "JOB INTERVIEW WENT VERY WELL","JUST FINISHED EATING LUNCH","JOIN US FOR DINNER TONIGHT",
  "JUST WANTED TO SAY HELLO","JOURNEY TO WORK TAKES LONG TIME",
  /*  K  */ "K","KNOW","KEEP","KIND","KITCHEN","KISS","KICK","KILL","KEY","KING",
  "KNOW YOU","KEEP SAFE","KIND PERSON","KEEP GOING","KNOW WHAT","KEEP CALM",
  "KNOW YOU ARE BUSY RIGHT NOW","KEEP SAFE ON YOUR WAY HOME","KIND OF YOU TO HELP ME",
  "KEEP GOING YOU ARE DOING GREAT","KNOW WHAT YOU MEAN EXACTLY",
  /*  L  */ "L","LOVE","LIKE","LOOK","LEARN","LEAVE","LATE","LUNCH","LIFE","LISTEN",
  "LOVE YOU","LIKE YOU","LOOK AT","LEARN MORE","LEAVE NOW","LUNCH TIME",
  "LIFE GOOD","LISTEN TO","LOOK GOOD","LOVE FAMILY","LIKE IT","LEARN SIGN",
  "LOVE YOU TOO VERY MUCH","LIKE TO SPEND TIME TOGETHER","LOOK FORWARD TO SEEING YOU",
  "LEARN SOMETHING NEW EVERY DAY","LATE FOR THE MEETING SORRY","LUNCH WAS DELICIOUS TODAY",
  "LIFE IS GOOD THANK GOD","LISTEN TO WHAT YOU ARE SAYING",
  /*  M  */ "M","ME","MY","MORE","MAKE","MEET","MORNING","MONEY","MOTHER","MOVE",
  "MEET YOU","MY FAMILY","MY NAME","MORE TIME","MAKE SURE","MORNING GOOD",
  "MOTHER GOOD","MOVE ON","MY FRIEND","MAKE HAPPY","MORE WORK","MEET FRIEND",
  "MEET YOU AT THE RESTAURANT","MY FAMILY IS DOING WELL","MY NAME IS NICE TO MEET",
  "MORE TIME TO TALK LATER","MAKE SURE TO DRIVE SAFE","MORNING WORKOUT FELT GREAT",
  "MOTHER CALLED ME THIS MORNING","MOVE FORWARD WITH THE PLAN",
  /*  N  */ "N","NO","NOW","NEED","NICE","NAME","NEW","NIGHT","NEVER","NEXT",
  "NO PROBLEM","NOW TIME","NEED HELP","NICE TO","NAME IS","NEW FRIEND",
  "GOOD NIGHT","NEVER MIND","NEXT TIME","NICE DAY","NEED YOU","NOW GO",
  "NO PROBLEM HAPPY TO HELP","NOW IS GOOD TIME TO TALK","NEED TO GO HOME SOON",
  "NICE TO MEET YOU TODAY","NAME IS WHAT AGAIN SORRY","NEW JOB STARTING NEXT WEEK",
  "GOOD NIGHT SLEEP WELL","NEVER MIND IT IS OKAY","NEXT TIME WE MEET FOR COFFEE",
  /*  O  */ "O","OK","OKAY","OF","ON","OR","OPEN","OFFICE","OVER","OLD",
  "OK GOOD","OKAY FINE","OPEN DOOR","OFFICE WORK","OVER THERE","OLD FRIEND",
  "OK GOOD TO HEAR FROM YOU","OKAY FINE WITH ME","OPEN THE DOOR PLEASE",
  "OFFICE WORK KEEPS ME BUSY","OVER THERE IS MY FRIEND","OLD FRIEND FROM SCHOOL DAYS",
  /*  P  */ "P","PLEASE","PEOPLE","PROBLEM","PERFECT","PLAY","PHONE","PICK","PIZZA","PRAY",
  "PLEASE HELP","NO PROBLEM","PEOPLE GOOD","PERFECT TIME","PLAY GAME","PHONE CALL",
  "PICK UP","PIZZA GOOD","PLEASE COME","PLEASE WAIT","PLAY WITH","PEOPLE NICE",
  "PLEASE HELP ME WITH THIS","NO PROBLEM AT ALL","PEOPLE ARE VERY NICE HERE",
  "PERFECT TIME TO MEET TODAY","PLAY GAMES WITH THE KIDS","PHONE CALL FROM MY MOTHER",
  "PICK UP GROCERIES ON WAY HOME","PIZZA FOR DINNER SOUNDS GOOD",
  /*  Q  */ "Q","QUESTION","QUICK","QUIET","QUIT","QUALITY","QUITE","QUEEN","QUOTE","QUIZ",
  "QUESTION FOR","QUICK TIME","QUIET PLEASE","QUIT WORK","QUALITY GOOD","QUITE GOOD",
  "QUESTION FOR YOU IF YOU HAVE TIME","QUICK MEETING BEFORE WE GO","QUIET PLEASE BABY SLEEPING",
  "QUIT WORK EARLY TODAY","QUALITY TIME WITH FAMILY",
  /*  R  */ "R","RIGHT","READY","REMEMBER","REALLY","RUN","READ","ROOM","RELAX","RESPECT",
  "RIGHT NOW","READY TO","REMEMBER ME","REALLY GOOD","RUN FAST","READ BOOK",
  "ROOM CLEAN","RELAX TIME","RESPECT YOU","RIGHT HERE","READY GO","REALLY NICE",
  "RIGHT NOW IS GOOD TIME","READY TO GO WHEN YOU ARE","REMEMBER TO CALL ME LATER",
  "REALLY GOOD TO SEE YOU","RUN IN THE PARK THIS MORNING","READ INTERESTING BOOK LATELY",
  "ROOM LOOKS VERY CLEAN TODAY","RELAX AND ENJOY YOUR WEEKEND",
  /*  S  */ "S","SEE","SORRY","SURE","SAFE","SCHOOL","STOP","SLEEP","SMILE","STUDY",
  "SEE YOU","SORRY FOR","SURE THING","SAFE DRIVE","SCHOOL GOOD","STOP HERE",
  "SLEEP WELL","SMILE MORE","STUDY HARD","SEE LATER","SORRY LATE","SURE OKAY",
  "SEE YOU LATER HAVE GOOD DAY","SORRY FOR BEING LATE TODAY","SURE THING NO PROBLEM",
  "SAFE DRIVE HOME CALL WHEN THERE","SCHOOL GOING WELL THIS YEAR","STOP BY MY HOUSE ANYTIME",
  "SLEEP WELL AND SWEET DREAMS","SMILE MORE YOU LOOK BEAUTIFUL","STUDY HARD FOR THE TEST",
  /*  T  */ "T","THANK","TIME","TAKE","TELL","TALK","TODAY","TOMORROW","TOGETHER","TRAVEL",
  "THANK YOU","TIME FOR","TAKE CARE","TELL ME","TALK TO","TODAY GOOD",
  "TOMORROW SEE","TOGETHER WE","TRAVEL SAFE","THANK GOD","TIME TO","TAKE TIME",
  "THANK YOU SO MUCH FOR HELP","TIME FOR LUNCH WANT TO JOIN","TAKE CARE OF YOURSELF",
  "TELL ME ABOUT YOUR DAY","TALK TO YOU LATER TONIGHT","TODAY WAS A GOOD DAY",
  "TOMORROW WE MEET FOR COFFEE","TOGETHER WE CAN DO ANYTHING","TRAVEL SAFE AND HAVE FUN",
  /*  U  */ "U","US","UP","UNDERSTAND","UNDER","UNTIL","USUALLY","UPSET","UNCLE","UNIQUE",
  "UP EARLY","UNDERSTAND YOU","UNTIL LATER","USUALLY GOOD","UPSET ABOUT","UNCLE GOOD",
  "UP EARLY FOR WORK TODAY","UNDERSTAND WHAT YOU ARE SAYING","UNTIL LATER TAKE CARE",
  "USUALLY GOOD AT REMEMBERING THINGS","UPSET ABOUT THE BAD NEWS","UNCLE VISITING US NEXT WEEK",
  /*  V  */ "V","VERY","VISIT","VIDEO","VOICE","VACATION","VOLUNTEER","VICTORY","VEGETABLE","VALUE",
  "VERY GOOD","VISIT YOU","VIDEO CALL","VOICE GOOD","VACATION TIME","VERY NICE",
  "VERY HAPPY","VISIT FAMILY","VERY WELL","VERY IMPORTANT","VISIT SOON","VERY MUCH",
  "VERY GOOD TO SEE YOU AGAIN","VISIT YOU NEXT WEEKEND","VIDEO CALL WITH FAMILY TONIGHT",
  "VACATION TIME WITH MY FAMILY","VERY HAPPY ABOUT THE NEWS","VISIT FAMILY FOR THE HOLIDAYS",
  "VERY WELL THANK YOU FOR ASKING","VERY IMPORTANT TO STAY HEALTHY",
  /*  W  */ "W","WHAT","WHEN","WHERE","WHO","WHY","WORK","WANT","WAIT","WELCOME",
  "WHAT TIME","WHEN YOU","WHERE YOU","WHO IS","WHY NOT","WORK HARD",
  "WANT TO","WAIT FOR","WELCOME HERE","WHAT HAPPEN","WORK DONE","WAIT PLEASE",
  "WELCOME HOME","WHAT YOU","WHERE GO","WORK GOOD","WANT HELP","WHO ARE",
  "WHAT TIME DO YOU GET OFF WORK","WHEN CAN WE MEET FOR LUNCH","WHERE DO YOU WANT TO GO",
  "WHO IS COMING TO THE PARTY","WHY NOT COME WITH US","WORK HARD AND STAY FOCUSED",
  "WANT TO GO OUT FOR DINNER","WAIT FOR ME I WILL BE RIGHT THERE","WELCOME TO OUR HOME",
  /*  X  */ "X","XRAY","XENIAL","XEROX","XMAS","EXTRA","EXACTLY","EXAMPLE","EXCITED","EXCUSE",
  "XRAY DONE","XMAS HAPPY","EXTRA TIME","EXACTLY RIGHT","EXAMPLE GOOD","EXCUSE ME",
  "XRAY RESULTS CAME BACK NORMAL","XMAS SHOPPING WITH THE FAMILY","EXTRA TIME TO SPEND TOGETHER",
  "EXACTLY RIGHT THAT IS WHAT I MEANT","EXCUSE ME CAN I ASK QUESTION",
  /*  Y  */ "Y","YES","YOU","YOUR","YEAR","YELLOW","YOUNG","YESTERDAY","YET","YARD",
  "YES SURE","YOU ARE","YOUR NAME","YEAR GOOD","YELLOW COLOR","YOUNG PERSON",
  "YESTERDAY GOOD","NOT YET","YARD CLEAN","YES OKAY","YOU GOOD","YOUR FAMILY",
  "YOU WELCOME","YOUR WORK","YES PLEASE","YOU HAPPY","YOUR FRIEND","YES RIGHT",
  "YES SURE I CAN HELP YOU","YOU ARE VERY KIND THANK YOU","YOUR FAMILY LOOKS VERY HAPPY",
  "YEAR HAS BEEN GOOD TO US","YESTERDAY WAS A GREAT DAY","NOT YET BUT MAYBE LATER",
  "YOU ARE WELCOME ANYTIME","YOUR WORK IS VERY IMPRESSIVE","YES PLEASE THAT SOUNDS GOOD",
  /*  Z  */ "Z","ZERO","ZONE","ZIP","ZOO","ZOOM","ZIGZAG","ZEAL","ZEBRA","ZEST",
  "ZERO PROBLEM","SAFE ZONE","ZIP CODE","ZOO VISIT","ZOOM CALL","ZEAL FOR",
  "ZERO PROBLEMS WITH THAT PLAN","SAFE ZONE FOR THE CHILDREN","ZIP CODE FOR MY ADDRESS",
  "ZOO VISIT WITH THE GRANDKIDS","ZOOM CALL WITH WORK TEAM"
];

/* ────────────────────────────────────────────────
   1.  STATE
────────────────────────────────────────────────── */
let finalSentence   = '';
let translatedText  = '';
let currentLanguage = 'en';
let lastSequence    = '';
let connectionStatus= 'listening';

/* ────────────────────────────────────────────────
   2.  TRANSLATIONS (letters + phrases)
────────────────────────────────────────────────── */
function addLetters(map,str){[...str].forEach(c=>{if(!map[c])map[c]=c});return map;}
const AZ="ABCDEFGHIJKLMNOPQRSTUVWXYZ";

const translations={
  en:addLetters({
    "HELLO":"HELLO","HOW ARE YOU":"HOW ARE YOU","THANK YOU":"THANK YOU",
    "GOOD MORNING":"GOOD MORNING","GOOD BYE":"GOOD BYE","PLEASE":"PLEASE",
    "SORRY":"SORRY","YES":"YES","NO":"NO","HELP":"HELP"},AZ),
  hi:addLetters({"A":"ए","B":"बी","C":"सी","D":"डी","E":"ई","F":"एफ़","G":"जी","H":"एच","I":"आई","J":"जे","K":"के","L":"एल","M":"एम","N":"एन","O":"ओ","P":"पी","Q":"क्यू","R":"आर","S":"एस","T":"टी","U":"यू","V":"वी","W":"डब्ल्यू","X":"एक्स","Y":"वाई","Z":"ज़ेड","HELLO":"नमस्ते","HOW ARE YOU":"आप कैसे हैं","THANK YOU":"धन्यवाद","GOOD MORNING":"सुप्रभात","GOOD BYE":"अलविदा","PLEASE":"कृपया","SORRY":"माफ करें","YES":"हाँ","NO":"नहीं","HELP":"सहायता"},AZ),
  te:addLetters({"A":"ఎ","B":"బి","C":"సి","D":"డి","E":"ఈ","F":"ఎఫ్","G":"జి","H":"హెచ్","I":"ఐ","J":"జె","K":"కె","L":"ఎల్","M":"ఎమ్","N":"ఎన్","O":"ఓ","P":"పీ","Q":"క్యూ","R":"ఆర్","S":"ఎస్","T":"టి","U":"యు","V":"వి","W":"డబ్ల్యు","X":"ఎక్స్","Y":"వై","Z":"జడ్","HELLO":"హలో","HOW ARE YOU":"మీరు ఎలా ఉన్నారు","THANK YOU":"ధన్యవాదాలు","GOOD MORNING":"శుభోదయం","GOOD BYE":"వీడ్కోలు","PLEASE":"దయచేసి","SORRY":"క్షమించండి","YES":"అవును","NO":"లేదు","HELP":"సహాయం"},AZ),
  ta:addLetters({"A":"ஏ","B":"பி","C":"சி","D":"டி","E":"ஈ","F":"எப்","G":"ஜி","H":"எச்","I":"ஐ","J":"ஜே","K":"கே","L":"எல்","M":"எம்","N":"என்","O":"ஓ","P":"பி","Q":"க்யூ","R":"ஆர்","S":"எஸ்","T":"டி","U":"யு","V":"வி","W":"டபிள்யு","X":"எக்ஸ்","Y":"வை","Z":"ஸெட்","HELLO":"வணக்கம்","HOW ARE YOU":"நீங்கள் எப்படி இருக்கிறீர்கள்","THANK YOU":"நன்றி","GOOD MORNING":"காலை வணக்கம்","GOOD BYE":"பிரியாவிடை","PLEASE":"தயவுசெய்து","SORRY":"மன்னிக்கவும்","YES":"ஆம்","NO":"இல்லை","HELP":"உதவி"},AZ),
  kn:addLetters({"A":"ಏ","B":"ಬೀ","C":"ಸಿ","D":"ಡಿ","E":"ಈ","F":"ಎಫ್","G":"ಜಿ","H":"ಎಚ್","I":"ಐ","J":"ಜೆ","K":"ಕೆ","L":"ಎಲ್","M":"ಎಮ್","N":"ಎನ್","O":"ಓ","P":"ಪಿ","Q":"ಕ್ಯೂ","R":"ಆರ್","S":"ಎಸ್","T":"ಟಿ","U":"ಯು","V":"ವಿ","W":"ಡಬ್ಲ್ಯೂ","X":"ಎಕ್ಸ್","Y":"ವೈ","Z":"ಝೆಡ್","HELLO":"ನಮಸ್ಕಾರ","HOW ARE YOU":"ನೀವು ಹೇಗಿದ್ದೀರಿ","THANK YOU":"ಧನ್ಯವಾದಗಳು","GOOD MORNING":"ಶುಭೋದಯ","GOOD BYE":"ವಿದಾಯ","PLEASE":"ದಯವಿಟ್ಟು","SORRY":"ಕ್ಷಮಿಸಿ","YES":"ಹೌದು","NO":"ಇಲ್ಲ","HELP":"ಸಹಾಯ"},AZ),
  ml:addLetters({"A":"എ","B":"ബി","C":"സി","D":"ഡി","E":"ഈ","F":"എഫ്","G":"ജി","H":"എച്ച്","I":"ഐ","J":"ജെ","K":"കെ","L":"എൽ","M":"എം","N":"എൻ","O":"ഓ","P":"പി","Q":"ക്യൂ","R":"ആർ","S":"എസ്","T":"ടി","U":"യു","V":"വി","W":"ഡബ്ല്യു","X":"എക്സ്","Y":"വൈ","Z":"സെഡ്","HELLO":"നമസ്കാരം","HOW ARE YOU":"നിങ്ങൾ എങ്ങനെയുണ്ട്","THANK YOU":"നന്ദി","GOOD MORNING":"സുപ്രഭാതം","GOOD BYE":"വിട","PLEASE":"ദയവായി","SORRY":"ക്ഷമിക്കണം","YES":"അതെ","NO":"ഇല്ല","HELP":"സഹായം"},AZ),
  mr:addLetters({"A":"ए","B":"बी","C":"सी","D":"डी","E":"ई","F":"एफ","G":"जी","H":"एच","I":"आय","J":"जे","K":"के","L":"एल","M":"एम","N":"एन","O":"ओ","P":"पी","Q":"क्यू","R":"आर","S":"एस","T":"टी","U":"यू","V":"वी","W":"डब्ल्यू","X":"एक्स","Y":"वाई","Z":"झेड","HELLO":"नमस्कार","HOW ARE YOU":"तुम्ही कसे आहात","THANK YOU":"धन्यवाद","GOOD MORNING":"शुभ सकाळ","GOOD BYE":"निरोप","PLEASE":"कृपया","SORRY":"माफ करा","YES":"होय","NO":"नाही","HELP":"मदत"},AZ),
  gu:addLetters({"A":"એ","B":"બી","C":"સી","D":"ડી","E":"ઈ","F":"એફ","G":"જી","H":"એચ","I":"આઈ","J":"જે","K":"કે","L":"એલ","M":"એમ","N":"એન","O":"ઓ","P":"પી","Q":"ક્યુ","R":"આર","S":"એસ","T":"ટી","U":"યુ","V":"વી","W":"ડબ્લ્યુ","X":"એક્સ","Y":"વાય","Z":"ઝેડ","HELLO":"નમસ્તે","HOW ARE YOU":"તમે કેમ છો","THANK YOU":"આભાર","GOOD MORNING":"સુપ્રભાત","GOOD BYE":"ગુડ બાય","PLEASE":"કૃપા કરીને","SORRY":"માફ કરશો","YES":"હા","NO":"ના","HELP":"મદદ"},AZ),
  bn:addLetters({"A":"এ","B":"বি","C":"সি","D":"ডি","E":"ই","F":"এফ","G":"জি","H":"এইচ","I":"আই","J":"জে","K":"কে","L":"এল","M":"এম","N":"এন","O":"ও","P":"পি","Q":"কিউ","R":"আর","S":"এস","T":"টি","U":"ইউ","V":"ভি","W":"ডাব্লিউ","X":"এক্স","Y":"ওয়াই","Z":"জেড","HELLO":"নমস্কার","HOW ARE YOU":"আপনি কেমন আছেন","THANK YOU":"ধন্যবাদ","GOOD MORNING":"শুভ সকাল","GOOD BYE":"বিদায়","PLEASE":"অনুগ্রহ করে","SORRY":"দুঃখিত","YES":"হ্যাঁ","NO":"না","HELP":"সাহায্য"},AZ),
  pa:addLetters({"A":"ਏ","B":"ਬੀ","C":"ਸੀ","D":"ਡੀ","E":"ਈ","F":"ਐਫ਼","G":"ਜੀ","H":"ਏਚ","I":"ਆਈ","J":"ਜੇ","K":"ਕੇ","L":"ਐੱਲ","M":"ਐੱਮ","N":"ਐੱਨ","O":"ਓ","P":"ਪੀ","Q":"ਕਿਊ","R":"ਆਰ","S":"ਐੱਸ","T":"ਟੀ","U":"ਯੂ","V":"ਵੀ","W":"ਡਬਲਯੂ","X":"ਐਕਸ","Y":"ਵਾਈ","Z":"ਜ਼ੈੱਡ","HELLO":"ਸਤ ਸ੍ਰੀ ਅਕਾਲ","HOW ARE YOU":"ਤੁਸੀਂ ਕਿਵੇਂ ਹੋ","THANK YOU":"ਧੰਨਵਾਦ","GOOD MORNING":"ਸ਼ੁਭ ਸਵੇਰ","GOOD BYE":"ਅਲਵਿਦਾ","PLEASE":"ਕਿਰਪਾ ਕਰਕੇ","SORRY":"ਮਾਫ਼ ਕਰਨਾ","YES":"ਹਾਂ","NO":"ਨਹੀਂ","HELP":"ਮਦਦ"},AZ)
};

/* ────────────────────────────────────────────────
   3.  UI HANDLERS
────────────────────────────────────────────────── */
document.querySelector('#btn').onclick=()=>document.querySelector('.sidebar').classList.toggle('active');
document.querySelectorAll('.lang-btn').forEach(btn=>{
  btn.onclick=()=>{document.querySelectorAll('.lang-btn').forEach(x=>x.classList.remove('active'));
                   btn.classList.add('active'); currentLanguage=btn.dataset.lang;
                   if(finalSentence) translateText(finalSentence);}
});

/* 4. Translation */
async function translateText(txt){
  const out=document.getElementById('translated-sentence');
  if(currentLanguage==='en'){translatedText=txt;out.textContent=txt;return;}
  if(translations[currentLanguage]?.[txt]){translatedText=translations[currentLanguage][txt];out.textContent=translatedText;return;}
  out.innerHTML='<span class="loading">Translating…</span>';
  try{
    const url=`https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=${currentLanguage}&dt=t&q=${encodeURIComponent(txt)}`;
    const data=await(await fetch(url)).json(); translatedText=data[0][0][0]||txt; out.textContent=translatedText;
  }catch(e){console.warn(e);translatedText=`[${txt}]`;out.textContent=translatedText;}
}

/* 5. Text‑to‑Speech */
const voiceTag ={en:'en-US',hi:'hi-IN'};              // browser voices usually present
const googleTag={en:'en',hi:'hi',te:'te',ta:'ta',kn:'kn',ml:'ml',mr:'mr',gu:'gu',bn:'bn',pa:'pa'};

const ttsAudio=new Audio();            // NO crossOrigin → avoids CORS block

function browserSpeak(txt,lang){
  if(!('speechSynthesis'in window))return false;
  const synth=speechSynthesis; synth.cancel();
  const voice=synth.getVoices().find(v=>v.lang===voiceTag[lang]);
  if(!voice) return false;
  const u=new SpeechSynthesisUtterance(txt); u.voice=voice; u.rate=.9; u.pitch=1; synth.speak(u); return true;
}

function playGoogleTTS(txt,lang){
  return new Promise((res,rej)=>{
    ttsAudio.src=`https://translate.google.com/translate_tts?ie=UTF-8&client=tw-ob&tl=${googleTag[lang]||'en'}&q=${encodeURIComponent(txt)}`;
    ttsAudio.oncanplaythrough=()=>ttsAudio.play().then(res).catch(rej);
    ttsAudio.onerror=rej;
  });
}

async function speak(txt,lang='en'){
  if(!txt.trim())return;
  if(browserSpeak(txt,lang))return;            // works for English/Hindi if voice exists
  try{await playGoogleTTS(txt,lang);}catch(e){console.warn('TTS failed',e);}
}

/* 6. Audio buttons */
document.getElementById('play-original').onclick=()=>finalSentence&&speak(finalSentence,'en');
document.getElementById('play-translation').onclick=()=>{
  const txt=document.getElementById('translated-sentence').textContent.trim();
  if(!txt||txt.includes('Translating'))return; speak(txt,currentLanguage);
};

/* 7. Suggestions + sequence polling (unchanged) */
function updateSuggestions(seq){
  const key=seq.replace(/ /g,'').toUpperCase(); const box=document.getElementById('suggestions'); box.innerHTML='';
  suggestionsData.filter(s=>s.replace(/ /g,'').startsWith(key)).slice(0,5).forEach(w=>{
    const d=document.createElement('div'); d.className='suggestion'; d.textContent=w;
    d.onclick=()=>{finalSentence=w;document.getElementById('original-sentence').textContent=w;translateText(w);speak(w,'en');};
    box.appendChild(d);
  });
}
function fetchSeq(){
  fetch('/sequence.txt?ts='+Date.now())
    .then(r=>r.ok?r.text():Promise.reject(r.status))
    .then(t=>{const c=t.trim();document.getElementById('signed-seq').textContent=c||'[Listening…]';
              if(c&&c.toUpperCase()!==lastSequence){updateSuggestions(c);lastSequence=c.toUpperCase();}})
    .catch(()=>document.getElementById('signed-seq').textContent='[Error]');}
window.onload=()=>{fetchSeq();setInterval(fetchSeq,1000);};
</script>

</body>
</html>
"""

# ────────────────────────── MJPEG streaming ───────────────────────────
latest_frame = None  # holds newest JPEG for the MJPEG endpoint

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        parsed_path = urlparse(self.path)
        path = parsed_path.path
        
        print(f"Request for: {path}")  # Debug print
        
        # Serve the main HTML page
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode())
            return

        # Serve sequence.txt content
        elif path.startswith("/sequence.txt"):
            try:
                # Create sequence.txt if it doesn't exist
                if not os.path.exists("sequence.txt"):
                    with open("sequence.txt", "w", encoding="utf-8") as f:
                        f.write("")
                
                with open("sequence.txt", "r", encoding="utf-8") as f:
                    data = f.read()
                
                self.send_response(200)
                self.send_header("Content-type", "text/plain; charset=utf-8")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET")
                self.send_header("Access-Control-Allow-Headers", "Content-Type")
                self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.end_headers()
                self.wfile.write(data.encode('utf-8'))
                #print(f"Served sequence.txt content: '{data}'")
                
            except Exception as e:
                print(f"Error serving sequence.txt: {e}")
                self.send_error(500, f"Error reading sequence.txt: {e}")
            return
        
        # Serve video feed
        elif path == "/video_feed":
            self.send_response(200)
            self.send_header("Content-type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.end_headers()
            
            try:
                while True:
                    global latest_frame
                    if latest_frame:
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + latest_frame + b"\r\n"
                        )
                        self.wfile.flush()
                    else:
                        # Send a placeholder frame if no camera data available
                        placeholder = b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00\x00\xff\xdb\x00C\x00\x08\x06\x06\x07\x06\x05\x08\x07\x07\x07\t\t\x08\n\x0c\x14\r\x0c\x0b\x0b\x0c\x19\x12\x13\x0f\x14\x1d\x1a\x1f\x1e\x1d\x1a\x1c\x1c $.\' ",#\x1c\x1c(7),01444\x1f\'9=82<.342\xff\xc0\x00\x11\x08\x01,\x01\x90\x03\x01"\x00\x02\x11\x01\x03\x11\x01\xff\xc4\x00\x14\x00\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08\xff\xc4\x00\x14\x10\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\xff\xda\x00\x0c\x03\x01\x00\x02\x11\x03\x11\x00\x3f\x00\xaa\xff\xd9'
                        self.wfile.write(
                            b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                            + placeholder + b"\r\n"
                        )
                        self.wfile.flush()
                    
                    time.sleep(0.033)  # ~30 FPS
                    
            except (ConnectionResetError, BrokenPipeError, OSError):
                print("Video feed client disconnected")
            except Exception as e:
                print(f"Video feed error: {e}")
            
            return
        
        # Handle other static files
        else:
            super().do_GET()
    
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        # Override to reduce server log spam
        if not any(x in args[0] for x in ["/video_feed", "/sequence.txt"]):
            super().log_message(format, *args)

def start_server():
    """Start the HTTP server in a separate thread"""
    try:
        # Use ThreadingTCPServer for better concurrency
        with socketserver.ThreadingTCPServer(("", 8000), MyHTTPRequestHandler) as httpd:
            httpd.allow_reuse_address = True
            print("HTTP Server starting on http://localhost:8000")
            print("Access your application at: http://localhost:8000/")
            
            # Create sequence.txt with initial content if it doesn't exist
            if not os.path.exists("sequence.txt"):
                with open("sequence.txt", "w") as f:
                    f.write("")
                print("Created empty sequence.txt file")
            
            httpd.serve_forever()
    except Exception as e:
        print(f"Server error: {e}")

# ─────────────────────── Load TensorFlow model ────────────────────────
def load_model():
    """Load the TensorFlow model and labels"""
    try:
        # Check if model files exist
        if not os.path.exists("logs/trained_labels.txt"):
            print("Warning: logs/trained_labels.txt not found")
            return None, None
        
        if not os.path.exists("logs/trained_graph.pb"):
            print("Warning: logs/trained_graph.pb not found")
            return None, None
        
        label_lines = [l.rstrip() for l in tf.io.gfile.GFile("logs/trained_labels.txt")]
        
        with tf.io.gfile.GFile("logs/trained_graph.pb", "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name="")
        
        print(f"Model loaded successfully with {len(label_lines)} labels")
        return label_lines, True
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None

def predict(img_bytes, sess, softmax, label_lines):
    """Predict sign language from image"""
    try:
        probs = sess.run(softmax, {"DecodeJpeg/contents:0": img_bytes})[0]
        idx = int(probs.argmax())
        return label_lines[idx], float(probs[idx])
    except Exception as e:
        print(f"Prediction error: {e}")
        return "nothing", 0.0

# ─────────────────────────── Main loop ────────────────────────────────
def main():
    # Start server in background thread
    server_thread = threading.Thread(target=start_server, daemon=True)
    server_thread.start()
    
    # Give server a moment to start
    time.sleep(1)
    
    # Load model
    label_lines, model_loaded = load_model()
    
    if not model_loaded:
        print("Model not loaded - running in demo mode")
        print("The web interface will still work, but sign recognition is disabled")
        
        # Keep server running even without model
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nShutting down...")
            return
    
    # Initialize TensorFlow session
    with tf.compat.v1.Session() as sess:
        try:
            softmax = sess.graph.get_tensor_by_name("final_result:0")
        except Exception as e:
            print(f"Error getting tensor: {e}")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Warning: Cannot open webcam - running in server-only mode")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
                return
        
        print("Webcam initialized successfully")
        print("Press ESC in the OpenCV window to quit")
        
        # Main processing loop
        sequence = mem = ""
        i = consecutive = 0
        res = "nothing"
        score = 0.0
        
        try:
            while True:
                ok, img = cap.read()
                if not ok:
                    print("Failed to read from webcam")
                    break
                
                img = cv2.flip(img, 1)
                
                # ROI for sign detection
                x1, y1, x2, y2 = 100, 100, 300, 300
                roi = img[y1:y2, x1:x2]
                
                i += 1
                if i == 4:  # predict every 4th frame
                    try:
                        res, score = predict(cv2.imencode(".jpg", roi)[1].tobytes(),
                                           sess, softmax, label_lines)
                    except Exception as e:
                        print(f"Prediction failed: {e}")
                        res, score = "nothing", 0.0
                    
                    i = 0
                    consecutive = consecutive + 1 if mem == res else 0
                    
                    if consecutive == 2 and res != "nothing":
                        if res == "space":
                            sequence += " "
                        elif res == "del":
                            sequence = sequence[:-1]
                        else:
                            sequence += res
                        consecutive = 0
                        
                        # Write to sequence file
                        try:
                            with open("sequence.txt", "w", encoding="utf-8") as f:
                                f.write(sequence)
                        except Exception as e:
                            print(f"Error writing sequence.txt: {e}")
                    
                    mem = res
                
                # Draw overlays
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(img, f"{res.upper()} ({score:.2f})", (100, 400),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Update MJPEG frame for web streaming
                global latest_frame
                latest_frame = cv2.imencode(".jpg", img)[1].tobytes()
                
                # Display local windows
                seq_img = np.zeros((200, 1200, 3), np.uint8)
                cv2.putText(seq_img, sequence.upper(), (30, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
                
                cv2.imshow("Sign Language Recognizer", img)
                cv2.imshow("Sequence", seq_img)
                
                if cv2.waitKey(1) == 27:  # ESC key
                    break
                    
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            print(f"Main loop error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("Cleanup completed")

if __name__ == "__main__":
    main()