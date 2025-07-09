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
  <style>
    body{font-family:Arial,sans-serif;background:#121212;color:#fff;text-align:center;padding:30px}
    h1{font-size:2em;margin-bottom:20px}
    #sequence-box,#final-box{margin:20px auto;padding:20px;width:80%;border:2px solid #555;border-radius:10px;background:#1e1e1e;font-size:1.5em}
    .suggestion{display:inline-block;margin:10px;padding:10px 20px;background:#444;border-radius:5px;cursor:pointer;transition:.2s}
    .suggestion:hover{background:#666}
    #suggestions{margin-top:20px}
    #webcam-container img{border:2px solid #555;border-radius:10px;width:400px;height:300px}
    .status-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        border-radius: 50%;
        margin-left: 10px;
    }
    .status-listening { background-color: #ffa500; }
    .status-connected { background-color: #00ff00; }
    .status-error { background-color: #ff0000; }
  </style>
</head>
<body>
    <h1>Sign Language Suggestions</h1>
    
    <div id="webcam-container">
        <!-- Stream served by Python at /video_feed -->
        <img src="/video_feed" alt="Webcam stream" onerror="this.src='data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj48cmVjdCB3aWR0aD0iNDAwIiBoZWlnaHQ9IjMwMCIgZmlsbD0iI2Y4ZjlmYSIvPjx0ZXh0IHg9IjIwMCIgeT0iMTUwIiBmb250LWZhbWlseT0iQXJpYWwsIHNhbnMtc2VyaWYiIGZvbnQtc2l6ZT0iMTQiIGZpbGw9IiM2Yzc1N2QiIHRleHQtYW5jaG9yPSJtaWRkbGUiPkNhbWVyYSBTdHJlYW0gVW5hdmFpbGFibGU8L3RleHQ+PC9zdmc+'">
    </div>
    
    <div id="sequence-box">
        Signed Sequence: <span id="signed-seq">[Listening...]</span>
        <span id="connection-status" class="status-indicator status-listening"></span>
    </div>
    
    <div id="suggestions"></div>
    
    <div id="final-box">
        Final Sentence: <span id="final-sentence">[Click a suggestion]</span>
    </div>

    <script>
        const suggestionsData = [
            // A - Complete sentences and conversational phrases
            "A", "AM", "AND", "ARE", "ALL", "ABOUT", "AFTER", "AGAIN", "ASK", "AWESOME",
            "AM FINE", "AM GOOD", "AM HAPPY", "AND YOU", "ARE YOU", "ASK YOU",
            "ARE YOU FREE TODAY", "AM GOING TO WORK", "AND HOW ABOUT YOU", "AWESOME TO MEET YOU",
            "ALL IS WELL", "AFTER WORK I GO HOME", "AGAIN NICE TO SEE YOU", "ASK ME ANYTHING",
            
            // B - Conversational flow
            "B", "BE", "BY", "BUT", "BYE", "BECAUSE", "BEFORE", "BEAUTIFUL", "BEST", "BUSY",
            "BE GOOD", "BYE BYE", "BE HAPPY", "BE SAFE", "BECAUSE I", "BEFORE YOU",
            "BYE SEE YOU LATER", "BE CAREFUL DRIVING HOME", "BECAUSE I HAVE WORK TODAY", 
            "BEAUTIFUL DAY TODAY", "BEST OF LUCK TO YOU", "BUSY WITH FAMILY RIGHT NOW",
            
            // C - Questions and responses
            "C", "CAN", "COME", "COOL", "CALL", "CHANGE", "CHOOSE", "CONGRATULATIONS", "CLEAN", "CAREFUL",
            "CAN YOU", "COME HERE", "CALL ME", "CAN I", "COME ON", "COOL DOWN",
            "CAN YOU HELP ME PLEASE", "COME VISIT US SOMETIME", "CALL ME WHEN YOU ARRIVE",
            "CAN I ASK YOU SOMETHING", "CONGRATULATIONS ON YOUR SUCCESS", "COOL TO MEET YOU TODAY",
            
            // D - Daily conversation starters
            "D", "DO", "DAY", "DONE", "DRIVE", "DRINK", "DIFFERENT", "DOCTOR", "DANCE", "DIFFICULT",
            "DO YOU", "GOOD DAY", "DONE WORK", "DRIVE SAFE", "DO IT", "DAY GOOD",
            "DO YOU HAVE TIME TO TALK", "DONE WITH WORK FOR TODAY", "DRIVE SAFE AND CALL WHEN HOME",
            "DAY WENT REALLY WELL", "DOCTOR APPOINTMENT WENT GOOD", "DIFFERENT FROM WHAT I EXPECTED",
            
            // E - Emotions and expressions
            "E", "EAT", "EVERY", "EXCELLENT", "EXCITED", "EMERGENCY", "EASY", "EARLY", "EVENING", "EXPLAIN",
            "EAT FOOD", "EVERY DAY", "EXCITED TO", "EASY TO", "EVENING GOOD", "EXPLAIN TO",
            "EAT DINNER WITH FAMILY TONIGHT", "EVERY DAY IS A BLESSING", "EXCITED TO SEE YOU SOON",
            "EXCELLENT WORK YOU DID", "EARLY MORNING WORKOUT FEELS GOOD", "EVENING PLANS WITH FRIENDS",
            
            // F - Family and feelings
            "F", "FINE", "FROM", "FRIEND", "FAMILY", "FOOD", "FUN", "FEEL", "FINISH", "FORGET",
            "FINE THANK", "FROM WHERE", "FRIEND GOOD", "FAMILY GOOD", "FEEL GOOD", "FINISH WORK",
            "FINE THANK YOU FOR ASKING", "FAMILY IS DOING WELL", "FRIEND COMING TO VISIT TODAY",
            "FOOD WAS DELICIOUS THANK YOU", "FUN TIME AT THE PARTY", "FEEL MUCH BETTER NOW",
            
            // G - Greetings and goodbyes
            "G", "GO", "GOOD", "GET", "GIVE", "GREAT", "GIRL", "GAME", "GREEN", "GROW",
            "GO HOME", "GOOD MORNING", "GOOD NIGHT", "GET UP", "GIVE ME", "GREAT JOB",
            "GOOD BYE", "GOOD DAY", "GOOD LUCK", "GO AHEAD", "GET READY", "GOOD WORK",
            "GOOD MORNING HOW ARE YOU", "GOOD NIGHT SLEEP WELL", "GREAT TO SEE YOU AGAIN",
            "GO HOME AND REST WELL", "GET READY FOR THE MEETING", "GOOD LUCK WITH YOUR INTERVIEW",
            
            // H - Common conversation starters
            "H", "HI", "HELLO", "HOW", "HELP", "HAPPY", "HOME", "HOSPITAL", "HEAR", "HOPE",
            "HI THERE", "HELLO FRIEND", "HOW ARE", "HOW ARE YOU", "HELP ME", "HAPPY TO",
            "HELLO I", "HELLO I AM", "HELLO HOW", "HELP YOU", "HOME SAFE", "HOPE YOU",
            "HI THERE HOW ARE YOU DOING", "HELLO NICE TO MEET YOU", "HOW WAS YOUR DAY TODAY",
            "HELP ME WITH THIS PLEASE", "HAPPY TO SEE YOU AGAIN", "HOME SAFE AND SOUND",
            "HOPE YOU HAVE A GREAT DAY", "HOW IS YOUR FAMILY DOING", "HELLO MY NAME IS",
            
            // I - Personal expressions
            "I", "I AM", "IS", "IN", "IT", "IF", "INTERESTING", "IMPORTANT", "INVITE", "INSIDE",
            "I AM FINE", "I AM GOOD", "I AM HAPPY", "I LOVE", "I NEED", "I WANT",
            "I CAN", "I WILL", "I HOPE", "I THANK", "I UNDERSTAND", "I LIKE",
            "I AM FINE THANK YOU", "I AM GOING TO THE STORE", "I LOVE SPENDING TIME WITH FAMILY",
            "I NEED YOUR HELP WITH SOMETHING", "I WANT TO LEARN MORE SIGN", "I CAN MEET YOU TOMORROW",
            "I WILL CALL YOU LATER", "I HOPE YOU FEEL BETTER SOON", "I UNDERSTAND WHAT YOU MEAN",
            
            // J - General conversation
            "J", "JOB", "JUST", "JUMP", "JOURNEY", "JOIN", "JUICE", "JACKET", "JEALOUS", "JOKE",
            "GOOD JOB", "JUST FINE", "JOIN US", "JOURNEY SAFE", "JUICE GOOD", "JUST OKAY",
            "JOB INTERVIEW WENT VERY WELL", "JUST FINISHED EATING LUNCH", "JOIN US FOR DINNER TONIGHT",
            "JUST WANTED TO SAY HELLO", "JOURNEY TO WORK TAKES LONG TIME",
            
            // K - Kindness and knowledge
            "K", "KNOW", "KEEP", "KIND", "KITCHEN", "KISS", "KICK", "KILL", "KEY", "KING",
            "KNOW YOU", "KEEP SAFE", "KIND PERSON", "KEEP GOING", "KNOW WHAT", "KEEP CALM",
            "KNOW YOU ARE BUSY RIGHT NOW", "KEEP SAFE ON YOUR WAY HOME", "KIND OF YOU TO HELP ME",
            "KEEP GOING YOU ARE DOING GREAT", "KNOW WHAT YOU MEAN EXACTLY",
            
            // L - Love and life
            "L", "LOVE", "LIKE", "LOOK", "LEARN", "LEAVE", "LATE", "LUNCH", "LIFE", "LISTEN",
            "LOVE YOU", "LIKE YOU", "LOOK AT", "LEARN MORE", "LEAVE NOW", "LUNCH TIME",
            "LIFE GOOD", "LISTEN TO", "LOOK GOOD", "LOVE FAMILY", "LIKE IT", "LEARN SIGN",
            "LOVE YOU TOO VERY MUCH", "LIKE TO SPEND TIME TOGETHER", "LOOK FORWARD TO SEEING YOU",
            "LEARN SOMETHING NEW EVERY DAY", "LATE FOR THE MEETING SORRY", "LUNCH WAS DELICIOUS TODAY",
            "LIFE IS GOOD THANK GOD", "LISTEN TO WHAT YOU ARE SAYING",
            
            // M - Meeting and messaging
            "M", "ME", "MY", "MORE", "MAKE", "MEET", "MORNING", "MONEY", "MOTHER", "MOVE",
            "MEET YOU", "MY FAMILY", "MY NAME", "MORE TIME", "MAKE SURE", "MORNING GOOD",
            "MOTHER GOOD", "MOVE ON", "MY FRIEND", "MAKE HAPPY", "MORE WORK", "MEET FRIEND",
            "MEET YOU AT THE RESTAURANT", "MY FAMILY IS DOING WELL", "MY NAME IS NICE TO MEET",
            "MORE TIME TO TALK LATER", "MAKE SURE TO DRIVE SAFE", "MORNING WORKOUT FELT GREAT",
            "MOTHER CALLED ME THIS MORNING", "MOVE FORWARD WITH THE PLAN",
            
            // N - Negative and neutral responses
            "N", "NO", "NOW", "NEED", "NICE", "NAME", "NEW", "NIGHT", "NEVER", "NEXT",
            "NO PROBLEM", "NOW TIME", "NEED HELP", "NICE TO", "NAME IS", "NEW FRIEND",
            "GOOD NIGHT", "NEVER MIND", "NEXT TIME", "NICE DAY", "NEED YOU", "NOW GO",
            "NO PROBLEM HAPPY TO HELP", "NOW IS GOOD TIME TO TALK", "NEED TO GO HOME SOON",
            "NICE TO MEET YOU TODAY", "NAME IS WHAT AGAIN SORRY", "NEW JOB STARTING NEXT WEEK",
            "GOOD NIGHT SLEEP WELL", "NEVER MIND IT IS OKAY", "NEXT TIME WE MEET FOR COFFEE",
            
            // O - Offers and opinions
            "O", "OK", "OKAY", "OF", "ON", "OR", "OPEN", "OFFICE", "OVER", "OLD",
            "OK GOOD", "OKAY FINE", "OPEN DOOR", "OFFICE WORK", "OVER THERE", "OLD FRIEND",
            "OK GOOD TO HEAR FROM YOU", "OKAY FINE WITH ME", "OPEN THE DOOR PLEASE",
            "OFFICE WORK KEEPS ME BUSY", "OVER THERE IS MY FRIEND", "OLD FRIEND FROM SCHOOL DAYS",
            
            // P - Polite expressions
            "P", "PLEASE", "PEOPLE", "PROBLEM", "PERFECT", "PLAY", "PHONE", "PICK", "PIZZA", "PRAY",
            "PLEASE HELP", "NO PROBLEM", "PEOPLE GOOD", "PERFECT TIME", "PLAY GAME", "PHONE CALL",
            "PICK UP", "PIZZA GOOD", "PLEASE COME", "PLEASE WAIT", "PLAY WITH", "PEOPLE NICE",
            "PLEASE HELP ME WITH THIS", "NO PROBLEM AT ALL", "PEOPLE ARE VERY NICE HERE",
            "PERFECT TIME TO MEET TODAY", "PLAY GAMES WITH THE KIDS", "PHONE CALL FROM MY MOTHER",
            "PICK UP GROCERIES ON WAY HOME", "PIZZA FOR DINNER SOUNDS GOOD",
            
            // Q - Questions
            "Q", "QUESTION", "QUICK", "QUIET", "QUIT", "QUALITY", "QUITE", "QUEEN", "QUOTE", "QUIZ",
            "QUESTION FOR", "QUICK TIME", "QUIET PLEASE", "QUIT WORK", "QUALITY GOOD", "QUITE GOOD",
            "QUESTION FOR YOU IF YOU HAVE TIME", "QUICK MEETING BEFORE WE GO", "QUIET PLEASE BABY SLEEPING",
            "QUIT WORK EARLY TODAY", "QUALITY TIME WITH FAMILY",
            
            // R - Responses and requests
            "R", "RIGHT", "READY", "REMEMBER", "REALLY", "RUN", "READ", "ROOM", "RELAX", "RESPECT",
            "RIGHT NOW", "READY TO", "REMEMBER ME", "REALLY GOOD", "RUN FAST", "READ BOOK",
            "ROOM CLEAN", "RELAX TIME", "RESPECT YOU", "RIGHT HERE", "READY GO", "REALLY NICE",
            "RIGHT NOW IS GOOD TIME", "READY TO GO WHEN YOU ARE", "REMEMBER TO CALL ME LATER",
            "REALLY GOOD TO SEE YOU", "RUN IN THE PARK THIS MORNING", "READ INTERESTING BOOK LATELY",
            "ROOM LOOKS VERY CLEAN TODAY", "RELAX AND ENJOY YOUR WEEKEND",
            
            // S - Social interactions
            "S", "SEE", "SORRY", "SURE", "SAFE", "SCHOOL", "STOP", "SLEEP", "SMILE", "STUDY",
            "SEE YOU", "SORRY FOR", "SURE THING", "SAFE DRIVE", "SCHOOL GOOD", "STOP HERE",
            "SLEEP WELL", "SMILE MORE", "STUDY HARD", "SEE LATER", "SORRY LATE", "SURE OKAY",
            "SEE YOU LATER HAVE GOOD DAY", "SORRY FOR BEING LATE TODAY", "SURE THING NO PROBLEM",
            "SAFE DRIVE HOME CALL WHEN THERE", "SCHOOL GOING WELL THIS YEAR", "STOP BY MY HOUSE ANYTIME",
            "SLEEP WELL AND SWEET DREAMS", "SMILE MORE YOU LOOK BEAUTIFUL", "STUDY HARD FOR THE TEST",
            
            // T - Time and thanks
            "T", "THANK", "TIME", "TAKE", "TELL", "TALK", "TODAY", "TOMORROW", "TOGETHER", "TRAVEL",
            "THANK YOU", "TIME FOR", "TAKE CARE", "TELL ME", "TALK TO", "TODAY GOOD",
            "TOMORROW SEE", "TOGETHER WE", "TRAVEL SAFE", "THANK GOD", "TIME TO", "TAKE TIME",
            "THANK YOU SO MUCH FOR HELP", "TIME FOR LUNCH WANT TO JOIN", "TAKE CARE OF YOURSELF",
            "TELL ME ABOUT YOUR DAY", "TALK TO YOU LATER TONIGHT", "TODAY WAS A GOOD DAY",
            "TOMORROW WE MEET FOR COFFEE", "TOGETHER WE CAN DO ANYTHING", "TRAVEL SAFE AND HAVE FUN",
            
            // U - Understanding and updates
            "U", "US", "UP", "UNDERSTAND", "UNDER", "UNTIL", "USUALLY", "UPSET", "UNCLE", "UNIQUE",
            "UP EARLY", "UNDERSTAND YOU", "UNTIL LATER", "USUALLY GOOD", "UPSET ABOUT", "UNCLE GOOD",
            "UP EARLY FOR WORK TODAY", "UNDERSTAND WHAT YOU ARE SAYING", "UNTIL LATER TAKE CARE",
            "USUALLY GOOD AT REMEMBERING THINGS", "UPSET ABOUT THE BAD NEWS", "UNCLE VISITING US NEXT WEEK",
            
            // V - Visits and values
            "V", "VERY", "VISIT", "VIDEO", "VOICE", "VACATION", "VOLUNTEER", "VICTORY", "VEGETABLE", "VALUE",
            "VERY GOOD", "VISIT YOU", "VIDEO CALL", "VOICE GOOD", "VACATION TIME", "VERY NICE",
            "VERY HAPPY", "VISIT FAMILY", "VERY WELL", "VERY IMPORTANT", "VISIT SOON", "VERY MUCH",
            "VERY GOOD TO SEE YOU AGAIN", "VISIT YOU NEXT WEEKEND", "VIDEO CALL WITH FAMILY TONIGHT",
            "VACATION TIME WITH MY FAMILY", "VERY HAPPY ABOUT THE NEWS", "VISIT FAMILY FOR THE HOLIDAYS",
            "VERY WELL THANK YOU FOR ASKING", "VERY IMPORTANT TO STAY HEALTHY",
            
            // W - Work and weekly activities
            "W", "WHAT", "WHEN", "WHERE", "WHO", "WHY", "WORK", "WANT", "WAIT", "WELCOME",
            "WHAT TIME", "WHEN YOU", "WHERE YOU", "WHO IS", "WHY NOT", "WORK HARD",
            "WANT TO", "WAIT FOR", "WELCOME HERE", "WHAT HAPPEN", "WORK DONE", "WAIT PLEASE",
            "WELCOME HOME", "WHAT YOU", "WHERE GO", "WORK GOOD", "WANT HELP", "WHO ARE",
            "WHAT TIME DO YOU GET OFF WORK", "WHEN CAN WE MEET FOR LUNCH", "WHERE DO YOU WANT TO GO",
            "WHO IS COMING TO THE PARTY", "WHY NOT COME WITH US", "WORK HARD AND STAY FOCUSED",
            "WANT TO GO OUT FOR DINNER", "WAIT FOR ME I WILL BE RIGHT THERE", "WELCOME TO OUR HOME",
            
            // X - Extra expressions
            "X", "XRAY", "XENIAL", "XEROX", "XMAS", "EXTRA", "EXACTLY", "EXAMPLE", "EXCITED", "EXCUSE",
            "XRAY DONE", "XMAS HAPPY", "EXTRA TIME", "EXACTLY RIGHT", "EXAMPLE GOOD", "EXCUSE ME",
            "XRAY RESULTS CAME BACK NORMAL", "XMAS SHOPPING WITH THE FAMILY", "EXTRA TIME TO SPEND TOGETHER",
            "EXACTLY RIGHT THAT IS WHAT I MEANT", "EXCUSE ME CAN I ASK QUESTION",
            
            // Y - Yes responses and you statements
            "Y", "YES", "YOU", "YOUR", "YEAR", "YELLOW", "YOUNG", "YESTERDAY", "YET", "YARD",
            "YES SURE", "YOU ARE", "YOUR NAME", "YEAR GOOD", "YELLOW COLOR", "YOUNG PERSON",
            "YESTERDAY GOOD", "NOT YET", "YARD CLEAN", "YES OKAY", "YOU GOOD", "YOUR FAMILY",
            "YOU WELCOME", "YOUR WORK", "YES PLEASE", "YOU HAPPY", "YOUR FRIEND", "YES RIGHT",
            "YES SURE I CAN HELP YOU", "YOU ARE VERY KIND THANK YOU", "YOUR FAMILY LOOKS VERY HAPPY",
            "YEAR HAS BEEN GOOD TO US", "YESTERDAY WAS A GREAT DAY", "NOT YET BUT MAYBE LATER",
            "YOU ARE WELCOME ANYTIME", "YOUR WORK IS VERY IMPRESSIVE", "YES PLEASE THAT SOUNDS GOOD",
            
            // Z - Zero problems and zones
            "Z", "ZERO", "ZONE", "ZIP", "ZOO", "ZOOM", "ZIGZAG", "ZEAL", "ZEBRA", "ZEST",
            "ZERO PROBLEM", "SAFE ZONE", "ZIP CODE", "ZOO VISIT", "ZOOM CALL", "ZEAL FOR",
            "ZERO PROBLEMS WITH THAT PLAN", "SAFE ZONE FOR THE CHILDREN", "ZIP CODE FOR MY ADDRESS",
            "ZOO VISIT WITH THE GRANDKIDS", "ZOOM CALL WITH WORK TEAM"
        ];
        
        let finalSentence = '';
        let lastSequence = '';
        let connectionStatus = 'listening';
        
        function updateConnectionStatus(status) {
            const indicator = document.getElementById('connection-status');
            indicator.className = `status-indicator status-${status}`;
            connectionStatus = status;
        }
        
        function speak(text) {
            if ('speechSynthesis' in window) {
                const utterance = new SpeechSynthesisUtterance(text);
                utterance.lang = 'en-US';
                utterance.rate = 1;
                utterance.pitch = 1;
                speechSynthesis.speak(utterance);
            }
        }
        
        function updateSuggestions(sequence) {
            const cleaned = sequence.replace(/ /g, '').toUpperCase();
            const suggestionsBox = document.getElementById('suggestions');
            suggestionsBox.innerHTML = '';
            
            const matchingSuggestions = suggestionsData
                .filter(suggestion => suggestion.replace(/ /g, '').startsWith(cleaned))
                .slice(0, 5);
            
            matchingSuggestions.forEach(word => {
                const suggestionDiv = document.createElement('div');
                suggestionDiv.className = 'suggestion';
                suggestionDiv.innerText = word;
                suggestionDiv.onclick = () => {
                    finalSentence = word;
                    document.getElementById('final-sentence').innerText = word;
                    speak(word);
                };
                suggestionsBox.appendChild(suggestionDiv);
            });
        }
        
        function fetchSequence() {
            console.log('Fetching sequence.txt...');
            const fetchURL = '/sequence.txt?ts=' + Date.now();
            console.log('Fetching from:', fetchURL);
            fetch(fetchURL)
                .then(response => {
                    console.log('Response status:', response.status);
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    return response.text();
                })
                .then(sequenceData => {
                    console.log('Raw sequence data:', JSON.stringify(sequenceData));
                    console.log('Sequence data length:', sequenceData.length);
                    
                    const cleaned = sequenceData.trim();
                    console.log('Cleaned sequence data:', JSON.stringify(cleaned));
                    console.log('Cleaned length:', cleaned.length);
                    
                    if (cleaned && cleaned.length > 0) {
                        console.log('Displaying sequence:', cleaned.toUpperCase());
                        document.getElementById('signed-seq').innerText = cleaned.toUpperCase();
                        updateConnectionStatus('connected');
                        
                        if (cleaned.toUpperCase() !== lastSequence) {
                            updateSuggestions(cleaned.toUpperCase());
                            lastSequence = cleaned.toUpperCase();
                        }
                    } else {
                        console.log('No content, showing listening...');
                        document.getElementById('signed-seq').innerText = '[Listening...]';
                        updateConnectionStatus('listening');
                        
                        if (lastSequence !== '') {
                            document.getElementById('suggestions').innerHTML = '';
                            lastSequence = '';
                        }
                    }
                })
                .catch(error => {
                    console.error('Failed to fetch sequence.txt:', error);
                    document.getElementById('signed-seq').innerText = '[Connection Error]';
                    updateConnectionStatus('error');
                    
                    document.getElementById('suggestions').innerHTML = '';
                    lastSequence = '';
                });
        }
        
        window.onload = () => {
            fetchSequence();
            setInterval(fetchSequence, 1000);
        };
        
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                fetchSequence();
            }
        });
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