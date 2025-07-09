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
            "HI", "HELLO", "HELLO I", "HELLO I AM", "HELLO I AM FINE",
            "HOW ARE YOU", "THANK YOU", "GOOD MORNING", "BYE", "I AM GOOD"
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
            // Server is running on port 8000, use relative URL since we're served by the same server
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
                    
                    // Process the sequence data
                    const cleaned = sequenceData.trim();
                    console.log('Cleaned sequence data:', JSON.stringify(cleaned));
                    console.log('Cleaned length:', cleaned.length);
                    
                    // Display the actual sequence data or listening status
                    if (cleaned && cleaned.length > 0) {
                        // Show the actual sequence from the file
                        console.log('Displaying sequence:', cleaned.toUpperCase());
                        document.getElementById('signed-seq').innerText = cleaned.toUpperCase();
                        updateConnectionStatus('connected');
                        
                        // Update suggestions based on the sequence
                        if (cleaned.toUpperCase() !== lastSequence) {
                            updateSuggestions(cleaned.toUpperCase());
                            lastSequence = cleaned.toUpperCase();
                        }
                    } else {
                        // File is empty or no content, show listening status
                        console.log('No content, showing listening...');
                        document.getElementById('signed-seq').innerText = '[Listening...]';
                        updateConnectionStatus('listening');
                        
                        // Clear suggestions when no sequence
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
                    
                    // Clear suggestions on error
                    document.getElementById('suggestions').innerHTML = '';
                    lastSequence = '';
                });
        }
        
        // Start fetching when page loads
        window.onload = () => {
            fetchSequence(); // Initial fetch
            setInterval(fetchSequence, 1000); // Then every second
        };
        
        // Handle page visibility changes to pause/resume when tab is not active
        document.addEventListener('visibilitychange', () => {
            if (document.visibilityState === 'visible') {
                fetchSequence(); // Immediate fetch when tab becomes visible
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
                print(f"Served sequence.txt content: '{data}'")
                
            except FileNotFoundError:
                self.send_error(404, "sequence.txt not found")
                print("Error: sequence.txt not found")
            return
        
        # Serve video feed (if you have this endpoint)
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
                    
                    time.sleep(0.033)  # ~30 FPS
                    
            except (ConnectionResetError, BrokenPipeError, OSError):
                print("Video feed client disconnected")
            except Exception as e:
                print(f"Video feed error: {e}")
            
            return
        
        # Handle other static files (CSS, JS, images, etc.)
        else:
            # Let the parent class handle other files
            super().do_GET()
    
    def do_OPTIONS(self):
        # Handle CORS preflight requests
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

def run_server(port=8000):
    try:
        # Create the server
        handler = MyHTTPRequestHandler
        httpd = socketserver.ThreadingTCPServer(("", port), handler)
        httpd.allow_reuse_address = True  # Allow port reuse
        
        print(f"Server starting on port {port}")
        print(f"Access your application at: http://localhost:{port}/")
        print(f"Test sequence.txt at: http://localhost:{port}/sequence.txt")
        print("Press Ctrl+C to stop the server")
        
        # Create sequence.txt with initial content if it doesn't exist
        if not os.path.exists("sequence.txt"):
            with open("sequence.txt", "w") as f:
                f.write("hh")
            print("Created sequence.txt with initial content 'hh'")
        
        # Start the server
        httpd.serve_forever()
        
    except KeyboardInterrupt:
        print("\nServer stopped by user")
        httpd.shutdown()
    except Exception as e:
        print(f"Error starting server: {e}")

if __name__ == "__main__":
    # You can change the port here if needed
    run_server(8000)
def start_server():
    with socketserver.TCPServer(("", 8000), MyHTTPRequestHandler) as httpd:
        print("Serving at http://localhost:8000")
        httpd.serve_forever()

# ─────────────────────── Load TensorFlow model ────────────────────────
label_lines = [l.rstrip()
               for l in tf.io.gfile.GFile("logs/trained_labels.txt")]

with tf.io.gfile.GFile("logs/trained_graph.pb", "rb") as f:
    graph_def = tf.compat.v1.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name="")

def predict(img_bytes, sess, softmax):
    probs = sess.run(softmax, {"DecodeJpeg/contents:0": img_bytes})[0]
    idx = int(probs.argmax())
    return label_lines[idx], float(probs[idx])

# ─────────────────────────── Main loop ────────────────────────────────
threading.Thread(target=start_server, daemon=True).start()

with tf.compat.v1.Session() as sess:
    softmax = sess.graph.get_tensor_by_name("final_result:0")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam")

    sequence = mem = ""
    i = consecutive = 0
    res = ""
    score = 0.0

    while True:
        ok, img = cap.read()
        if not ok:
            break
        img = cv2.flip(img, 1)

        # ROI
        x1, y1, x2, y2 = 100, 100, 300, 300
        roi = img[y1:y2, x1:x2]

        i += 1
        if i == 4:  # predict every 4th frame
            res, score = predict(cv2.imencode(".jpg", roi)[1].tobytes(),
                                 sess, softmax)
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
            mem = res
            with open("sequence.txt", "w") as f:
                f.write(sequence)

        # draw overlays
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, res.upper(), (100, 400),
                    cv2.FONT_HERSHEY_SIMPLEX, 4, (255, 255, 255), 4)

        # update MJPEG frame
        latest_frame = cv2.imencode(".jpg", img)[1].tobytes()

        # optional local view
        seq_img = np.zeros((200, 1200, 3), np.uint8)
        cv2.putText(seq_img, sequence.upper(), (30, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        cv2.imshow("Sign Language Recognizer", img)
        cv2.imshow("Sequence", seq_img)

        if cv2.waitKey(1) == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
