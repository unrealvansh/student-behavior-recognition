from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'behavior-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# TODO: Import your pipeline when ready
# from src.pipeline import BehaviorRecognitionPipeline

camera = None
is_streaming = False

def generate_frames():
    global camera, is_streaming
    
    # TODO: Replace with real pipeline
    # pipeline = BehaviorRecognitionPipeline()
    
    camera = cv2.VideoCapture(0)
    
    while is_streaming:
        success, frame = camera.read()
        if not success:
            break
        
        # TODO: Run pipeline
        # frame, stats = pipeline.process(frame)
        
        # Encode frame
        _, buffer = cv2.imencode('.jpg', frame)
        frame_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # Dummy stats (replace with real data)
        socketio.emit('frame_update', {
            'frame': frame_b64,
            'stats': {
                'attentive': 18,
                'phone': 3,
                'sleeping': 2,
                'hand': 5,
                'total': 28,
                'detections': 154,
                'alerts': 6,
                'status': 'Running'
            },
            'detection': {
                'time': datetime.now().strftime('%I:%M %p'),
                'student': 'Student 05',
                'behaviour': 'Attentive',
                'color': 'green-text'
            }
        })
        
        time.sleep(0.05)
    
    camera.release()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    emit('status', {'message': 'Connected'})

@socketio.on('start_stream')
def start_stream():
    global is_streaming
    if not is_streaming:
        is_streaming = True
        thread = threading.Thread(target=generate_frames)
        thread.daemon = True
        thread.start()
    emit('stream_status', {'streaming': True})

@socketio.on('stop_stream')
def stop_stream():
    global is_streaming
    is_streaming = False
    emit('stream_status', {'streaming': False})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
