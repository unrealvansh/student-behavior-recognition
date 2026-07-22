"""Flask + SocketIO server for the Student Behaviour Recognition Dashboard.

Streams webcam frames through the SmartPipeline (YOLOv8 → ByteTrack →
MobileNetV2 → temporal smoothing) and pushes annotated frames + live
statistics to the browser via WebSocket.
"""

from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import cv2
import base64
import threading
import time
import logging

app = Flask(__name__,
            template_folder='frontend/templates',
            static_folder='frontend/static')
app.config['SECRET_KEY'] = 'behavior-secret'

# ── FORCE threading mode (gevent is installed but incompatible with OpenCV) ──
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Logging ───────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Global state ──────────────────────────────────────────────────────────
camera = None
is_streaming = False
pipeline = None          # Lazy-initialized on first stream start
pipeline_lock = threading.Lock()


def get_pipeline():
    """Lazy-load the SmartPipeline (heavy models only loaded once)."""
    global pipeline
    if pipeline is None:
        with pipeline_lock:
            if pipeline is None:
                logger.info("Loading AI pipeline (YOLOv8 + MobileNetV2)...")
                from src.full_pipeline import SmartPipeline
                pipeline = SmartPipeline(
                    classifier_weights="models/behavior_classifier.pth",
                    tracker_model="yolov8m.pt",
                    label_window=15,
                    conf_threshold=0.3,
                )
                logger.info("Pipeline loaded successfully.")
    return pipeline


def generate_frames():
    """Capture webcam frames, run AI pipeline, push results via SocketIO."""
    global camera, is_streaming

    try:
        pipe = get_pipeline()
        pipe.reset()
    except Exception as e:
        logger.error(f"Failed to load AI pipeline: {e}")
        socketio.emit('stream_error', {'message': f'AI pipeline error: {e}'})
        is_streaming = False
        return

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        logger.error("Could not open webcam (index 0)")
        socketio.emit('stream_error', {'message': 'Could not open webcam. Check that no other app is using it.'})
        is_streaming = False
        return

    logger.info("Webcam opened — streaming started")

    frame_count = 0

    while is_streaming:
        success, frame = camera.read()
        if not success:
            logger.warning("Failed to read frame from webcam")
            break

        try:
            # ── Run the 3-step AI pipeline ────────────────────────────────
            annotated_frame, stats, detections = pipe.process_frame(frame)
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            continue

        # Clean up stale tracked IDs every 30 frames
        frame_count += 1
        if frame_count % 30 == 0:
            pipe.cleanup_stale_ids(set())

        # ── Encode annotated frame to JPEG → base64 ──────────────────────
        _, buffer = cv2.imencode('.jpg', annotated_frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, 75])
        frame_b64 = base64.b64encode(buffer).decode('utf-8')

        # ── Build the payload for the dashboard ───────────────────────────
        latest_detection = None
        if detections:
            alerts = [d for d in detections
                      if d['color'] in ('red-text', 'yellow-text')]
            latest_detection = alerts[-1] if alerts else detections[-1]

        socketio.emit('frame_update', {
            'frame': frame_b64,
            'stats': stats,
            'detection': latest_detection,
        })

        # ~15 fps target
        time.sleep(0.065)

    if camera is not None:
        camera.release()
        camera = None
    logger.info("Streaming stopped")


# ── Routes ────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', active_page='dashboard')


@app.route('/analytics')
def analytics():
    return render_template('analytics.html', active_page='analytics')


@app.route('/students')
def students():
    return render_template('students.html', active_page='students')


@app.route('/alerts')
def alerts():
    return render_template('alerts.html', active_page='alerts')


@app.route('/settings')
def settings():
    return render_template('settings.html', active_page='settings')


@socketio.on('connect')
def handle_connect():
    logger.info("Client connected")
    emit('status', {'message': 'Connected to BehaveX AI server'})


@socketio.on('start_stream')
def start_stream():
    global is_streaming
    logger.info("Received start_stream event")
    if not is_streaming:
        is_streaming = True
        thread = threading.Thread(target=generate_frames, daemon=True)
        thread.start()
    emit('stream_status', {'streaming': True})


@socketio.on('stop_stream')
def stop_stream():
    global is_streaming
    logger.info("Received stop_stream event")
    is_streaming = False
    emit('stream_status', {'streaming': False})


if __name__ == '__main__':
    logger.info("Starting BehaveX server on http://0.0.0.0:5000")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000,
                 allow_unsafe_werkzeug=True)
