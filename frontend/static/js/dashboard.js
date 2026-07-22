/**
 * BehaveX Dashboard — SocketIO client
 *
 * Connects to the Flask server via WebSocket.
 * When the user clicks "Start Monitoring", emits 'start_stream'.
 * The server opens the webcam, runs the AI pipeline, and pushes
 * annotated frames + stats back via the 'frame_update' event.
 */

// ── SocketIO Connection ──────────────────────────────────────────────────

const socket = io();

// ── DOM References ───────────────────────────────────────────────────────

const liveFeed       = document.getElementById('live-feed');
const placeholder    = document.getElementById('camera-placeholder');
const startBtn       = document.getElementById('start-btn');
const badge          = document.getElementById('connection-badge');
const tableBody      = document.getElementById('detections-body');

// 5 behavior stat counters
const statEls = {
    sitting:  document.getElementById('stat-sitting'),
    reading:  document.getElementById('stat-reading'),
    hand:     document.getElementById('stat-hand'),
    phone:    document.getElementById('stat-phone'),
    sleeping: document.getElementById('stat-sleeping'),
};

// Session summary
const sumEls = {
    total:      document.getElementById('sum-total'),
    detections: document.getElementById('sum-detections'),
    alerts:     document.getElementById('sum-alerts'),
    status:     document.getElementById('sum-status'),
};

let isStreaming = false;

// ── Connection status ────────────────────────────────────────────────────

socket.on('connect', () => {
    if (badge) {
        badge.textContent = 'Connected';
        badge.className = 'badge badge-online';
    }
    console.log('[BehaveX] Connected to server');
});

socket.on('disconnect', () => {
    if (badge) {
        badge.textContent = 'Offline';
        badge.className = 'badge badge-offline';
    }
    console.log('[BehaveX] Disconnected');
});

// ── Start / Stop button ──────────────────────────────────────────────────

if (startBtn) {
    startBtn.addEventListener('click', () => {
        if (!isStreaming) {
            // ── START ────────────────────────────────────────────────────
            isStreaming = true;
            socket.emit('start_stream');

            startBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Monitoring';
            startBtn.classList.add('btn-active');

            if (placeholder) {
                placeholder.innerHTML = '<i class="fas fa-spinner fa-spin" style="font-size:2rem;opacity:0.4;"></i><p>Loading AI models… please wait</p>';
            }

            if (sumEls.status) {
                sumEls.status.textContent = 'Loading AI…';
                sumEls.status.className = 'yellow-text';
            }

            console.log('[BehaveX] start_stream emitted');
        } else {
            // ── STOP ─────────────────────────────────────────────────────
            isStreaming = false;
            socket.emit('stop_stream');

            startBtn.innerHTML = '<i class="fas fa-play"></i> Start Monitoring';
            startBtn.classList.remove('btn-active');

            if (liveFeed) liveFeed.style.display = 'none';
            if (placeholder) {
                placeholder.style.display = 'flex';
                placeholder.innerHTML = '<i class="fas fa-video" style="font-size:2.5rem;margin-bottom:12px;opacity:0.3;"></i><p>Stream stopped. Click <strong>"Start Monitoring"</strong> to resume.</p>';
            }

            if (sumEls.status) {
                sumEls.status.textContent = 'Idle';
                sumEls.status.className = 'gray-text';
            }

            console.log('[BehaveX] stop_stream emitted');
        }
    });
}

// Auto-start if navigated with ?autostart=1
if (new URLSearchParams(window.location.search).get('autostart') === '1') {
    window.history.replaceState({}, '', '/');
    const doStart = () => { if (startBtn) startBtn.click(); };
    if (socket.connected) {
        setTimeout(doStart, 300);
    } else {
        socket.once('connect', () => setTimeout(doStart, 300));
    }
}

// ── Receive frames + stats from server ───────────────────────────────────

socket.on('frame_update', (data) => {
    // Show the live video frame
    if (placeholder) placeholder.style.display = 'none';
    if (liveFeed) {
        liveFeed.style.display = 'block';
        liveFeed.src = 'data:image/jpeg;base64,' + data.frame;
    }

    // Update the 5 behavior stat counters
    const s = data.stats;
    if (statEls.sitting)  statEls.sitting.textContent  = s.sitting  || 0;
    if (statEls.reading)  statEls.reading.textContent  = s.reading  || 0;
    if (statEls.hand)     statEls.hand.textContent     = s.hand     || 0;
    if (statEls.phone)    statEls.phone.textContent    = s.phone    || 0;
    if (statEls.sleeping) statEls.sleeping.textContent = s.sleeping || 0;

    // Update session summary
    if (sumEls.total)      sumEls.total.textContent      = s.total      || 0;
    if (sumEls.detections) sumEls.detections.textContent = s.detections  || 0;
    if (sumEls.alerts)     sumEls.alerts.textContent     = s.alerts      || 0;
    if (sumEls.status) {
        sumEls.status.textContent = 'Running';
        sumEls.status.className   = 'green-text';
    }

    // Add latest detection to the table
    if (data.detection && tableBody) {
        const d = data.detection;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${d.time}</td>
            <td>${d.student}</td>
            <td class="${d.color}">${d.behaviour}</td>
        `;
        tableBody.insertBefore(row, tableBody.firstChild);

        // Keep only the last 20 rows
        while (tableBody.children.length > 20) {
            tableBody.removeChild(tableBody.lastChild);
        }
    }
});

// ── Error handling ───────────────────────────────────────────────────────

socket.on('stream_error', (data) => {
    console.error('[BehaveX] Stream error:', data.message);
    if (placeholder) {
        placeholder.style.display = 'flex';
        placeholder.innerHTML = `<i class="fas fa-exclamation-triangle" style="font-size:2rem;color:var(--color-red);margin-bottom:12px;"></i><p>⚠ ${data.message}</p><small>Check that your webcam is connected and not used by another app</small>`;
    }
    if (liveFeed) liveFeed.style.display = 'none';
    if (sumEls.status) {
        sumEls.status.textContent = 'Error';
        sumEls.status.className = 'red-text';
    }
    isStreaming = false;
    if (startBtn) {
        startBtn.innerHTML = '<i class="fas fa-play"></i> Start Monitoring';
        startBtn.classList.remove('btn-active');
    }
});

socket.on('stream_status', (data) => {
    console.log('[BehaveX] Stream status:', data.streaming);
});
