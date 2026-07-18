const socket = io();

// Elements
const liveFeed = document.getElementById('live-feed');
const placeholder = document.getElementById('camera-placeholder');
const liveToggle = document.getElementById('live-toggle');

const stats = {
    attentive: document.getElementById('stat-attentive'),
    phone: document.getElementById('stat-phone'),
    sleeping: document.getElementById('stat-sleeping'),
    hand: document.getElementById('stat-hand')
};

const summary = {
    total: document.getElementById('sum-total'),
    detections: document.getElementById('sum-detections'),
    alerts: document.getElementById('sum-alerts'),
    status: document.getElementById('sum-status')
};

const tableBody = document.getElementById('detections-body');

let isStreaming = false;

// Toggle stream on "Live Monitoring" click
liveToggle.addEventListener('click', () => {
    if (!isStreaming) {
        socket.emit('start_stream');
        liveToggle.style.background = '#16a34a';
        summary.status.textContent = 'Running';
        summary.status.className = 'green-text';
    } else {
        socket.emit('stop_stream');
        liveToggle.style.background = '';
        summary.status.textContent = 'Idle';
        liveFeed.style.display = 'none';
        placeholder.style.display = 'block';
        placeholder.textContent = 'Stream stopped';
    }
    isStreaming = !isStreaming;
});

// Receive frame
socket.on('frame_update', (data) => {
    // Show video
    placeholder.style.display = 'none';
    liveFeed.style.display = 'block';
    liveFeed.src = 'data:image/jpeg;base64,' + data.frame;
    
    // Update stats
    const s = data.stats;
    stats.attentive.textContent = s.attentive;
    stats.phone.textContent = s.phone;
    stats.sleeping.textContent = s.sleeping;
    stats.hand.textContent = s.hand;
    
    summary.total.textContent = s.total;
    summary.detections.textContent = s.detections;
    summary.alerts.textContent = s.alerts;
    
    // Add detection row
    if (data.detection) {
        const d = data.detection;
        const row = document.createElement('tr');
        row.innerHTML = `
            <td>${d.time}</td>
            <td>${d.student}</td>
            <td class="${d.color}">${d.behaviour}</td>
        `;
        tableBody.insertBefore(row, tableBody.firstChild);
        
        // Keep only last 10 rows
        while (tableBody.children.length > 10) {
            tableBody.removeChild(tableBody.lastChild);
        }
    }
});

socket.on('stream_status', (data) => {
    console.log('Stream status:', data.streaming);
});
