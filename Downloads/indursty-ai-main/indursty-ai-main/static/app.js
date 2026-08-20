document.addEventListener('DOMContentLoaded', () => {
    loadDatasetStats();
    loadModelBenchmarks();
    loadSampleImages();
    setupDropzone();
});

// Tab Switcher
function switchTab(tabId, btnElement) {
    document.querySelectorAll('.tab-content').forEach(tab => tab.classList.remove('active'));
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));

    document.getElementById(tabId).classList.add('active');
    btnElement.classList.add('active');
}

// 1. Load Dataset Stats & Render Chart
async function loadDatasetStats() {
    try {
        const res = await fetch('/api/stats');
        const data = await res.json();

        document.getElementById('stat-clean').innerText = data.total_clean_unique.toLocaleString();
        document.getElementById('stat-audit').innerText = `${data.duplicates_removed} Dups`;

        const splits = data.classification_splits;
        const totalDef = splits.train.defective + splits.val.defective + splits.test.defective;
        const totalOK = splits.train.ok + splits.val.ok + splits.test.ok;

        document.getElementById('stat-def').innerText = totalDef.toLocaleString();
        document.getElementById('stat-ok').innerText = totalOK.toLocaleString();

        // Render Chart.js
        const ctx = document.getElementById('splitChart').getContext('2d');
        new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Train Set (70%)', 'Val Set (15%)', 'Test Set (15%)'],
                datasets: [
                    {
                        label: 'Defective (def_front)',
                        data: [splits.train.defective, splits.val.defective, splits.test.defective],
                        backgroundColor: '#ff3366'
                    },
                    {
                        label: 'OK (ok_front)',
                        data: [splits.train.ok, splits.val.ok, splits.test.ok],
                        backgroundColor: '#00e676'
                    }
                ]
            },
            options: {
                responsive: true,
                scales: {
                    x: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { stacked: true, grid: { color: 'rgba(255,255,255,0.05)' } }
                },
                plugins: {
                    legend: { labels: { color: '#f1f5f9' } }
                }
            }
        });
    } catch (err) {
        console.error('Error loading dataset stats:', err);
    }
}

// 2. Load Model Benchmark Table
async function loadModelBenchmarks() {
    try {
        const res = await fetch('/api/models');
        const data = await res.json();

        const tbody = document.querySelector('#benchmark-table tbody');
        tbody.innerHTML = '';

        for (const [modelName, metrics] of Object.entries(data)) {
            const tr = document.createElement('tr');
            tr.innerHTML = `
                <td><strong style="color: var(--primary);">${modelName}</strong></td>
                <td>${metrics.primary_task}</td>
                <td><strong style="color: var(--accent-green);">${(metrics.accuracy * 100).toFixed(2)}%</strong></td>
                <td>${metrics.f1_score.toFixed(4)}</td>
                <td>${metrics.map_50 ? metrics.map_50.toFixed(3) : 'N/A'}</td>
                <td>${metrics.avg_latency_ms} ms</td>
                <td>${metrics.parameters}</td>
            `;
            tbody.appendChild(tr);
        }
    } catch (err) {
        console.error('Error loading model benchmarks:', err);
    }
}



// 4. Setup Drag & Drop File Upload
async function loadSampleImages() {
    try {
        const res = await fetch('/api/sample-images');
        const samples = await res.json();
        const sampleList = document.getElementById('sampleList');
        sampleList.innerHTML = '';

        samples.forEach(sample => {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = sample.label.toLowerCase().includes('def')
                ? 'sample-button sample-defective'
                : 'sample-button sample-ok';
            button.innerText = `${sample.label}: ${sample.name}`;
            button.addEventListener('click', () => inspectSamplePath(sample.path));
            sampleList.appendChild(button);
        });
    } catch (err) {
        console.error('Error loading verified samples:', err);
    }
}

async function inspectSamplePath(path) {
    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('loadingOverlay').style.display = 'flex';

    try {
        const res = await fetch('/api/inspect-path', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path })
        });
        if (!res.ok) throw new Error(`Inspection failed: ${res.status}`);
        const result = await res.json();
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('resultContainer').style.display = 'block';
        renderInspectionResults(result);
    } catch (err) {
        console.error('Error inspecting sample:', err);
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('resultContainer').style.display = 'block';
        document.getElementById('consensus-status').innerText = 'Inspection Error';
    }
}

// 4. Setup Drag & Drop File Upload
function setupDropzone() {
    const dropzone = document.getElementById('dropzone');

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length > 0) {
            handleFileUpload(files[0]);
        }
    });
}

// 5. Handle Image Upload & Execute Multi-Model Inspection
async function handleFileUpload(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);

    document.getElementById('resultContainer').style.display = 'none';
    document.getElementById('loadingOverlay').style.display = 'flex';

    try {
        const res = await fetch('/api/inspect', {
            method: 'POST',
            body: formData
        });
        const result = await res.json();

        // Slight delay for premium animation feel
        setTimeout(() => {
            document.getElementById('loadingOverlay').style.display = 'none';
            document.getElementById('resultContainer').style.display = 'block';
            renderInspectionResults(result);
        }, 600);

    } catch (err) {
        console.error('Error inspecting image:', err);
        document.getElementById('loadingOverlay').style.display = 'none';
        document.getElementById('resultContainer').style.display = 'block';
        document.getElementById('consensus-status').innerText = 'Inspection Error';
    }
}



// Render Results & AI Agent Diagnostics
function renderInspectionResults(result) {
    const statusEl = document.getElementById('consensus-status');
    statusEl.innerText = result.consensus_status;
    statusEl.style.color = result.consensus_status.includes('DEFECTIVE') ? 'var(--accent-red)' : 'var(--accent-green)';

    document.getElementById('consensus-confidence').innerText = `${result.consensus_confidence}%`;

    document.getElementById('img-yolo').src = result.yolo_result.image_b64;
    document.getElementById('yolo-info').innerText = `YOLOv11 BBoxes: ${result.yolo_result.box_count} detected`;

    document.getElementById('img-eff').src = result.efficientnet_result.gradcam_b64;
    document.getElementById('eff-info').innerText = `EfficientNet-B3 Confidence: ${result.efficientnet_result.confidence}%`;

    document.getElementById('img-vit').src = result.vit_result.attention_b64;
    document.getElementById('vit-info').innerText = `ViT Self-Attention Confidence: ${result.vit_result.confidence}%`;
}
