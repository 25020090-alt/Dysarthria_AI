let audioContext;
let mediaStream;
let processor;
let source;
let audioChunks = [];
let isRecording = false;

const micBtn = document.getElementById('mic-btn');
const instructionText = document.getElementById('instruction-text');
const statusBadge = document.getElementById('status-badge');
const transcriptionText = document.getElementById('transcription-text');
const waveform = document.getElementById('waveform');

// Tạo hiệu ứng sóng âm mồi
for(let i=0; i<40; i++) {
    const bar = document.createElement('div');
    bar.className = 'bar';
    waveform.appendChild(bar);
}

micBtn.addEventListener('click', toggleRecording);

const fileUpload = document.getElementById('audio-upload');
const uploadBtn = document.getElementById('upload-btn');

uploadBtn.addEventListener('click', () => {
    fileUpload.click();
});

fileUpload.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    instructionText.innerText = "UPLOADING...";
    statusBadge.innerText = "Processing File";
    statusBadge.style.color = "#8a2be2";
    transcriptionText.innerHTML = "<span class='placeholder'>Analyzing uploaded file...</span>";
    waveform.classList.add('active');

    const formData = new FormData();
    formData.append("audio_file", file);

    try {
        const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            transcriptionText.innerHTML = `<strong>Result:</strong> ${data.text}`;
            statusBadge.innerText = "Success";
            statusBadge.style.color = "#00ff88";
        } else {
            transcriptionText.innerText = "Lỗi xử lý AI: " + data.error;
            statusBadge.innerText = "Error";
            statusBadge.style.color = "#ff416c";
        }
    } catch (err) {
        console.error(err);
        transcriptionText.innerText = "Không thể kết nối đến máy chủ AI.";
        statusBadge.innerText = "Offline";
    }
    
    waveform.classList.remove('active');
    instructionText.innerText = "TAP TO SPEAK";
    fileUpload.value = ""; // Reset file input
});

async function toggleRecording() {
    if (!isRecording) {
        startRecording();
    } else {
        stopRecording();
    }
}

async function startRecording() {
    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        // Dùng AudioContext để lấy raw PCM thay vì WebM (Giúp Python Soundfile đọc được)
        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
        source = audioContext.createMediaStreamSource(mediaStream);
        processor = audioContext.createScriptProcessor(4096, 1, 1);
        
        audioChunks = [];
        
        processor.onaudioprocess = (e) => {
            if (!isRecording) return;
            const channelData = e.inputBuffer.getChannelData(0);
            audioChunks.push(new Float32Array(channelData));
            
            // Chạy hiệu ứng sóng âm nhấp nhô
            const bars = document.querySelectorAll('.bar');
            bars.forEach(bar => {
                const height = Math.random() * 25 + 5;
                bar.style.height = `${height}px`;
            });
        };
        
        source.connect(processor);
        processor.connect(audioContext.destination);
        
        isRecording = true;
        micBtn.classList.add('recording');
        instructionText.innerText = "TAP TO STOP";
        statusBadge.innerText = "Listening...";
        statusBadge.style.color = "#ff4b2b";
        waveform.classList.add('active');
        transcriptionText.innerHTML = "<span class='placeholder'>Listening to your voice...</span>";
        
    } catch (err) {
        console.error("Lỗi Microphone:", err);
        alert("Vui lòng cấp quyền Microphone trên trình duyệt để sử dụng AI.");
    }
}

function stopRecording() {
    isRecording = false;
    micBtn.classList.remove('recording');
    instructionText.innerText = "PROCESSING...";
    statusBadge.innerText = "Transcribing";
    statusBadge.style.color = "#00d2ff";
    waveform.classList.remove('active');
    
    // Tắt luồng
    if(source) source.disconnect();
    if(processor) processor.disconnect();
    if(mediaStream) mediaStream.getTracks().forEach(track => track.stop());
    if(audioContext) audioContext.close();
    
    processAndSendAudio();
}

async function processAndSendAudio() {
    // 1. Gộp mảng âm thanh
    const length = audioChunks.reduce((acc, chunk) => acc + chunk.length, 0);
    const flattened = new Float32Array(length);
    let offset = 0;
    for (const chunk of audioChunks) {
        flattened.set(chunk, offset);
        offset += chunk.length;
    }
    
    // 2. Chuyển PCM sang dạng chuẩn WAV
    const wavBlob = encodeWAV(flattened, 16000);
    
    // 3. Gửi lên máy chủ FastAPI
    const formData = new FormData();
    formData.append("audio_file", wavBlob, "recording.wav");
    
    try {
        const response = await fetch("/api/transcribe", {
            method: "POST",
            body: formData
        });
        
        const data = await response.json();
        
        if (data.success) {
            transcriptionText.innerHTML = `<strong>Result:</strong> ${data.text}`;
            statusBadge.innerText = "Success";
            statusBadge.style.color = "#00ff88";
        } else {
            transcriptionText.innerText = "Lỗi xử lý AI: " + data.error;
            statusBadge.innerText = "Error";
            statusBadge.style.color = "#ff416c";
        }
    } catch (err) {
        console.error(err);
        transcriptionText.innerText = "Không thể kết nối đến máy chủ AI.";
        statusBadge.innerText = "Offline";
    }
    
    instructionText.innerText = "TAP TO SPEAK";
}

// Bộ mã hóa: Convert Float32Array sang cấu trúc file WAV chuẩn
function encodeWAV(samples, sampleRate) {
    const buffer = new ArrayBuffer(44 + samples.length * 2);
    const view = new DataView(buffer);
    const writeString = (view, offset, string) => {
        for (let i = 0; i < string.length; i++) view.setUint8(offset + i, string.charCodeAt(i));
    };
    
    writeString(view, 0, 'RIFF');
    view.setUint32(4, 36 + samples.length * 2, true);
    writeString(view, 8, 'WAVE');
    writeString(view, 12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); 
    view.setUint16(22, 1, true); 
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true); 
    view.setUint16(34, 16, true); 
    writeString(view, 36, 'data');
    view.setUint32(40, samples.length * 2, true);
    
    let offset = 44;
    for (let i = 0; i < samples.length; i++) {
        let s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([view], { type: 'audio/wav' });
}
