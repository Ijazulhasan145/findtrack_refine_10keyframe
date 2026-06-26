document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('segmentation-form');
    const dropArea = document.getElementById('drop-area');
    const fileInput = document.getElementById('video-input');
    const fileMsg = document.querySelector('.file-msg');
    
    // UI States
    const stateEmpty = document.getElementById('state-empty');
    const stateLoading = document.getElementById('state-loading');
    const stateResult = document.getElementById('state-result');
    
    // Result elements
    const resultVideo = document.getElementById('result-video');
    const downloadBtn = document.getElementById('download-btn');
    const submitBtn = document.getElementById('submit-btn');

    // --- Drag and Drop Logic ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropArea.addEventListener(eventName, () => {
            dropArea.classList.remove('dragover');
        }, false);
    });

    dropArea.addEventListener('drop', (e) => {
        const dt = e.dataTransfer;
        const files = dt.files;
        if (files.length) {
            fileInput.files = files;
            updateFileMsg();
        }
    }, false);

    fileInput.addEventListener('change', updateFileMsg);

    function updateFileMsg() {
        if (fileInput.files.length > 0) {
            const fileName = fileInput.files[0].name;
            fileMsg.textContent = fileName;
            fileMsg.style.color = 'var(--primary)';
        }
    }

    // --- UI State Management ---
    function showState(stateElement) {
        stateEmpty.classList.remove('active');
        stateLoading.classList.remove('active');
        stateResult.classList.remove('active');
        stateElement.classList.add('active');
    }

    // --- Form Submission Logic ---
    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        if (fileInput.files.length === 0) {
            alert('Please select a video file first.');
            return;
        }

        const formData = new FormData(form);

        // Update UI to Loading
        submitBtn.disabled = true;
        submitBtn.innerHTML = '<span>Processing...</span><i class="ph ph-spinner-gap"></i>';
        showState(stateLoading);

        try {
            // Call the FastAPI backend using a relative path so it works seamlessly with Ngrok
            const response = await fetch('/api/segment', {
                method: 'POST',
                body: formData,
                headers: {
                    'ngrok-skip-browser-warning': 'true'
                }
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Failed to process video');
            }

            // Convert response to Blob
            const videoBlob = await response.blob();
            const videoUrl = URL.createObjectURL(videoBlob);

            // Update UI with Result
            resultVideo.src = videoUrl;
            downloadBtn.href = videoUrl;
            
            showState(stateResult);
            
        } catch (error) {
            console.error('Error:', error);
            alert(`An error occurred: ${error.message}`);
            showState(stateEmpty);
        } finally {
            // Restore Submit Button
            submitBtn.disabled = false;
            submitBtn.innerHTML = '<span>Segment Video</span><i class="ph-bold ph-arrow-right"></i>';
        }
    });
});
