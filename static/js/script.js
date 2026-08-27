/**
 * AI Music Genre Classifier - Client-side UI & Interaction Logic
 */

// Initialize Theme immediately to prevent flash
(function() {
    const savedTheme = localStorage.getItem('symphony_theme') || 'light';
    document.documentElement.setAttribute('data-theme', savedTheme);
})();

document.addEventListener('DOMContentLoaded', () => {
    // Theme Switcher
    const themeToggleBtns = document.querySelectorAll('.theme-toggle-btn');
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
    updateThemeIcons(currentTheme);

    themeToggleBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            const activeTheme = document.documentElement.getAttribute('data-theme') || 'light';
            const newTheme = activeTheme === 'light' ? 'dark' : 'light';
            document.documentElement.setAttribute('data-theme', newTheme);
            localStorage.setItem('symphony_theme', newTheme);
            updateThemeIcons(newTheme);
        });
    });

    function updateThemeIcons(theme) {
        themeToggleBtns.forEach(btn => {
            if (theme === 'dark') {
                btn.innerHTML = '<i class="fas fa-sun"></i>';
                btn.title = 'Switch to Light Mode';
            } else {
                btn.innerHTML = '<i class="fas fa-moon"></i>';
                btn.title = 'Switch to Dark Mode';
            }
        });
    }

    // 1. Audio Upload & Dropzone Handling
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('fileInput');
    const selectedFileCard = document.getElementById('selectedFileCard');
    const selectedFileName = document.getElementById('selectedFileName');
    const selectedFileSize = document.getElementById('selectedFileSize');
    const submitBtn = document.getElementById('submitBtn');
    const uploadForm = document.getElementById('uploadForm');
    const processingOverlay = document.getElementById('processingOverlay');
    const processingStatus = document.getElementById('processingStatus');
    const previewAudioPlayer = document.getElementById('previewAudioPlayer');
    const previewAudioSource = document.getElementById('previewAudioSource');

    if (dropzone && fileInput) {
        // Drag & Drop events
        ['dragenter', 'dragover'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            dropzone.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
                dropzone.classList.remove('dragover');
            });
        });

        dropzone.addEventListener('drop', (e) => {
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                fileInput.files = files;
                handleFileSelection(files[0]);
            }
        });

        fileInput.addEventListener('change', (e) => {
            if (fileInput.files.length > 0) {
                handleFileSelection(fileInput.files[0]);
            }
        });
    }

    function handleFileSelection(file) {
        const validExtensions = ['wav', 'mp3', 'flac', 'ogg', 'm4a'];
        const fileExt = file.name.split('.').pop().toLowerCase();

        if (!validExtensions.includes(fileExt)) {
            alert('Please select a valid audio file (.wav, .mp3, .flac, .ogg, .m4a)');
            return;
        }

        if (selectedFileName) selectedFileName.textContent = file.name;
        if (selectedFileSize) selectedFileSize.textContent = formatBytes(file.size);
        if (selectedFileCard) selectedFileCard.style.display = 'block';
        if (submitBtn) submitBtn.disabled = false;

        // Audio preview URL
        if (previewAudioPlayer && previewAudioSource) {
            const objectUrl = URL.createObjectURL(file);
            previewAudioSource.src = objectUrl;
            previewAudioPlayer.load();
            previewAudioPlayer.style.display = 'block';
        }
    }

    function formatBytes(bytes, decimals = 2) {
        if (bytes === 0) return '0 Bytes';
        const k = 1024;
        const dm = decimals < 0 ? 0 : decimals;
        const sizes = ['Bytes', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
    }

    // 2. Form Submission with AI Progress Simulator
    if (uploadForm) {
        uploadForm.addEventListener('submit', (e) => {
            if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
                // If using sample audio without file upload
                const sampleName = document.getElementById('sampleNameInput');
                if (!sampleName || !sampleName.value) {
                    e.preventDefault();
                    alert('Please select or upload an audio track first.');
                    return;
                }
            }

            if (processingOverlay) {
                processingOverlay.style.display = 'flex';
                simulateProcessingSteps();
            }
        });
    }

    function simulateProcessingSteps() {
        const steps = [
            'Decoding audio frequencies (22,050 Hz)...',
            'Computing 128-band Mel-Spectrogram matrix...',
            'Evaluating Deep Learning Neural Network...',
            'Generating Waveform & Centroid plots...',
            'Computing genre confidence distribution...'
        ];

        let stepIndex = 0;
        const interval = setInterval(() => {
            stepIndex++;
            if (stepIndex < steps.length && processingStatus) {
                processingStatus.textContent = steps[stepIndex];
            } else {
                clearInterval(interval);
            }
        }, 1200);
    }

    // 3. Audio Play/Pause UI Controller
    const audioPlayers = document.querySelectorAll('.custom-audio-controller');
    audioPlayers.forEach(player => {
        const audio = player.querySelector('audio');
        const playBtn = player.querySelector('.play-btn');
        const waveBars = player.querySelector('.wave-bars');

        if (audio && playBtn) {
            playBtn.addEventListener('click', () => {
                if (audio.paused) {
                    audio.play();
                    playBtn.innerHTML = '<i class="fas fa-pause"></i>';
                    if (waveBars) waveBars.style.opacity = '1';
                } else {
                    audio.pause();
                    playBtn.innerHTML = '<i class="fas fa-play"></i>';
                    if (waveBars) waveBars.style.opacity = '0.3';
                }
            });

            audio.addEventListener('ended', () => {
                playBtn.innerHTML = '<i class="fas fa-play"></i>';
                if (waveBars) waveBars.style.opacity = '0.3';
            });
        }
    });

    // 4. Quick Sample Selector
    window.selectPresetSample = function(samplePath, genreName) {
        const sampleInput = document.getElementById('sampleNameInput');
        if (sampleInput) {
            sampleInput.value = samplePath;
        }
        if (selectedFileName) selectedFileName.textContent = `${samplePath} (${genreName})`;
        if (selectedFileSize) selectedFileSize.textContent = 'GTZAN Benchmark Sample (30s)';
        if (selectedFileCard) selectedFileCard.style.display = 'block';
        if (submitBtn) submitBtn.disabled = false;

        if (uploadForm) {
            if (processingOverlay) {
                processingOverlay.style.display = 'flex';
                simulateProcessingSteps();
            }
            uploadForm.submit();
        }
    };
});
