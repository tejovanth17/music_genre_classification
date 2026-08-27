// File size validation
document.addEventListener('DOMContentLoaded', function() {
    const fileInput = document.getElementById('file');
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            const file = this.files[0];
            if (file) {
                const maxSize = 50 * 1024 * 1024; // 50MB
                if (file.size > maxSize) {
                    alert('File size must be less than 50MB');
                    this.value = '';
                }
            }
        });
    }
});

// Auto cleanup on page unload
window.addEventListener('beforeunload', function() {
    const sessionId = document.querySelector('[data-session-id]')?.dataset.sessionId;
    if (sessionId) {
        fetch(`/cleanup/${sessionId}`);
    }
});
