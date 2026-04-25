document.addEventListener('DOMContentLoaded', () => {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const imagePreview = document.getElementById('image-preview');
    const analyzeBtn = document.getElementById('analyze-btn');
    const loadingStatus = document.getElementById('loading-status');
    const placeholderText = document.getElementById('placeholder-text');
    const resultContainer = document.getElementById('result-container');
    const predictionLabel = document.getElementById('prediction-label');
    const confidencePercent = document.getElementById('confidence-percent');
    const vizCard = document.getElementById('viz-card');
    const origImgDisplay = document.getElementById('orig-img-display');
    const gradImgDisplay = document.getElementById('grad-img-display');
    const downloadReportBtn = document.getElementById('download-report');
    const actionsContainer = document.getElementById('actions-container');

    let currentFile = null;

    // --- Upload Logic ---
    dropZone.addEventListener('click', () => fileInput.click());

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('dragover');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('dragover');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    function handleFile(file) {
        currentFile = file;
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            imagePreview.style.display = 'block';
            dropZone.style.display = 'none';
            analyzeBtn.disabled = false;
            
            // Reset Results
            placeholderText.style.display = 'block';
            resultContainer.style.display = 'none';
            vizCard.style.display = 'none';
            actionsContainer.style.display = 'none';
        };
        reader.readAsDataURL(file);
    }

    // --- Analysis Logic ---
    analyzeBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        // UI State: Loading
        analyzeBtn.disabled = true;
        loadingStatus.style.display = 'flex';
        placeholderText.style.display = 'none';
        resultContainer.style.display = 'none';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/analyze', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Analysis failed');

            const data = await response.json();
            displayResults(data);
        } catch (error) {
            console.error(error);
            alert('Error during analysis. Please check the backend connection.');
            analyzeBtn.disabled = false;
        } finally {
            loadingStatus.style.display = 'none';
        }
    });

    function displayResults(data) {
        // Update Results Card
        resultContainer.style.display = 'block';
        resultContainer.className = 'result-card ' + (data.is_malignant ? 'malignant' : 'benign');
        predictionLabel.textContent = data.label;
        confidencePercent.textContent = data.percent.toFixed(2) + '%';

        // Update Visualization Card
        vizCard.style.display = 'block';
        origImgDisplay.src = `data:image/png;base64,${data.original_image}`;
        if (data.gradcam_image) {
            gradImgDisplay.src = `data:image/png;base64,${data.gradcam_image}`;
        }

        actionsContainer.style.display = 'block';
        analyzeBtn.disabled = false;
    }

    // --- Report Logic ---
    downloadReportBtn.addEventListener('click', async () => {
        if (!currentFile) return;

        downloadReportBtn.disabled = true;
        downloadReportBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Generating...';

        const formData = new FormData();
        formData.append('file', currentFile);

        try {
            const response = await fetch('/report', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) throw new Error('Report generation failed');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'thyroid_analysis_report.docx';
            document.body.appendChild(a);
            a.click();
            a.remove();
        } catch (error) {
            console.error(error);
            alert('Error generating report.');
        } finally {
            downloadReportBtn.disabled = false;
            downloadReportBtn.innerHTML = '<i class="fas fa-file-download"></i> Generate Clinical Report';
        }
    });
});
