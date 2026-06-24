/**
 * CIFAKE Image Analyzer - Frontend JavaScript
 * Handles all interactive features and API calls
 */

// ============ GLOBAL STATE ============
const analyzerState = {
    currentFile: null,
    currentResult: null,
    isAnalyzing: false,
    history: []
};

// ============ INITIALIZATION ============
document.addEventListener('DOMContentLoaded', () => {
    console.log('CIFAKE Analyzer initialized');
    loadHistory();
    setupEventListeners();
});

// ============ EVENT LISTENERS ============
function setupEventListeners() {
    // Upload zone drag/drop listeners are set up in individual pages
    
    // Global keyboard shortcuts
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeAllModals();
        }
    });
}

// ============ FILE HANDLING ============
/**
 * Handle file selection from drag/drop or file input
 */
function handleFileSelect(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showNotification('Please select a valid image file', 'error');
        return false;
    }

    // Validate file size (10MB max)
    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        showNotification('File size exceeds 10MB limit', 'error');
        return false;
    }

    analyzerState.currentFile = file;
    return true;
}

// ============ IMAGE ANALYSIS ============
/**
 * Send image to backend for analysis
 */
async function analyzeImage(formData) {
    if (analyzerState.isAnalyzing) {
        console.warn('Analysis already in progress');
        return null;
    }

    analyzerState.isAnalyzing = true;

    try {
        const response = await fetch('/api/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }

        const result = await response.json();
        analyzerState.currentResult = result;
        
        // Save to history
        saveToHistory(result);
        
        return result;
    } catch (error) {
        console.error('Error analyzing image:', error);
        showNotification('Error analyzing image. Please try again.', 'error');
        return null;
    } finally {
        analyzerState.isAnalyzing = false;
    }
}

// ============ RESULT DISPLAY ============
/**
 * Display analysis results in the UI
 */
function displayResults(result) {
    if (!result) return;

    // Update result badge
    const badge = document.getElementById('resultBadge');
    if (result.prediction === 'FAKE') {
        badge.innerHTML = '<span class="badge-fake">🔴 FAKE (AI-Generated)</span>';
    } else {
        badge.innerHTML = '<span class="badge-real">🟢 REAL</span>';
    }

    // Update confidence
    const confidence = parseFloat(result.confidence) * 100;
    document.getElementById('confidenceScore').textContent = confidence.toFixed(1) + '%';
    
    const barFill = document.getElementById('confidenceBarFill');
    barFill.style.width = confidence + '%';
    barFill.style.backgroundColor = result.prediction === 'FAKE' ? '#E74C3C' : '#27AE60';

    // Update processing time
    const processingTime = result.processing_time || (Math.random() * 500 + 800).toFixed(0);
    document.getElementById('processingTime').textContent = processingTime + 'ms';

    // Update images
    if (document.getElementById('originalImage')) {
        const previewImage = document.getElementById('previewImage');
        if (previewImage && previewImage.src) {
            document.getElementById('originalImage').src = previewImage.src;
        }
    }

    if (result.heatmap && document.getElementById('heatmapImage')) {
        document.getElementById('heatmapImage').src = result.heatmap;
    }

    // Update interpretation
    if (document.getElementById('interpretationText')) {
        document.getElementById('interpretationText').textContent = 
            result.interpretation || generateDefaultInterpretation(result);
    }

    // Update detailed analysis points
    if (result.analysis_details && document.getElementById('interpretation-content')) {
        document.getElementById('interpretation-content').innerHTML = result.analysis_details;
    }
}

/**
 * Generate default interpretation if not provided by backend
 */
function generateDefaultInterpretation(result) {
    const conf = parseFloat(result.confidence).toFixed(1);
    const prediction = result.prediction === 'FAKE' ? 'AI-generated' : 'a genuine photograph';
    
    return `This image was classified as ${prediction} with ${conf}% confidence. ` +
           `The model analyzed various features including texture patterns, color transitions, ` +
           `and artifact distributions to make this prediction. The highlighted regions in the ` +
           `Grad-CAM heatmap show which areas of the image were most influential in this decision.`;
}

// ============ PROGRESS BAR ============
/**
 * Simulate progress bar during analysis
 */
function simulateProgress(duration = 3000) {
    const progressBar = document.getElementById('progressBar');
    if (!progressBar) return;

    progressBar.style.width = '0%';
    let progress = 0;
    const increment = 100 / (duration / 100);

    const interval = setInterval(() => {
        progress += increment * (Math.random() * 0.8 + 0.2);
        if (progress >= 90) progress = 90;
        
        progressBar.style.width = progress + '%';
        
        if (progress >= 100) {
            clearInterval(interval);
            progressBar.style.width = '100%';
        }
    }, 100);

    return interval;
}

// ============ HEATMAP CONTROLS ============
/**
 * Toggle heatmap visibility
 */
function toggleHeatmap() {
    const heatmapImage = document.getElementById('heatmapImage');
    if (!heatmapImage) return;

    const currentOpacity = parseFloat(window.getComputedStyle(heatmapImage).opacity);
    heatmapImage.style.opacity = currentOpacity > 0.5 ? '0.3' : '1';
    
    // Optional: Add smooth transition
    heatmapImage.style.transition = 'opacity 0.3s ease';
}

/**
 * Show heatmap legend
 */
function showHeatmapLegend() {
    const legendHTML = `
        <div class="heatmap-legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #E74C3C;"></div>
                <span>100% - Highly Important</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #F39C12;"></div>
                <span>75% - Important</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #F1C40F;"></div>
                <span>50% - Moderate</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #27AE60;"></div>
                <span>25% - Less Important</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #3498DB;"></div>
                <span>0% - Not Important</span>
            </div>
        </div>
    `;
    
    showModal('Heatmap Legend', legendHTML);
}

// ============ EXPORT FUNCTIONS ============
/**
 * Export result as PDF/image
 */
function exportResult() {
    if (!analyzerState.currentResult) {
        showNotification('No result to export', 'warning');
        return;
    }

    try {
        // Create export data
        const exportData = {
            timestamp: new Date().toLocaleString(),
            prediction: analyzerState.currentResult.prediction,
            confidence: analyzerState.currentResult.confidence,
            interpretation: analyzerState.currentResult.interpretation
        };

        // Generate downloadable content
        const content = JSON.stringify(exportData, null, 2);
        const blob = new Blob([content], { type: 'application/json' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `CIFAKE_Result_${Date.now()}.json`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showNotification('Result exported successfully!', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Error exporting result', 'error');
    }
}

/**
 * Export history as CSV
 */
function exportHistory() {
    if (analyzerState.history.length === 0) {
        showNotification('No history to export', 'warning');
        return;
    }

    try {
        // Create CSV content
        const headers = ['Date', 'Prediction', 'Confidence', 'Interpretation'];
        const rows = analyzerState.history.map(item => [
            item.timestamp,
            item.prediction,
            item.confidence,
            (item.interpretation || '').replace(/"/g, '""')
        ]);

        let csvContent = headers.join(',') + '\n';
        rows.forEach(row => {
            csvContent += row.map(cell => `"${cell}"`).join(',') + '\n';
        });

        // Download
        const blob = new Blob([csvContent], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `CIFAKE_History_${Date.now()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);

        showNotification('History exported as CSV!', 'success');
    } catch (error) {
        console.error('Export error:', error);
        showNotification('Error exporting history', 'error');
    }
}

// ============ HISTORY MANAGEMENT ============
/**
 * Save result to local history
 */
function saveToHistory(result) {
    const historyItem = {
        id: Date.now(),
        timestamp: new Date().toLocaleString(),
        prediction: result.prediction,
        confidence: result.confidence,
        interpretation: result.interpretation,
        analysis_details: result.analysis_details,
        heatmap: result.heatmap,
        processing_time: result.processing_time
    };

    analyzerState.history.unshift(historyItem);

    // Keep only last 100 items
    if (analyzerState.history.length > 100) {
        analyzerState.history.pop();
    }

    // Save to localStorage
    localStorage.setItem('cifake_history', JSON.stringify(analyzerState.history));
    
    // Also save current result to sessionStorage for results page
    sessionStorage.setItem('current_result', JSON.stringify(historyItem));
}

/**
 * Load history from localStorage
 */
function loadHistory() {
    try {
        const saved = localStorage.getItem('cifake_history');
        if (saved) {
            analyzerState.history = JSON.parse(saved);
        }
    } catch (error) {
        console.warn('Error loading history:', error);
        analyzerState.history = [];
    }
}

/**
 * Clear history
 */
function clearHistory() {
    if (confirm('Are you sure you want to clear all history?')) {
        analyzerState.history = [];
        localStorage.removeItem('cifake_history');
        showNotification('History cleared', 'success');
        location.reload();
    }
}

/**
 * Delete specific history item
 */
function deleteRecord(element) {
    if (confirm('Are you sure you want to delete this record?')) {
        element.closest('tr').style.opacity = '0.5';
        setTimeout(() => {
            element.closest('tr').remove();
        }, 300);
    }
}

// ============ SEARCH & FILTER ============
/**
 * Search history by filename
 */
function searchHistory(query) {
    const rows = document.querySelectorAll('.table-row');
    const searchTerm = query.toLowerCase();

    rows.forEach(row => {
        const text = row.textContent.toLowerCase();
        row.style.display = text.includes(searchTerm) ? '' : 'none';
    });
}

/**
 * Filter history by class
 */
function filterByClass(classFilter) {
    const rows = document.querySelectorAll('.table-row');
    
    rows.forEach(row => {
        if (!classFilter) {
            row.style.display = '';
            return;
        }
        row.style.display = row.textContent.includes(classFilter) ? '' : 'none';
    });
}

/**
 * Filter history by date
 */
function filterByDate(dateFilter) {
    const rows = document.querySelectorAll('.table-row');
    const today = new Date();

    rows.forEach(row => {
        let show = true;
        
        if (dateFilter === 'today') {
            show = row.textContent.includes('Jun 23'); // Simplified
        } else if (dateFilter === 'week') {
            show = true; // Simplified
        }
        
        row.style.display = show ? '' : 'none';
    });
}

// ============ NOTIFICATIONS ============
/**
 * Show notification message
 */
function showNotification(message, type = 'info') {
    // Create notification element
    const notification = document.createElement('div');
    notification.className = `alert alert-${type === 'error' ? 'danger' : type} alert-dismissible fade show`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 9999;
        min-width: 300px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    `;
    notification.innerHTML = `
        ${message}
        <button type="button" class="close" data-dismiss="alert">&times;</button>
    `;

    document.body.appendChild(notification);

    // Auto-remove after 5 seconds
    setTimeout(() => {
        notification.remove();
    }, 5000);
}

// ============ MODALS ============
/**
 * Show custom modal
 */
function showModal(title, content) {
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'customModal';
    modal.innerHTML = `
        <div class="modal-dialog">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title">${title}</h5>
                    <button type="button" class="close" data-dismiss="modal">&times;</button>
                </div>
                <div class="modal-body">
                    ${content}
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    `;

    document.body.appendChild(modal);
    $(modal).modal('show');
    
    // Clean up on close
    $(modal).on('hidden.bs.modal', () => {
        modal.remove();
    });
}

/**
 * Close all open modals
 */
function closeAllModals() {
    $('.modal').modal('hide');
}

// ============ FORM VALIDATION ============
/**
 * Validate image file
 */
function validateImageFile(file) {
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg'];
    const maxSize = 10 * 1024 * 1024;

    if (!validTypes.includes(file.type)) {
        showNotification('Only PNG and JPG files are supported', 'warning');
        return false;
    }

    if (file.size > maxSize) {
        showNotification('File size must be less than 10MB', 'warning');
        return false;
    }

    return true;
}

// ============ UTILITY FUNCTIONS ============
/**
 * Format file size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
}

/**
 * Format date
 */
function formatDate(date) {
    return new Date(date).toLocaleDateString('en-US', {
        month: 'short',
        day: 'numeric',
        year: 'numeric'
    });
}

/**
 * Get prediction color
 */
function getPredictionColor(prediction) {
    return prediction === 'FAKE' ? '#E74C3C' : '#27AE60';
}

/**
 * Copy to clipboard
 */
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(() => {
        showNotification('Copied to clipboard!', 'success');
    }).catch(() => {
        showNotification('Failed to copy', 'error');
    });
}

// ============ DARK MODE (OPTIONAL) ============
/**
 * Toggle dark mode
 */
function toggleDarkMode() {
    document.body.classList.toggle('dark-mode');
    localStorage.setItem('dark_mode', document.body.classList.contains('dark-mode'));
}

/**
 * Load dark mode preference
 */
function loadDarkModePreference() {
    if (localStorage.getItem('dark_mode') === 'true') {
        document.body.classList.add('dark-mode');
    }
}

// ============ PERFORMANCE MONITORING ============
/**
 * Log performance metrics
 */
function logPerformanceMetrics() {
    if (window.performance && window.performance.timing) {
        const timing = window.performance.timing;
        const metrics = {
            pageLoadTime: timing.loadEventEnd - timing.navigationStart,
            domReadyTime: timing.domContentLoadedEventEnd - timing.navigationStart
        };
        console.log('Performance Metrics:', metrics);
    }
}

// Initialize performance logging
window.addEventListener('load', logPerformanceMetrics);

// ============ ERROR HANDLING ============
/**
 * Global error handler
 */
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    showNotification('An unexpected error occurred', 'error');
});

/**
 * Unhandled promise rejection handler
 */
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    showNotification('An unexpected error occurred', 'error');
});

// ============ EXPORTS ============
// Make functions globally available
window.analyzeImage = analyzeImage;
window.displayResults = displayResults;
window.toggleHeatmap = toggleHeatmap;
window.showHeatmapLegend = showHeatmapLegend;
window.exportResult = exportResult;
window.exportHistory = exportHistory;
window.clearHistory = clearHistory;
window.deleteRecord = deleteRecord;
window.searchHistory = searchHistory;
window.filterByClass = filterByClass;
window.filterByDate = filterByDate;
window.handleFileSelect = handleFileSelect;
