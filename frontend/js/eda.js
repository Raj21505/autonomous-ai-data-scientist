let datasetId = null;
let targetColumn = null;
const API_BASE = (() => {
    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1';

    if (window.API_BASE_URL) {
        return window.API_BASE_URL;
    }

    if (isLocalHost && window.location.port && window.location.port !== '8000') {
        return 'http://localhost:8000';
    }

    return '';
})();

let sectionGraphs = {
    distribution: [],
    comparison: []
};
let renderedSections = {
    distribution: false,
    comparison: false
};

document.addEventListener('DOMContentLoaded', async () => {
    const urlParams = new URLSearchParams(window.location.search);
    datasetId = urlParams.get('id') || sessionStorage.getItem('dataset_id') || sessionStorage.getItem('datasetId');
    targetColumn = urlParams.get('target') || sessionStorage.getItem('targetColumn') || '';

    if (!datasetId) {
        showError('Missing dataset ID. Please start from the beginning.');
        return;
    }

    await generateGraphGallery();
});

function getImageUrl(category, imageName) {
    return `${API_BASE}/eda-image/${datasetId}/${category}/${imageName}`;
}

function getGraphSources(summary) {
    const distribution = [];
    const comparison = [];

    const distributions = summary.distributions || {};
    Object.keys(distributions.histograms || {}).forEach((feature) => {
        distribution.push({ title: `Distribution - ${feature}`, src: getImageUrl('distributions', `histograms::${feature}`) });
    });

    Object.keys(distributions.box_plots || {}).forEach((feature) => {
        distribution.push({ title: `Box Plot - ${feature}`, src: getImageUrl('distributions', `box_plots::${feature}`) });
    });

    Object.keys(distributions.value_counts || {}).forEach((feature) => {
        distribution.push({ title: `Value Counts - ${feature}`, src: getImageUrl('distributions', `value_counts::${feature}`) });
    });

    const missingData = summary.missing_data || {};
    if (missingData.heatmap) {
        comparison.push({ title: 'Missing Data Heatmap', src: getImageUrl('missing_data', 'heatmap') });
    }
    if (missingData.missing_pattern) {
        comparison.push({ title: 'Missing Pattern Correlations', src: getImageUrl('missing_data', 'missing_pattern') });
    }

    const cat = summary.categorical_analysis || {};
    if (cat.target_plot) {
        distribution.push({ title: 'Target Distribution', src: getImageUrl('categorical_analysis', 'target_plot') });
    }

    const featureRelationships = summary.feature_relationships || {};
    if (featureRelationships.pairplot) {
        comparison.push({ title: 'Feature Pairplot', src: getImageUrl('feature_relationships', 'pairplot') });
    }

    Object.keys(featureRelationships.target_scatter || {}).forEach((feature) => {
        comparison.push({ title: `${feature} vs Target`, src: getImageUrl('feature_relationships', `target_scatter::${feature}`) });
    });

    Object.keys(featureRelationships.target_categorical || {}).forEach((feature) => {
        comparison.push({ title: `${feature} vs Target (Categorical)`, src: getImageUrl('feature_relationships', `target_categorical::${feature}`) });
    });

    if (summary.correlation_plot_base64) {
        comparison.push({ title: 'Correlation Matrix', src: `data:image/png;base64,${summary.correlation_plot_base64}` });
    }

    return { distribution, comparison };
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return String(text)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function renderGraphGallery(graphs, galleryId, emptyMessage) {
    const gallery = document.getElementById(galleryId);
    if (!gallery) return;
    gallery.innerHTML = '';

    if (!graphs.length) {
        gallery.innerHTML = `<div class="empty-state">${emptyMessage}</div>`;
        return;
    }

    const cards = graphs.map((graph) => {
        const safeTitle = escapeHtml(graph.title);
        return [
            '<div class="image-card">',
            `<h4>${safeTitle}</h4>`,
            `<img src="${graph.src}" alt="${safeTitle}" onerror="handleImageError(this)">`,
            `<button class="btn btn-primary" style="width: 100%; margin-top: 6px;" onclick="downloadImage('${graph.src}', '${safeTitle}')">⬇️ Download Graph</button>`,
            '</div>'
        ].join('');
    }).join('');

    gallery.innerHTML = cards;
}

async function generateGraphGallery() {
    const loadingDiv = document.getElementById('loading');
    const errorDiv = document.getElementById('errorMessage');

    loadingDiv.style.display = 'block';
    errorDiv.style.display = 'none';

    try {
        const formData = new URLSearchParams();
        formData.append('target', targetColumn || '');

        const response = await fetch(`${API_BASE}/eda/${datasetId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`${response.status} ${response.statusText}: ${errorText}`);
        }

        const data = await response.json();
        const summary = data.eda_summary || {};
        sectionGraphs = getGraphSources(summary);

        const distributionGallery = document.getElementById('distributionGallery');
        const comparisonGallery = document.getElementById('comparisonGallery');
        if (distributionGallery) {
            distributionGallery.innerHTML = '<div class="empty-state">Click "View Graphs" to load this section.</div>';
        }
        if (comparisonGallery) {
            comparisonGallery.innerHTML = '<div class="empty-state">Click "View Graphs" to load this section.</div>';
        }
    } catch (error) {
        showError(`Failed to generate EDA graphs: ${error.message}`);
    } finally {
        loadingDiv.style.display = 'none';
    }
}

function toggleSection(sectionName) {
    const section = document.getElementById(`${sectionName}Section`);
    if (!section) return;

    const isOpen = section.classList.contains('open');
    section.classList.toggle('open', !isOpen);

    const button = section.querySelector('.section-toggle');
    if (button) {
        button.textContent = isOpen ? 'View Graphs' : 'Hide Graphs';
    }

    if (!isOpen && !renderedSections[sectionName]) {
        const galleryId = sectionName === 'distribution' ? 'distributionGallery' : 'comparisonGallery';
        const emptyMessage = sectionName === 'distribution'
            ? 'No distribution graphs available for this dataset.'
            : 'No comparison graphs available for this dataset.';

        renderGraphGallery(sectionGraphs[sectionName] || [], galleryId, emptyMessage);
        renderedSections[sectionName] = true;
    }
}

function handleImageError(imgElement) {
    const card = imgElement.closest('.image-card');
    if (!card) return;
    imgElement.remove();
    const note = document.createElement('p');
    note.textContent = 'Graph not available';
    note.style.color = '#b91c1c';
    note.style.margin = '8px 0';
    card.insertBefore(note, card.querySelector('button'));
}

async function downloadImage(src, title) {
    try {
        const response = await fetch(src);
        if (!response.ok) throw new Error('Unable to fetch graph image');

        const blob = await response.blob();
        const url = URL.createObjectURL(blob);

        const link = document.createElement('a');
        link.href = url;
        link.download = `${title.replace(/[^a-zA-Z0-9-_ ]/g, '').trim().replace(/\s+/g, '_') || 'eda_graph'}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();

        URL.revokeObjectURL(url);
    } catch (error) {
        alert(`Download failed: ${error.message}`);
    }
}

async function downloadEDAReport() {
    const container = document.querySelector('.eda-container');
    if (!container) return;

    if (typeof html2canvas === 'undefined') {
        alert('Unable to download full EDA right now. html2canvas library is not loaded.');
        return;
    }

    try {
        const canvas = await html2canvas(container, {
            scale: 2,
            backgroundColor: '#f5f5f7',
            useCORS: true,
            allowTaint: true
        });

        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `EDA_Full_${datasetId}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
    } catch (error) {
        alert(`Full EDA download failed: ${error.message}`);
    }
}

async function downloadSection(sectionName) {
    const targetId = sectionName === 'distribution' ? 'distributionSection' : 'comparisonSection';
    const container = document.getElementById(targetId);
    if (!container) return;

    if (typeof html2canvas === 'undefined') {
        alert('Unable to download section right now. html2canvas library is not loaded.');
        return;
    }

    try {
        const canvas = await html2canvas(container, {
            scale: 2,
            backgroundColor: '#f5f5f7',
            useCORS: true,
            allowTaint: true
        });

        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `EDA_${sectionName}_${datasetId}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
    } catch (error) {
        alert(`Section download failed: ${error.message}`);
    }
}

function showError(message) {
    const errorDiv = document.getElementById('errorMessage');
    errorDiv.textContent = message;
    errorDiv.style.display = 'block';
}

function goBack() {
    window.location.href = 'target.html';
}
