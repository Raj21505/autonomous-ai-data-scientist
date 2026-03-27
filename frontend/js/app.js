const API_BASE = (() => {
    const host = window.location.hostname;
    const isLocalHost = host === 'localhost' || host === '127.0.0.1';

    if (window.API_BASE_URL) {
        return window.API_BASE_URL;
    }

    // When running frontend on a separate local static server, keep API on 8000.
    if (isLocalHost && window.location.port && window.location.port !== '8000') {
        return 'http://localhost:8000';
    }

    return '';
})();

function initFileUploadBox(inputId, filenameId, emptyText) {
    const input = document.getElementById(inputId);
    const nameNode = document.getElementById(filenameId);
    const shell = input ? input.closest('.file-upload')?.querySelector('.file-upload-shell') : null;

    if (!input || !nameNode) return;

    const updateName = () => {
        const hasFile = input.files && input.files.length > 0;
        nameNode.textContent = hasFile ? input.files[0].name : (emptyText || 'No file selected');
        nameNode.classList.toggle('has-file', hasFile);
    };

    input.addEventListener('change', updateName);
    updateName();

    if (!shell) return;

    shell.addEventListener('dragover', (event) => {
        event.preventDefault();
        shell.classList.add('is-dragover');
    });

    shell.addEventListener('dragleave', () => {
        shell.classList.remove('is-dragover');
    });

    shell.addEventListener('drop', (event) => {
        event.preventDefault();
        shell.classList.remove('is-dragover');

        const droppedFiles = event.dataTransfer?.files;
        if (!droppedFiles || !droppedFiles.length) return;

        const transfer = new DataTransfer();
        transfer.items.add(droppedFiles[0]);
        input.files = transfer.files;
        input.dispatchEvent(new Event('change'));
    });
}

function ensureGlobalLoader() {
    let overlay = document.getElementById('globalLoaderOverlay');
    if (overlay) return overlay;

    overlay = document.createElement('div');
    overlay.id = 'globalLoaderOverlay';
    overlay.className = 'global-loader-overlay';
    overlay.innerHTML = [
        '<div class="global-loader-card" role="status" aria-live="polite" aria-busy="true">',
        '<div class="global-loader-spinner" aria-hidden="true"></div>',
        '<div class="global-loader-title" id="globalLoaderMessage">Processing request</div>',
        '<div class="global-loader-subtitle">Please wait while we prepare your results</div>',
        '<div class="global-loader-dots" aria-hidden="true"><span></span><span></span><span></span></div>',
        '</div>'
    ].join('');

    document.body.appendChild(overlay);
    return overlay;
}

function setGlobalLoader(active, message) {
    const overlay = ensureGlobalLoader();
    const messageNode = document.getElementById('globalLoaderMessage');
    if (messageNode && message) {
        messageNode.textContent = message;
    }
    overlay.classList.toggle('active', Boolean(active));
    document.body.style.overflow = active ? 'hidden' : '';
}

async function uploadDataset() {
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    if (!file) return alert('Please select a dataset file');

    const form = new FormData();
    form.append('file', file);

    setGlobalLoader(true, 'Uploading and analyzing your dataset...');

    try {
        const res = await fetch(`${API_BASE}/upload`, {
            method: 'POST',
            body: form
        });

        if (!res.ok) {
            alert('Upload failed');
            return;
        }

        const payload = await res.json();
        const id = payload.id;
        const analysis = payload.analysis;

        sessionStorage.removeItem('clean_report');
        sessionStorage.removeItem('model_target');
        sessionStorage.removeItem('model_active_tab');
        sessionStorage.removeItem('targetColumn');
        sessionStorage.setItem('dataset_id', id);
        sessionStorage.setItem('uploaded_analysis', JSON.stringify(analysis || {}));
        sessionStorage.setItem('last_page', 'cleaning.html');
        window.location.href = 'cleaning.html';
    } catch (error) {
        alert(`Upload failed: ${error.message}`);
    } finally {
        setGlobalLoader(false);
    }
}

function renderAnalysis(analysis) {
    const div = document.getElementById('analysis');
    div.innerHTML = '';

    const html = [];

    html.push('<div class="action-row">');
    html.push('<button class="secondary" onclick="goToUploadHome()">← Back to Upload</button>');
    html.push('</div>');
    
    html.push('<h2 class="page-section-title">Dataset Overview</h2>');
    
    html.push('<div class="content-grid">');
    html.push(`<div class="stat-card"><div class="stat-label">Rows</div><div class="stat-value">${analysis.rows}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Columns</div><div class="stat-value">${analysis.columns}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Duplicates</div><div class="stat-value">${analysis.duplicates}</div></div>`);
    html.push('</div>');

    const missingCounts = analysis.missing_counts || {};
    const missingColumns = Object.entries(missingCounts).filter(([, value]) => Number(value) > 0);
    if (missingColumns.length > 0) {
        html.push('<h3 class="section-spaced">Missing Values</h3>');
        html.push('<div class="content-grid">');
        missingColumns.forEach(([column, value]) => {
            html.push(`<div class="stat-card"><div class="stat-label">${column}</div><div class="stat-value">${value}</div></div>`);
        });
        html.push('</div>');
    }

    html.push('<div class="content-grid">');
    html.push('<div><h3>Numeric Features</h3><div class="grid-item">');
    (analysis.numeric_features || []).forEach(c => {
        html.push(`<div class="clean-list-item">${c}</div>`);
    });
    html.push('</div></div>');
    
    html.push('<div><h3>Categorical Features</h3><div class="grid-item">');
    (analysis.categorical_features || []).forEach(c => {
        html.push(`<div class="clean-list-item">${c}</div>`);
    });
    html.push('</div></div>');
    html.push('</div>');

    html.push('<div class="card section-spaced">');
    html.push('<label>Select Target Column</label>');
    html.push('<div class="form-split">');
    html.push('<select id="targetSelect"></select>');
    html.push('<button id="cleanBtn" onclick="runCleaning()">Analyze</button>');
    html.push('</div>');
    html.push('</div>');

    html.push('<h3 class="section-spaced">Data Preview (10 Rows)</h3>');
    html.push(renderTable(analysis.sample));

    div.innerHTML = html.join('');

    const sel = document.getElementById('targetSelect');
    (analysis.column_names || []).forEach(c => {
        const opt = document.createElement('option');
        opt.value = c;
        opt.text = c;
        sel.appendChild(opt);
    });

    const savedTarget = sessionStorage.getItem('targetColumn');
    if (savedTarget && (analysis.column_names || []).includes(savedTarget)) {
        sel.value = savedTarget;
    }

    sel.addEventListener('change', () => {
        sessionStorage.setItem('targetColumn', sel.value || '');
    });
}

function renderTable(rows) {
    if (!rows || rows.length === 0) return '<p class="text-muted">No preview available</p>';
    
    const cols = Object.keys(rows[0]);
    let html = '<div class="table-wrap"><table><thead><tr>';
    
    cols.forEach(c => html += `<th>${escapeHtml(c)}</th>`);
    html += '</tr></thead><tbody>';
    
    rows.forEach((r, idx) => {
        html += '<tr>';
        cols.forEach(c => html += `<td>${escapeHtml(r[c])}</td>`);
        html += '</tr>';
    });
    
    html += '</tbody></table></div>';
    return html;
}

function escapeHtml(text) {
    if (text === null || text === undefined) return '';
    return (''+text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function renderCleaningExplanations(explanations, reportData) {
    explanations = explanations || {};

    const columnsRemoved = explanations.columns_removed || [];
    const rowsRemoved = explanations.rows_removed || [];
    const missingFilled = explanations.missing_filled || [];
    const summary = explanations.summary || [];
    const removedColumnNames = columnsRemoved
        .map((item) => item && item.column)
        .filter(Boolean);
    const fallbackRemovedNames = Array.isArray(reportData && reportData.cols_removed)
        ? reportData.cols_removed.filter(Boolean)
        : [];
    const allRemovedNames = Array.from(new Set([...removedColumnNames, ...fallbackRemovedNames]));

    const getSummaryLine = (keyword) => {
        const match = summary.find((item) => (item || '').toLowerCase().includes(keyword.toLowerCase()));
        return match || '';
    };

    const extractCount = (line) => {
        const match = (line || '').match(/(\d+)/g);
        if (!match || !match.length) return null;
        return Number(match[match.length - 1]);
    };

    const duplicatesLine = getSummaryLine('duplicates removed');
    const outliersLine = getSummaryLine('outliers detected and handled');
    const invalidLine = getSummaryLine('invalid values removed');
    const rareLine = getSummaryLine('rare categories handled');
    const shapeLine = getSummaryLine('initial shape');
    const rowsShapeLine = getSummaryLine('rows:');
    const colsShapeLine = getSummaryLine('columns:');

    const duplicatesRemoved = Number(reportData && reportData.duplicates_removed);
    const outlierColumns = extractCount(outliersLine);
    const invalidColumns = extractCount(invalidLine);
    const rareColumns = extractCount(rareLine);
    const rowsRemovedCount = Number(reportData && reportData.rows_removed);
    const colsRemovedCount = Number(reportData && reportData.cols_removed_count);

    const rowRemovalNotes = Number.isFinite(rowsRemovedCount) ? rowsRemovedCount : rowsRemoved.length;
    const colRemovalNotes = Number.isFinite(colsRemovedCount) ? colsRemovedCount : allRemovedNames.length;

    const removedItems = [];
    removedItems.push(`${colRemovalNotes || 0} column(s) removed`);
    removedItems.push(`Duplicates removed: ${Number.isFinite(duplicatesRemoved) ? duplicatesRemoved : 0}`);
    if (rowRemovalNotes > 0) {
        removedItems.push(`Rows removed: ${rowRemovalNotes}`);
    }
    if (allRemovedNames.length > 0) {
        removedItems.push(`Removed column name${allRemovedNames.length > 1 ? 's' : ''}: ${allRemovedNames.join(', ')}`);
    }

    let filledMessage = 'No missing values needed filling.';
    if (missingFilled.length === 1 && missingFilled[0] && missingFilled[0].details) {
        filledMessage = missingFilled[0].details;
    } else if (missingFilled.length > 1) {
        const columns = missingFilled
            .map((item) => item.column)
            .filter(Boolean)
            .slice(0, 3);
        if (columns.length > 0) {
            filledMessage = `Filled missing values in ${missingFilled.length} columns (e.g., ${columns.join(', ')}).`;
        } else {
            filledMessage = `Filled missing values in ${missingFilled.length} columns.`;
        }
    }

    const fixParts = [];
    if (outlierColumns !== null) fixParts.push(`Outliers handled in ${outlierColumns} column(s)`);
    if (invalidColumns !== null) fixParts.push(`Invalid values cleaned in ${invalidColumns} column(s)`);
    if (rareColumns !== null) fixParts.push(`Rare categories grouped in ${rareColumns} column(s)`);
    const fixesItems = fixParts.length ? fixParts : ['No extra data quality fixes were required'];

    const finalSizeItems = [];
    if (shapeLine) finalSizeItems.push(shapeLine);
    if (rowsShapeLine) finalSizeItems.push(rowsShapeLine);
    if (colsShapeLine) finalSizeItems.push(colsShapeLine);
    if (!finalSizeItems.length && Number.isFinite(reportData && reportData.rows_before)) {
        finalSizeItems.push(`Rows: ${reportData.rows_before} -> ${reportData.rows_after}`);
    }
    if (!finalSizeItems.length && Number.isFinite(reportData && reportData.cols_before)) {
        finalSizeItems.push(`Columns: ${reportData.cols_before} -> ${reportData.cols_after}`);
    }
    if (!finalSizeItems.length) {
        finalSizeItems.push('Final dataset size information is not available.');
    }

    const renderSummaryCard = (icon, title, items) => {
        const safeItems = (items || []).filter(Boolean);
        return [
            '<article class="cleaning-summary-card">',
            '<div class="cleaning-summary-card-header">',
            `<div class="cleaning-summary-icon">${escapeHtml(icon)}</div>`,
            `<h4 class="cleaning-summary-card-title">${escapeHtml(title)}</h4>`,
            '</div>',
            '<div class="cleaning-summary-card-body">',
            safeItems.map((line) => `<div class="cleaning-summary-line">${escapeHtml(line)}</div>`).join(''),
            '</div>',
            '</article>'
        ].join('');
    };

    const summaryLines = [];
    summary.forEach((line) => {
        if (line) summaryLines.push(String(line));
    });
    rowsRemoved.forEach((line) => {
        if (line) summaryLines.push(String(line));
    });
    columnsRemoved.forEach((item) => {
        if (!item) return;
        if (item.details) {
            summaryLines.push(String(item.details));
            return;
        }
        if (item.column) {
            const reason = item.reason ? ` (${item.reason})` : '';
            summaryLines.push(`Column '${item.column}' removed${reason}.`);
        }
    });
    if (allRemovedNames.length > 0) {
        summaryLines.push(`All removed columns: ${allRemovedNames.join(', ')}`);
    }
    missingFilled.forEach((item) => {
        if (item && item.details) {
            summaryLines.push(String(item.details));
        }
    });

    const uniqueSummaryLines = [];
    const seenSummaryLines = new Set();
    summaryLines.forEach((line) => {
        const normalized = (line || '').trim();
        if (!normalized || seenSummaryLines.has(normalized)) return;
        seenSummaryLines.add(normalized);
        uniqueSummaryLines.push(normalized);
    });

    const html = [];
    html.push('<section class="cleaning-summary-shell card">');
    html.push('<h3 class="cleaning-summary-title">Data Cleaning Summary</h3>');
    html.push('<div class="cleaning-summary-grid">');
    html.push(renderSummaryCard('🗑️', 'Removed', removedItems));
    html.push(renderSummaryCard('🧽', 'Missing Values Filled', [filledMessage]));
    html.push(renderSummaryCard('✅', 'Data Quality Fixes', fixesItems));
    html.push(renderSummaryCard('🧾', 'Final Dataset Size', finalSizeItems));
    html.push('</div>');

    html.push('</section>');
    return html.join('');
}

async function runCleaning() {
    const id = sessionStorage.getItem('dataset_id');
    if (!id) return alert('No dataset found');
    
    const target = document.getElementById('targetSelect').value;
    if (!target) return alert('Select a target column');

    sessionStorage.setItem('targetColumn', target);

    const cleanBtn = document.getElementById('cleanBtn');
    if (cleanBtn) {
        cleanBtn.disabled = true;
    }

    const form = new FormData();
    form.append('id', id);
    form.append('target', target);

    setGlobalLoader(true, 'Cleaning data and preparing report...');

    try {
        const res = await fetch(`${API_BASE}/clean`, {
            method: 'POST',
            body: form
        });

        if (!res.ok) {
            alert('Cleaning failed');
            return;
        }
        const data = await res.json();

        sessionStorage.setItem('clean_report', JSON.stringify(data || {}));
        sessionStorage.setItem('last_page', 'target.html');
        window.location.href = 'target.html';
    } catch (error) {
        alert(`Cleaning failed: ${error.message}`);
    } finally {
        setGlobalLoader(false);
        if (cleanBtn) {
            cleanBtn.disabled = false;
        }
    }
}

function renderCleaningReport(data) {
    const div = document.getElementById('analysis');
    if (!div) return;

    const html = [];
    html.push('<div class="action-row">');
    html.push('<button class="secondary" onclick="goToOverview()">← Back to Dataset Overview</button>');
    html.push('</div>');
    html.push('<h2 class="page-section-title">Cleaning Report</h2>');
    html.push('<h3>Dataset Transformation</h3>');
    html.push('<div class="content-grid">');
    html.push(`<div class="stat-card"><div class="stat-label">Rows Before</div><div class="stat-value">${data.rows_before}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Rows After</div><div class="stat-value">${data.rows_after}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Rows Removed</div><div class="stat-value danger-text">${data.rows_removed}</div></div>`);
    html.push('</div>');

    html.push('<div class="content-grid">');
    html.push(`<div class="stat-card"><div class="stat-label">Columns Before</div><div class="stat-value">${data.cols_before}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Columns After</div><div class="stat-value">${data.cols_after}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Total Missing</div><div class="stat-value">${data.total_missing_cells}</div></div>`);
    html.push(`<div class="stat-card"><div class="stat-label">Imputed</div><div class="stat-value success-text">${data.imputed_cells}</div></div>`);
    html.push('</div>');

    html.push('<h3>Imputation Methods</h3>');
    html.push('<div class="content-grid">');
    for (let m in data.imputed_by_method) {
        html.push(`<div class="stat-card"><div class="stat-label">${m}</div><div class="stat-value">${data.imputed_by_method[m]}</div></div>`);
    }
    html.push('</div>');

    html.push(renderCleaningExplanations(data.cleaning_explanations, data));

    html.push('<h3 class="section-spaced">Cleaned Data Sample (10 Rows)</h3>');
    html.push(renderTable((data.sample || []).slice(0, 10)));

    html.push('<div class="action-row section-spaced">');
    html.push('<button id="showEdaBtn" class="info" onclick="showEDA()">📊 View EDA</button>');
    html.push('<button id="seeFull" class="secondary" onclick="downloadData()">💾 Download Data</button>');
    html.push('<button id="trainModelsBtn" class="success" onclick="goToModels()">🤖 Train Models</button>');
    html.push('</div>');

    div.innerHTML = html.join('');
}

function downloadData() {
    const id = sessionStorage.getItem('dataset_id');
    const url = `${API_BASE}/data/${id}?full=1`;
    fetch(url)
        .then(r => r.blob())
        .then(blob => {
            const a = document.createElement('a');
            a.href = URL.createObjectURL(blob);
            a.download = `cleaned_${id}.csv`;
            document.body.appendChild(a);
            a.click();
            a.remove();
        })
        .catch(err => alert('Download failed: ' + err.message));
}

function goToModels() {
    sessionStorage.setItem('last_page', 'target.html');
    window.location.href = 'model.html';
}

function showEDA() {
    const id = sessionStorage.getItem('dataset_id');
    const target = sessionStorage.getItem('targetColumn');
    if (!id || !target) return alert('Dataset not configured');
    sessionStorage.setItem('last_page', 'target.html');
    window.location.href = `eda.html?id=${id}&target=${encodeURIComponent(target)}`;
}

function goToOverview() {
    window.location.href = 'cleaning.html';
}

function goToUploadHome() {
    window.location.href = 'index.html';
}

function continueWorkflow() {
    const hasReport = !!sessionStorage.getItem('clean_report');
    const hasOverview = !!sessionStorage.getItem('uploaded_analysis');

    if (hasReport) {
        window.location.href = 'target.html';
        return;
    }

    if (hasOverview) {
        window.location.href = 'cleaning.html';
        return;
    }

    alert('No saved workflow found. Please upload a dataset first.');
}

function renderResumeCard() {
    const div = document.getElementById('analysis');
    if (!div) return;

    const datasetId = sessionStorage.getItem('dataset_id');
    const hasOverview = !!sessionStorage.getItem('uploaded_analysis');
    const hasReport = !!sessionStorage.getItem('clean_report');

    if (!datasetId || (!hasOverview && !hasReport)) {
        div.innerHTML = '';
        return;
    }

    const step = hasReport ? 'Cleaning report ready' : 'Dataset overview ready';
    div.innerHTML = [
        '<div class="card section-spaced">',
        '<h3>Resume Previous Workflow</h3>',
        `<p><strong>Dataset ID:</strong> ${escapeHtml(datasetId)}</p>`,
        `<p><strong>Status:</strong> ${escapeHtml(step)}</p>`,
        '<div class="action-row">',
        '<button class="info" onclick="continueWorkflow()">▶ Continue</button>',
        '</div>',
        '</div>'
    ].join('');
}

document.addEventListener('DOMContentLoaded', () => {
    const page = (window.location.pathname.split('/').pop() || 'index.html').toLowerCase();

    if (page === 'index.html' || page === '') {
        renderResumeCard();
        initFileUploadBox('fileInput', 'fileInputName', 'No dataset selected');
    }

    if (page === 'cleaning.html') {
        const rawAnalysis = sessionStorage.getItem('uploaded_analysis');
        if (!rawAnalysis) {
            alert('No uploaded dataset found. Please upload again.');
            window.location.href = 'index.html';
            return;
        }

        try {
            const analysis = JSON.parse(rawAnalysis);
            renderAnalysis(analysis);
        } catch (err) {
            alert('Could not load dataset analysis. Please upload again.');
            window.location.href = 'index.html';
        }
    }

    if (page === 'target.html') {
        const rawReport = sessionStorage.getItem('clean_report');
        if (!rawReport) {
            alert('No analysis report found. Please run analysis first.');
            window.location.href = 'cleaning.html';
            return;
        }

        try {
            const report = JSON.parse(rawReport);
            renderCleaningReport(report);
        } catch (err) {
            alert('Could not load analysis report. Please run analysis again.');
            window.location.href = 'cleaning.html';
        }
    }

    const uploadBtn = document.getElementById('uploadBtn');
    if (uploadBtn) uploadBtn.addEventListener('click', uploadDataset);
});
