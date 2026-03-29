// Get dataset ID from session storage
const datasetId = sessionStorage.getItem('dataset_id');
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

function setPostTrainingTabsVisible(isVisible) {
    const displayValue = isVisible ? '' : 'none';
    const ids = ['tabResults', 'tabDashboard', 'tabPrediction'];
    ids.forEach((id) => {
        const btn = document.getElementById(id);
        if (btn) {
            btn.style.display = displayValue;
        }
    });
}

function getTrainingStateKey() {
    return `model_trained_${datasetId}`;
}

async function isModelTrained() {
    if (!datasetId) return false;

    if (sessionStorage.getItem(getTrainingStateKey()) === '1') {
        return true;
    }

    try {
        const response = await fetch(`${API_BASE}/model-results/${datasetId}`);
        if (!response.ok) return false;

        sessionStorage.setItem(getTrainingStateKey(), '1');
        return true;
    } catch (error) {
        return false;
    }
}

if (!datasetId) {
    alert('No dataset found. Please upload a dataset first.');
    window.location.href = 'index.html';
}

// Initialize page - fetch column names
async function initializeML() {
    try {
        const trained = await isModelTrained();
        setPostTrainingTabsVisible(trained);

        const response = await fetch(`${API_BASE}/data/${datasetId}`);
        if (!response.ok) throw new Error('Failed to fetch data');
        
        const data = await response.json();
        const columns = Object.keys(data.sample[0] || {});
        
        // Populate select dropdown
        const select = document.getElementById('targetSelectML');
        columns.forEach(col => {
            const option = document.createElement('option');
            option.value = col;
            option.textContent = col;
            select.appendChild(option);
        });

        const savedTarget = sessionStorage.getItem('model_target') || sessionStorage.getItem('targetColumn');
        if (savedTarget && columns.includes(savedTarget)) {
            select.value = savedTarget;
            await showTargetInfo();
        }
        
        // Add change event listener to persist selection and refresh target hints
        select.addEventListener('change', async () => {
            const selectedTarget = select.value || '';
            sessionStorage.setItem('model_target', selectedTarget);
            if (selectedTarget) {
                sessionStorage.setItem('targetColumn', selectedTarget);
            }
            await showTargetInfo();
        });

        const lastActiveTab = sessionStorage.getItem('model_active_tab');
        const allowedTab = trained ? lastActiveTab : 'training';
        if (allowedTab && document.getElementById(allowedTab)) {
            activateTab(allowedTab);
        } else {
            activateTab('training');
        }

        if (trained && savedTarget) {
            try {
                await loadModelResults();
                enablePredictionUI();
                loadInteractiveDashboardSchema(false);
            } catch (err) {
                // Keep init robust even when no model has been trained yet.
            }
        }
    } catch (error) {
        console.error('Error initializing:', error);
        alert('Error loading dataset columns');
    }
}

// Show info about selected target
async function showTargetInfo() {
    const target = document.getElementById('targetSelectML').value;
    const hintDiv = document.getElementById('targetHint');
    
    if (!target) {
        hintDiv.style.display = 'none';
        return;
    }
    
    try {
        const response = await fetch(`${API_BASE}/data/${datasetId}`);
        if (!response.ok) throw new Error('Failed to fetch data');
        
        const data = await response.json();
        const values = data.sample.map(row => row[target]).filter(v => v !== null && v !== undefined);
        
        // Get unique values
        const unique = new Set(values);
        const uniqueCount = unique.size;
        
        // Check if values are numeric
        const isNumeric = values.every(v => !isNaN(v) && v !== '');
        
        // Create hint text
        let hint = `${uniqueCount} unique values`;
        let isClassification = false;
        
        if (uniqueCount === 1) {
            hint += ' ❌ Invalid - all values are the same.';
            hintDiv.style.backgroundColor = '#ffebee';
            hintDiv.style.color = '#c62828';
        } else if (isNumeric && uniqueCount <= 20) {
            // Likely classification (discrete integer values)
            hint += ' ✓ Good for Classification';
            hintDiv.style.backgroundColor = '#e8f5e9';
            hintDiv.style.color = '#2e7d32';
            isClassification = true;
        } else if (isNumeric && uniqueCount > 20) {
            // Likely regression (many continuous values)
            hint += ' ✓ Good for Regression (continuous values)';
            hintDiv.style.backgroundColor = '#e8f5e9';
            hintDiv.style.color = '#2e7d32';
        } else {
            // Categorical text values
            if (uniqueCount >= 2 && uniqueCount <= 10) {
                hint += ' ✓ Perfect for Classification';
                hintDiv.style.backgroundColor = '#e8f5e9';
                hintDiv.style.color = '#2e7d32';
                isClassification = true;
            } else {
                hint += ' ⚠️ Many categories - may work but classification models prefer 2-10 classes.';
                hintDiv.style.backgroundColor = '#fff3e0';
                hintDiv.style.color = '#e65100';
            }
        }
        
        document.getElementById('targetInfo').textContent = hint;
        hintDiv.style.display = 'block';
    } catch (error) {
        console.error('Error getting target info:', error);
    }
}

// Train models
async function trainModels() {
    const trainBtn = document.getElementById('trainModelsBtn');
    const target = document.getElementById('targetSelectML').value;
    
    if (!target) {
        alert('Please select a target column');
        return;
    }

    sessionStorage.setItem('model_target', target);
    sessionStorage.setItem('targetColumn', target);

    if (trainBtn) {
        trainBtn.disabled = true;
    }
    setGlobalLoader(true, 'Training models and evaluating metrics...');
    
    // Show loading state
    document.getElementById('trainingStatus').style.display = 'block';
    document.getElementById('trainingResults').style.display = 'none';
    
    const formData = new FormData();
    formData.append('id', datasetId);
    formData.append('target', target);
    
    try {
        const response = await fetch(`${API_BASE}/train-models`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            let errorMsg = error.detail || 'Training failed';
            throw new Error(errorMsg);
        }
        
        const results = await response.json();
        
        // Hide loading
        document.getElementById('trainingStatus').style.display = 'none';
        
        // Display results
        displayTrainingResults(results);

        sessionStorage.setItem(getTrainingStateKey(), '1');
        setPostTrainingTabsVisible(true);
        
        // Enable other tabs
        enablePredictionUI();
        
    } catch (error) {
        document.getElementById('trainingStatus').style.display = 'none';
        
        // Show detailed error message
        const errorDiv = document.createElement('div');
        errorDiv.className = 'card';
        errorDiv.style.borderLeft = '4px solid #f44336';
        errorDiv.style.backgroundColor = '#ffebee';
        
        const errorTitle = document.createElement('h3');
        errorTitle.textContent = '❌ Training Error';
        errorTitle.style.color = '#c62828';
        
        const errorText = document.createElement('pre');
        errorText.style.whiteSpace = 'pre-wrap';
        errorText.style.wordWrap = 'break-word';
        errorText.style.color = '#333';
        errorText.style.fontSize = '14px';
        errorText.textContent = error.message;
        
        const statusDiv = document.getElementById('trainingStatus');
        statusDiv.parentNode.insertBefore(errorDiv, statusDiv.nextSibling);
        errorDiv.appendChild(errorTitle);
        errorDiv.appendChild(errorText);
    } finally {
        setGlobalLoader(false);
        if (trainBtn) {
            trainBtn.disabled = false;
        }
    }
}

// Display training results
async function displayTrainingResults(results) {
    const html = [];
    const taskType = results.task_type || 'classification';
    const isRegression = taskType === 'regression';
    
    // Task type indicator
    const taskLabel = isRegression ? 'Regression' : 'Classification';
    const taskIcon = isRegression ? '📊' : '🎯';
    html.push(`<p style="color: #0071e3; font-weight: bold; margin-bottom: 15px;">${taskIcon} Task Type: ${taskLabel}</p>`);
    
    // Best model info
    if (isRegression) {
        // Regression metrics
        html.push(`
            <div class="model-section">
                <h2>🏆 Best Model: ${results.best_model}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div class="metric-box">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">${(results.best_model_metrics.r2_score).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Mean Absolute Error</div>
                        <div class="metric-value">${(results.best_model_metrics.mae).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Root Mean Squared Error</div>
                        <div class="metric-value">${(results.best_model_metrics.rmse).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Mean Absolute % Error</div>
                        <div class="metric-value">${(results.best_model_metrics.mape * 100).toFixed(2)}%</div>
                    </div>
                </div>
            </div>
        `);
    } else {
        // Classification metrics
        html.push(`
            <div class="model-section">
                <h2>🏆 Best Model: ${results.best_model}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div class="metric-box">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">${(results.best_model_metrics.accuracy * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">${(results.best_model_metrics.precision * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">${(results.best_model_metrics.recall * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value">${(results.best_model_metrics.f1_score * 100).toFixed(2)}%</div>
                    </div>
                </div>
            </div>
        `);
    }
    
    // Model comparison table
    html.push('<div class="model-comparison">');
    html.push('<h3>Model Comparison</h3>');
    html.push('<table>');
    
    if (isRegression) {
        html.push('<thead><tr><th>Model</th><th>R² Score</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead>');
        html.push('<tbody>');
        
        results.comparison.forEach(model => {
            const isbest = model.model === results.best_model ? 'best' : '';
            html.push(`
                <tr class="${isbest}">
                    <td>${model.model}</td>
                    <td>${(model.r2_score).toFixed(4)}</td>
                    <td>${(model.mae).toFixed(4)}</td>
                    <td>${(model.rmse).toFixed(4)}</td>
                    <td>${(model.mape * 100).toFixed(2)}%</td>
                </tr>
            `);
        });
    } else {
        html.push('<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>');
        html.push('<tbody>');
        
        results.comparison.forEach(model => {
            const isbest = model.model === results.best_model ? 'best' : '';
            html.push(`
                <tr class="${isbest}">
                    <td>${model.model}</td>
                    <td>${(model.accuracy * 100).toFixed(2)}%</td>
                    <td>${(model.precision * 100).toFixed(2)}%</td>
                    <td>${(model.recall * 100).toFixed(2)}%</td>
                    <td>${(model.f1_score * 100).toFixed(2)}%</td>
                </tr>
            `);
        });
    }
    
    html.push('</tbody></table></div>');
    
    // Display preparation info
    const prepInfo = results.preparation;
    html.push(`
        <div class="card section-spaced">
            <h3>Data Preparation</h3>
            <p><strong>Total Samples:</strong> ${prepInfo.total_samples}</p>
            <p><strong>Training Samples:</strong> ${prepInfo.train_samples}</p>
            <p><strong>Test Samples:</strong> ${prepInfo.test_samples}</p>
            <p><strong>Features Used:</strong> ${prepInfo.n_features}</p>
            ${isRegression ? 
                `<p><strong>Target Range:</strong> ${prepInfo.target_min?.toFixed(2) || 'N/A'} - ${prepInfo.target_max?.toFixed(2) || 'N/A'}</p>` :
                `<p><strong>Target Classes:</strong> ${prepInfo.target_classes?.join(', ') || 'N/A'}</p>`
            }
        </div>
    `);
    
    document.getElementById('trainingDetails').innerHTML = html.join('');
    document.getElementById('trainingResults').style.display = 'block';

    renderInteractiveDashboard(results.dashboard_schema);
    
    // Also load results in the Results tab
    loadModelResults();
}

function renderInteractiveDashboard(schema) {
    const status = document.getElementById('interactiveDashboardStatus');
    const container = document.getElementById('interactiveDashboardContainer');
    const kpiGrid = document.getElementById('dashboardKpiGrid');
    const chartsRow2 = document.getElementById('dashboardChartsRow2');
    const chartsRow3 = document.getElementById('dashboardChartsRow3');
    const palette = ['#0071e3', '#4f84c4', '#7fa8d9', '#6e6e73', '#9ca3af', '#4b5563'];

    if (!status || !container || !kpiGrid || !chartsRow2 || !chartsRow3) {
        return;
    }

    if (!schema) {
        status.textContent = 'Dashboard schema is not available yet. Train model or refresh dashboard.';
        container.style.display = 'none';
        return;
    }

    status.textContent = `Loaded dashboard for target: ${schema.target || 'N/A'} (${schema.task_type || 'unknown'})`;
    container.style.display = 'block';

    const kpis = (schema.kpis || []).slice(0, 3);
    kpiGrid.innerHTML = kpis.map((k, idx) => `
        <div class="dashboard-kpi-card kpi-${idx + 1}">
            <div class="dashboard-kpi-label">${k.label || ''}</div>
            <div class="dashboard-kpi-value">${k.value ?? ''}${k.suffix || ''}</div>
        </div>
    `).join('');

    const charts = schema.charts || [];
    const row2Charts = charts.slice(0, 3);
    const row3Charts = charts.slice(3, 7);

    const renderCards = (chartList, startIndex) => chartList.map((chart, offset) => {
        const i = startIndex + offset;
        return `
        <div class="dashboard-chart-card ${chart.span === 2 ? 'span-2' : ''}">
            <h4 class="dashboard-chart-title">${chart.title || `Chart ${i + 1}`}</h4>
            <div id="dashboard-chart-${i}" class="dashboard-plot"></div>
        </div>
    `;
    }).join('');

    chartsRow2.innerHTML = renderCards(row2Charts, 0);
    chartsRow3.innerHTML = renderCards(row3Charts, row2Charts.length);

    const visibleCharts = [...row2Charts, ...row3Charts];
    visibleCharts.forEach((chart, i) => {
        const chartId = `dashboard-chart-${i}`;
        const layout = {
            margin: { t: 8, r: 8, b: 30, l: 34 },
            paper_bgcolor: '#ffffff',
            plot_bgcolor: '#ffffff',
            xaxis: { title: chart.x_title || '', tickfont: { size: 10 }, showgrid: false, zeroline: false },
            yaxis: { title: chart.y_title || '', tickfont: { size: 10 }, showgrid: false, zeroline: false },
            font: { size: 10, color: '#4b5563' },
            showlegend: true,
            legend: { orientation: 'h', y: 1.08, x: 0 }
        };

        let data = [];
        const color = palette[i % palette.length];
        if (chart.type === 'bar') {
            const isHorizontal = chart.orientation === 'h';
            data = [{
                type: 'bar',
                orientation: isHorizontal ? 'h' : 'v',
                x: isHorizontal ? (chart.y || []) : (chart.x || []),
                y: isHorizontal ? (chart.x || []) : (chart.y || []),
                marker: { color, opacity: 0.9 }
            }];
        } else if (chart.type === 'pie') {
            data = [{
                type: 'pie',
                labels: chart.labels || [],
                values: chart.values || [],
                hole: 0.38,
                textinfo: 'label+percent',
                textposition: 'inside',
                insidetextorientation: 'radial',
                hovertemplate: '%{label}: %{value} (%{percent})<extra></extra>'
            }];
        } else if (chart.type === 'histogram') {
            data = [{ type: 'histogram', x: chart.x || [], marker: { color, opacity: 0.85 } }];
        } else if (chart.type === 'line') {
            data = [{ type: 'scatter', mode: 'lines+markers', x: chart.x || [], y: chart.y || [], line: { color, width: 3 }, marker: { size: 7, color } }];
        } else if (chart.type === 'scatter') {
            data = [{ type: 'scatter', mode: 'markers', x: chart.x || [], y: chart.y || [], marker: { color, size: 8, opacity: 0.75 } }];
        } else if (chart.type === 'box') {
            data = [{ type: 'box', x: chart.x || [], y: chart.y || [], marker: { color }, line: { color } }];
        } else if (chart.type === 'heatmap') {
            data = [{ type: 'heatmap', x: chart.x || [], y: chart.y || [], z: chart.z || [], colorscale: 'RdBu', zmid: 0 }];
        }

        Plotly.newPlot(chartId, data, layout, { responsive: true, displayModeBar: false, staticPlot: false });
    });
}

async function downloadDashboardImage() {
    const container = document.getElementById('interactiveDashboardContainer');
    const status = document.getElementById('interactiveDashboardStatus');
    if (!container || !status || container.style.display === 'none') {
        alert('Dashboard is not ready to download.');
        return;
    }

    try {
        const canvas = await html2canvas(container, { scale: 2, backgroundColor: '#f5f5f7' });
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `interactive_dashboard_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (error) {
        alert(`Unable to download dashboard image: ${error.message}`);
    }
}

function openFullDashboardPage() {
    window.location.href = 'dashboard.html';
}

async function loadInteractiveDashboardSchema(refresh = false) {
    const status = document.getElementById('interactiveDashboardStatus');
    const container = document.getElementById('interactiveDashboardContainer');
    if (!status || !container) {
        return;
    }

    status.textContent = 'Generating interactive dashboard schema...';
    container.style.display = 'none';

    try {
        const url = `${API_BASE}/dashboard-schema/${datasetId}?refresh=${refresh ? 1 : 0}`;
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load dashboard schema');
        }

        const payload = await response.json();
        renderInteractiveDashboard(payload.dashboard_schema);
    } catch (error) {
        status.textContent = `Dashboard unavailable: ${error.message}`;
        container.style.display = 'none';
    }
}

// Load model results
async function loadModelResults() {
    try {
        const response = await fetch(`${API_BASE}/model-results/${datasetId}`);
        if (!response.ok) throw new Error('Failed to fetch results');
        
        const results = await response.json();
        const taskType = results.task_type || 'classification';
        const isRegression = taskType === 'regression';
        const html = [];
        
        // Safely extract metrics
        const metrics = results.best_model_metrics || {};
        const comparison = results.comparison || [];
        
        // Task type indicator
        const taskLabel = isRegression ? 'Regression' : 'Classification';
        const taskIcon = isRegression ? '📊' : '🎯';
        html.push(`<p style="color: #0071e3; font-weight: bold; margin-bottom: 15px;">${taskIcon} Task Type: ${taskLabel}</p>`);
        
        // Best model summary
        if (isRegression) {
            const r2 = metrics.r2_score !== undefined ? metrics.r2_score : 0;
            const mae = metrics.mae !== undefined ? metrics.mae : 0;
            const rmse = metrics.rmse !== undefined ? metrics.rmse : 0;
            const mape = metrics.mape !== undefined ? metrics.mape : 0;
            
            html.push(`
                <h2>Best Model: ${results.best_model}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div class="metric-box">
                        <div class="metric-label">R² Score</div>
                        <div class="metric-value">${(r2).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Mean Absolute Error</div>
                        <div class="metric-value">${(mae).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Root Mean Squared Error</div>
                        <div class="metric-value">${(rmse).toFixed(4)}</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Mean Absolute % Error</div>
                        <div class="metric-value">${(mape * 100).toFixed(2)}%</div>
                    </div>
                </div>
            `);
        } else {
            const accuracy = metrics.accuracy !== undefined ? metrics.accuracy : 0;
            const precision = metrics.precision !== undefined ? metrics.precision : 0;
            const recall = metrics.recall !== undefined ? metrics.recall : 0;
            const f1Score = metrics.f1_score !== undefined ? metrics.f1_score : 0;
            
            html.push(`
                <h2>Best Model: ${results.best_model}</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px;">
                    <div class="metric-box">
                        <div class="metric-label">Accuracy</div>
                        <div class="metric-value">${(accuracy * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Precision</div>
                        <div class="metric-value">${(precision * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">Recall</div>
                        <div class="metric-value">${(recall * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-box">
                        <div class="metric-label">F1-Score</div>
                        <div class="metric-value">${(f1Score * 100).toFixed(2)}%</div>
                    </div>
                </div>
            `);
        }
        
        // Model comparison
        html.push('<div class="model-comparison">');
        html.push('<h3>All Models Performance</h3>');
        html.push('<table>');
        
        if (isRegression) {
            html.push('<thead><tr><th>Model</th><th>R² Score</th><th>MAE</th><th>RMSE</th><th>MAPE</th></tr></thead>');
            html.push('<tbody>');
            
            comparison.forEach(model => {
                const isbest = model.model === results.best_model ? 'best' : '';
                const r2 = model.r2_score !== undefined ? model.r2_score : 0;
                const mae = model.mae !== undefined ? model.mae : 0;
                const rmse = model.rmse !== undefined ? model.rmse : 0;
                const mape = model.mape !== undefined ? model.mape : 0;
                
                html.push(`
                    <tr class="${isbest}">
                        <td>${model.model}</td>
                        <td>${(r2).toFixed(4)}</td>
                        <td>${(mae).toFixed(4)}</td>
                        <td>${(rmse).toFixed(4)}</td>
                        <td>${(mape * 100).toFixed(2)}%</td>
                    </tr>
                `);
            });
        } else {
            html.push('<thead><tr><th>Model</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1-Score</th></tr></thead>');
            html.push('<tbody>');
            
            comparison.forEach(model => {
                const isbest = model.model === results.best_model ? 'best' : '';
                const accuracy = model.accuracy !== undefined ? model.accuracy : 0;
                const precision = model.precision !== undefined ? model.precision : 0;
                const recall = model.recall !== undefined ? model.recall : 0;
                const f1Score = model.f1_score !== undefined ? model.f1_score : 0;
                
                html.push(`
                    <tr class="${isbest}">
                        <td>${model.model}</td>
                        <td>${(accuracy * 100).toFixed(2)}%</td>
                        <td>${(precision * 100).toFixed(2)}%</td>
                        <td>${(recall * 100).toFixed(2)}%</td>
                        <td>${(f1Score * 100).toFixed(2)}%</td>
                    </tr>
                `);
            });
        }
        
        html.push('</tbody></table></div>');
        
        // Feature importance (if available - more relevant for tree-based models)
        if (results.feature_importance && Object.keys(results.feature_importance).length > 0) {
            html.push('<div class="feature-importance">');
            html.push('<h3>Feature Importance</h3>');

            const allImportanceEntries = Object.entries(results.feature_importance);
            const filteredImportanceEntries = allImportanceEntries.filter(([feature]) => !isDateLikeFeatureName(feature));

            if (!filteredImportanceEntries.length) {
                html.push('<p class="text-muted">Date/time-derived columns were hidden from feature importance.</p>');
                html.push('</div>');
                document.getElementById('modelResultsContent').innerHTML = html.join('');
                return;
            }

            const maxImportance = Math.max(...filteredImportanceEntries.map(([, value]) => Number(value) || 0));
            const safeMaxImportance = maxImportance > 0 ? maxImportance : 1;

            filteredImportanceEntries.forEach(([feature, importance]) => {
                const numericImportance = Number(importance) || 0;
                const percentage = ((numericImportance / safeMaxImportance) * 100).toFixed(1);
                html.push(`
                    <div class="feature-bar">
                        <div class="feature-name">${feature}</div>
                        <div class="feature-value" style="width: ${percentage}%"></div>
                        <div class="feature-percent">${percentage}%</div>
                    </div>
                `);
            });

            if (filteredImportanceEntries.length < allImportanceEntries.length) {
                html.push('<p class="text-muted">Date/time-derived columns are hidden in this view.</p>');
            }
            
            html.push('</div>');
        }
        
        document.getElementById('modelResultsContent').innerHTML = html.join('');
        
    } catch (error) {
        console.error('Error loading results:', error);
        document.getElementById('modelResultsContent').innerHTML = `<p style="color: red;">Error loading results: ${error.message}</p>`;
    }
}

// Enable prediction UI
function enablePredictionUI() {
    loadPredictionForm();
}

function inferInputTypeFromSample(columnName, sampleRows) {
    const values = (sampleRows || [])
        .map((row) => row?.[columnName])
        .filter((value) => value !== null && value !== undefined && String(value).trim() !== '');

    if (!values.length) return 'text';

    const loweredName = String(columnName || '').toLowerCase();
    const explicitDateName = /(date|dob|timestamp|created|updated|time)/.test(loweredName);

    const numericCount = values.filter((value) => !isNaN(value) && String(value).trim() !== '').length;
    const dateLikeValues = values.filter((value) => {
        const str = String(value).trim();
        if (!str) return false;
        if (!isNaN(Number(str)) && str.length < 6) return false;
        const parsed = Date.parse(str);
        return !Number.isNaN(parsed);
    });

    const numericRatio = numericCount / values.length;
    const dateRatio = dateLikeValues.length / values.length;

    if (dateRatio >= 0.7 || (explicitDateName && dateRatio >= 0.3)) {
        const hasTime = dateLikeValues.some((value) => /\d{1,2}:\d{2}|T\d{1,2}:\d{2}/.test(String(value)));
        return hasTime ? 'datetime-local' : 'date';
    }

    if (numericRatio >= 0.9) {
        return 'number';
    }

    return 'text';
}

function isDateLikeFeatureName(featureName) {
    const raw = String(featureName || '');
    if (!raw) return false;

    const hasDateKeyword = /(^|[^a-zA-Z])(date|datetime|timestamp|time|deliverydate|orderdate)([^a-zA-Z]|$)/i.test(raw);
    const hasDateToken = /\d{4}[-_/]\d{1,2}[-_/]\d{1,2}/.test(raw);

    return hasDateKeyword || hasDateToken;
}

function formatInputValueByType(value, inputType) {
    if (value === null || value === undefined) return '';
    const raw = String(value).trim();
    if (!raw) return '';

    if (inputType === 'date') {
        const parsed = new Date(raw);
        if (Number.isNaN(parsed.getTime())) return '';
        return parsed.toISOString().slice(0, 10);
    }

    if (inputType === 'datetime-local') {
        const parsed = new Date(raw);
        if (Number.isNaN(parsed.getTime())) return '';
        const year = parsed.getFullYear();
        const month = String(parsed.getMonth() + 1).padStart(2, '0');
        const day = String(parsed.getDate()).padStart(2, '0');
        const hour = String(parsed.getHours()).padStart(2, '0');
        const minute = String(parsed.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day}T${hour}:${minute}`;
    }

    return raw;
}

function parseSinglePredictionValue(element) {
    const rawValue = element?.value;
    const inputType = element?.dataset?.inputType || element?.type || 'text';

    if (rawValue === '' || rawValue === undefined || rawValue === null) {
        return undefined;
    }

    if (inputType === 'number') {
        return Number(rawValue);
    }

    if (inputType === 'date' || inputType === 'datetime-local') {
        return String(rawValue);
    }

    return String(rawValue).trim();
}

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

// Load prediction form
async function loadPredictionForm() {
    try {
        // Get feature names
        const response = await fetch(`${API_BASE}/data/${datasetId}`);
        if (!response.ok) throw new Error('Failed to fetch data');
        
        const data = await response.json();
        const sampleRows = data.sample || [];
        let columns = Object.keys(data.sample[0] || {});
        
        // Get target column to exclude from features
        const modelResponse = await fetch(`${API_BASE}/model-results/${datasetId}`);
        let target = null;
        if (modelResponse.ok) {
            const modelData = await modelResponse.json();
            target = modelData.target;
            columns = columns.filter(col => col !== target);
        }
        
        // Get categorical info
        const catResponse = await fetch(`${API_BASE}/categorical-info/${datasetId}`);
        const catData = catResponse.ok ? await catResponse.json() : { binary_categorical: {}, clustered_categorical: {}, fully_categorical: [] };
        
        // Create sets for easy lookup
        const binaryCols = new Set(Object.keys(catData.binary_categorical));
        const clusteredCols = new Set(Object.keys(catData.clustered_categorical));
        const removedCols = new Set(catData.fully_categorical);
        
        const html = [];
        
        // Add tabs for single vs batch prediction
        html.push(`
            <div class="prediction-tab-strip">
                <button onclick="switchPredictionMode('single')" id="singlePredBtn" class="prediction-tab-btn active">
                    📝 Single Prediction
                </button>
                <button onclick="switchPredictionMode('batch')" id="batchPredBtn" class="prediction-tab-btn">
                    📊 Batch Prediction
                </button>
            </div>
        `);
        
        // Single Prediction Section
        html.push('<div id="singlePredictionSection" style="display: block;">');
        html.push('<h3>Enter Feature Values for Prediction</h3>');
        html.push('<div class="prediction-form">');
        
        columns.forEach(col => {
            // Skip removed columns
            if (removedCols.has(col)) return;
            
            // Find categorical info
            const binaryInfo = catData.binary_categorical[col];
            const clusteredInfo = catData.clustered_categorical[col];
            
            if (binaryInfo) {
                const [positive_class, negative_class] = binaryInfo;
                html.push(`
                    <div class="form-group">
                        <label>${col} (Binary)</label>
                        <select id="feature_${col}" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                            <option value="">-- Select --</option>
                            <option value="${positive_class}">${positive_class}</option>
                            <option value="${negative_class}">${negative_class}</option>
                        </select>
                    </div>
                `);
            } else if (clusteredInfo) {
                const options = clusteredInfo.map(val => `<option value="${val}">${val}</option>`).join('');
                html.push(`
                    <div class="form-group">
                        <label>${col} (Categorical)</label>
                        <select id="feature_${col}" style="width: 100%; padding: 10px; border: 1px solid #ddd; border-radius: 4px;">
                            <option value="">-- Select --</option>
                            ${options}
                        </select>
                    </div>
                `);
            } else {
                const inferredType = inferInputTypeFromSample(col, sampleRows);
                const firstSample = sampleRows.length ? sampleRows[0][col] : '';
                const prefillValue = formatInputValueByType(firstSample, inferredType);
                const helpText = inferredType === 'date'
                    ? 'Use date format (DD-MM-YYYY).'
                    : inferredType === 'datetime-local'
                        ? 'Use date and time format.'
                        : inferredType === 'text'
                            ? 'String values are supported.'
                            : 'Numeric values are supported.';

                html.push(`
                    <div class="form-group">
                        <label>${col} (${inferredType === 'datetime-local' ? 'date & time' : inferredType})</label>
                        <input type="${inferredType}" id="feature_${col}" data-input-type="${inferredType}" placeholder="Enter ${col}" ${inferredType === 'number' ? 'step="any"' : ''} value="${prefillValue}">
                        <small style="display:block; color:#667085; margin-top:6px;">${helpText}</small>
                    </div>
                `);
            }
        });
        
        html.push('<button onclick="makePrediction()">🔮 Predict</button>');
        html.push('</div>');
        html.push('<div id="singlePredictionOutput"></div>');
        html.push('</div>');
        
        // Batch Prediction Section
        html.push(`
            <div id="batchPredictionSection" style="display: none;">
                <h3>Upload Dataset for Batch Prediction</h3>
                <div style="background: #f7f7f8; padding: 20px; border-radius: 12px; margin: 15px 0; border: 1px solid rgba(0,0,0,0.05);">
                    <p style="color: #666; margin-bottom: 15px;">
                        Upload a CSV file with the same columns as your training data (without the target column).
                        <br/>Predictions will be made for each row.
                    </p>
                    <div class="form-group">
                        <label>Select CSV File:</label>
                        <div class="file-upload">
                            <input type="file" id="batchPredFile" class="file-upload-input" accept=".csv">
                            <div class="file-upload-shell">
                                <label for="batchPredFile" class="file-upload-trigger">Select CSV File</label>
                                <div id="batchPredFileName" class="file-upload-name">No CSV selected</div>
                            </div>
                            <p class="file-upload-hint">Accepted format: CSV only.</p>
                        </div>
                    </div>
                    <button id="batchPredictBtn" onclick="predictBatch()">
                        🚀 Predict on Dataset
                    </button>
                </div>
                <div id="batchPredictionOutput"></div>
            </div>
        `);
        
        document.getElementById('predictionContent').innerHTML = html.join('');
        initFileUploadBox('batchPredFile', 'batchPredFileName', 'No CSV selected');
        
    } catch (error) {
        console.error('Error loading prediction form:', error);
        document.getElementById('predictionContent').innerHTML = `<p style="color: red;">Error loading form: ${error.message}</p>`;
    }
}

// Switch between single and batch prediction modes
function switchPredictionMode(mode) {
    const singleBtn = document.getElementById('singlePredBtn');
    const batchBtn = document.getElementById('batchPredBtn');
    const singleSection = document.getElementById('singlePredictionSection');
    const batchSection = document.getElementById('batchPredictionSection');

    if (!singleBtn || !batchBtn || !singleSection || !batchSection) return;
    
    if (mode === 'single') {
        singleBtn.classList.add('active');
        batchBtn.classList.remove('active');
        singleSection.style.display = 'block';
        batchSection.style.display = 'none';
    } else {
        singleBtn.classList.remove('active');
        batchBtn.classList.add('active');
        singleSection.style.display = 'none';
        batchSection.style.display = 'block';
    }
}

// Make batch prediction
async function predictBatch() {
    const batchBtn = document.getElementById('batchPredictBtn');
    const outputDiv = document.getElementById('batchPredictionOutput');

    try {
        const fileInput = document.getElementById('batchPredFile');
        if (!fileInput.files.length) {
            alert('Please select a CSV file');
            return;
        }
        
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('id', datasetId);
        formData.append('file', file);

        if (batchBtn) {
            batchBtn.disabled = true;
        }
        if (outputDiv) {
            outputDiv.innerHTML = '';
        }

        setGlobalLoader(true, 'Running batch prediction on your dataset...');
        
        const response = await fetch(`${API_BASE}/predict-batch`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Batch prediction failed');
        }
        
        const result = await response.json();
        
        // Display results
        displayBatchResults(result);
        
    } catch (error) {
        console.error('Batch prediction error:', error);
        if (outputDiv) {
            outputDiv.innerHTML = `
            <div style="background: #fff3f2; color: #b42318; padding: 15px; border-radius: 12px; margin-top: 15px; border: 1px solid rgba(217,45,32,0.2);">
                <strong>Error:</strong> ${error.message}
            </div>
        `;
        }
    } finally {
        setGlobalLoader(false);
        if (batchBtn) {
            batchBtn.disabled = false;
        }
    }
}

// Display batch prediction results
function displayBatchResults(result) {
    const outputDiv = document.getElementById('batchPredictionOutput');
    const html = [];
    
    html.push(`
        <div style="background: #ecfdf3; color: #027a48; padding: 15px; border-radius: 12px; margin: 15px 0; border: 1px solid rgba(16,185,129,0.25);">
            <strong>✓ Predictions Complete!</strong>
            <br/>Total rows: ${result.total_rows} | Successfully predicted: ${result.predicted_rows}
        </div>
    `);
    
    // Display predictions in table format
    if (result.predictions && result.predictions.length > 0) {
        html.push('<div style="overflow-x: auto; margin: 15px 0;">');
        html.push('<table style="width: 100%; border-collapse: collapse; font-size: 13px;">');
        
        // Table header
        html.push('<thead><tr style="background: #0071e3; color: white;">');
        result.columns.forEach(col => {
            html.push(`<th style="padding: 10px; text-align: left; border: 1px solid #ddd;">${col}</th>`);
        });
        html.push('</tr></thead>');
        
        // Table body (show first 100 rows)
        html.push('<tbody>');
        result.predictions.slice(0, 100).forEach((row, idx) => {
            const bgColor = idx % 2 === 0 ? '#fff' : '#f9f9f9';
            html.push(`<tr style="background: ${bgColor};">`);
            result.columns.forEach(col => {
                const value = row[col];
                let displayValue = value;
                if (value === null || value === undefined) {
                    displayValue = '-';
                } else if (typeof value === 'object') {
                    displayValue = JSON.stringify(value);
                }
                html.push(`<td style="padding: 10px; border: 1px solid #ddd;">${displayValue}</td>`);
            });
            html.push('</tr>');
        });
        html.push('</tbody>');
        html.push('</table>');
        html.push('</div>');
        
        if (result.predictions.length > 100) {
            html.push(`<p style="color: #666; font-size: 12px;">Showing 100 of ${result.predictions.length} rows</p>`);
        }
    }
    
    // Download button
    html.push(`
        <button onclick="downloadPredictions(${JSON.stringify(result).replace(/"/g, '&quot;')});" 
                style="background: #0071e3; color: white; padding: 10px 20px; border: none; border-radius: 999px; cursor: pointer; margin-top: 15px;">
            💾 Download Results as CSV
        </button>
    `);
    
    outputDiv.innerHTML = html.join('');
}

// Download batch predictions as CSV
function downloadPredictions(result) {
    try {
        // Create CSV content
        const columns = result.columns;
        const predictions = result.predictions;
        
        // Header row
        const csvContent = [columns.join(',')];
        
        // Data rows
        predictions.forEach(row => {
            const values = columns.map(col => {
                let value = row[col];
                if (value === null || value === undefined) {
                    return '';
                }
                // Escape quotes and wrap in quotes if contains comma
                if (typeof value === 'string' && (value.includes(',') || value.includes('"'))) {
                    value = '"' + value.replace(/"/g, '""') + '"';
                }
                return value;
            });
            csvContent.push(values.join(','));
        });
        
        // Create blob and download
        const csv = csvContent.join('\n');
        const blob = new Blob([csv], { type: 'text/csv' });
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `predictions_${new Date().getTime()}.csv`;
        document.body.appendChild(a);
        a.click();
        window.URL.revokeObjectURL(url);
        document.body.removeChild(a);
    } catch (error) {
        alert('Error downloading file: ' + error.message);
    }
}

// Make prediction
async function makePrediction() {
    try {
        // Gather feature values
        const response = await fetch(`${API_BASE}/data/${datasetId}`);
        const data = await response.json();
        let columns = Object.keys(data.sample[0] || {});
        
        // Get target column to exclude
        const modelResponse = await fetch(`${API_BASE}/model-results/${datasetId}`);
        let target = null;
        if (modelResponse.ok) {
            const modelData = await modelResponse.json();
            target = modelData.target;
            columns = columns.filter(col => col !== target);
        }
        
        const inputData = {};
        for (let col of columns) {
            const element = document.getElementById(`feature_${col}`);
            const value = parseSinglePredictionValue(element);
            if (value === '' || value === undefined) {
                alert(`Please provide a value for ${col}`);
                return;
            }

            inputData[col] = value;
        }
        
        // Make prediction
        const formData = new FormData();
        formData.append('id', datasetId);
        formData.append('input_data', JSON.stringify(inputData));
        
        const predResponse = await fetch(`${API_BASE}/predict`, {
            method: 'POST',
            body: formData
        });
        
        if (!predResponse.ok) {
            const error = await predResponse.json();
            throw new Error(error.detail || 'Prediction failed');
        }
        
        const result = await predResponse.json();
        displayPrediction(result, inputData);
        
    } catch (error) {
        console.error('Error:', error);
        const output = document.getElementById('singlePredictionOutput');
        if (output) {
            output.innerHTML = `<div class="prediction-result error"><strong>Error:</strong> ${error.message}</div>`;
        }
    }
}

// Display prediction result
function displayPrediction(result, inputData) {
    const html = [];
    const isRegression = result.task_type === 'regression';
    
    html.push('<div class="prediction-result">');
    html.push(`<h3>🎯 ${isRegression ? 'Predicted Value' : 'Prediction Result'}</h3>`);
    html.push(`<p><strong>Model Used:</strong> ${result.model_used} (${isRegression ? 'Regression' : 'Classification'})</p>`);
    
    if (isRegression) {
        // Regression: show predicted numeric value
        html.push('<div style="background: white; padding: 15px; border-radius: 4px; margin: 15px 0;">');
        html.push(`<p style="font-size: 14px; color: #666;">Predicted Value:</p>`);
        html.push(`<p style="font-size: 32px; font-weight: bold; color: #0071e3;">${(result.prediction).toFixed(4)}</p>`);
        html.push('</div>');
    } else {
        // Classification: show prediction and probabilities
        if (result.prediction && typeof result.prediction === 'object') {
            html.push('<h4>Prediction:</h4>');
            html.push('<ul>');
            Object.entries(result.prediction).forEach(([key, value]) => {
                html.push(`<li><strong>${key}:</strong> ${value}</li>`);
            });
            html.push('</ul>');
        } else if (result.prediction) {
            html.push(`<p style="font-size: 18px; font-weight: bold;">Predicted Class: <span style="color: #0071e3;">${result.prediction}</span></p>`);
        }
        
        // Show probabilities if available
        if (result.probabilities) {
            html.push('<h4>Confidence Scores:</h4>');
            html.push('<ul>');
            if (Array.isArray(result.probabilities[0])) {
                result.probabilities[0].forEach((prob, idx) => {
                    html.push(`<li>Class ${idx}: ${(prob * 100).toFixed(2)}%</li>`);
                });
            }
            html.push('</ul>');
        }
    }
    
    html.push('</div>');
    
    // Show input data summary
    html.push('<div class="card section-spaced">');
    html.push('<h4>Input Features</h4>');
    html.push('<ul>');
    Object.entries(inputData).forEach(([key, value]) => {
        html.push(`<li><strong>${key}:</strong> ${value}</li>`);
    });
    html.push('</ul>');
    html.push('</div>');
    
    document.getElementById('singlePredictionOutput').innerHTML = html.join('');
}

// Tab switching
function activateTab(tabName) {
    // Hide all tabs
    document.querySelectorAll('.tab-content').forEach(tab => {
        tab.classList.remove('active');
    });
    
    // Deactivate all tab buttons
    document.querySelectorAll('.tab-button').forEach(btn => {
        btn.classList.remove('active');
    });
    
    // Show selected tab
    document.getElementById(tabName).classList.add('active');

    // Activate selected button
    document.querySelectorAll('.tab-button').forEach(btn => {
        if (btn.getAttribute('onclick') === `switchTab('${tabName}')`) {
            btn.classList.add('active');
        }
    });

    sessionStorage.setItem('model_active_tab', tabName);
    
    // Load results if switching to results tab
    if (tabName === 'results') {
        loadModelResults();
    }
    if (tabName === 'dashboard') {
        loadInteractiveDashboardSchema(false);
    }

}

async function switchTab(tabName) {
    if (tabName !== 'training') {
        const trained = await isModelTrained();
        if (!trained) {
            setPostTrainingTabsVisible(false);
            alert('Please train the model first.');
            activateTab('training');
            return;
        }
        setPostTrainingTabsVisible(true);
    }

    if (tabName === 'dashboard') {
        const trained = await isModelTrained();
        if (trained) {
            openFullDashboardPage();
            return;
        }

        alert('Please train a model first to open the interactive dashboard.');
        activateTab('training');
        return;
    }

    activateTab(tabName);

    if (typeof event !== 'undefined' && event && event.target) {
        document.querySelectorAll('.tab-button').forEach(btn => {
            btn.classList.remove('active');
        });
        event.target.classList.add('active');
    }

}

// Go back to upload
function goBack() {
    window.location.href = 'target.html';
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', initializeML);
