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
let currentSchema = null;
let currentView = 'all';

const VIEW_LABELS = {
    all: 'Dashboard',
    model: 'Model Performance',
    feature: 'Feature Stats',
    quality: 'Data Quality'
};

function goBackModel() {
    window.location.href = 'model.html';
}

function _renderKpis(schema) {
    const grid = document.getElementById('dashboardKpiGrid');
    if (!grid) return;

    const kpis = (schema.kpis || []).slice(0, 4);
    grid.innerHTML = kpis.map((k, idx) => `
        <div class="kpi-card kpi-${idx + 1}">
            <div class="kpi-label">${k.label || ''}</div>
            <div class="kpi-value">${k.value ?? ''}${k.suffix || ''}</div>
        </div>
    `).join('');
}

function _classifyChart(chart) {
    const text = `${chart?.title || ''} ${chart?.x_title || ''} ${chart?.y_title || ''}`.toLowerCase();

    if (text.includes('missing') || text.includes('completeness') || text.includes('quality') || text.includes('null')) {
        return 'quality';
    }
    if (
        text.includes('model') ||
        text.includes('comparison') ||
        text.includes('accuracy') ||
        text.includes('precision') ||
        text.includes('recall') ||
        text.includes('f1') ||
        text.includes('r2') ||
        text.includes('auc')
    ) {
        return 'model';
    }
    return 'feature';
}

function _isQualityOnlyChart(chart) {
    const text = `${chart?.title || ''} ${chart?.x_title || ''} ${chart?.y_title || ''}`.toLowerCase();

    if (text.includes('relationship:')) return false;
    if (text.includes('category share')) return false;
    if (text.includes('model comparison') || text.includes('feature importance') || text.includes('distribution')) return false;

    return (
        chart?.type === 'heatmap' ||
        text.includes('completeness') ||
        text.includes('missing') ||
        text.includes('null') ||
        text.includes('duplicate') ||
        text.includes('quality')
    );
}

function _getChartsByView(schema, view) {
    const allCharts = (schema?.charts || []).slice(0, 8);
    if (view === 'all') return allCharts;

    if (view === 'feature') {
        const lowerTitle = (chart) => (chart?.title || '').toLowerCase();
        const allHeatmaps = allCharts.filter((chart) => chart?.type === 'heatmap');
        const featureCharts = allCharts.filter((chart) => _classifyChart(chart) === 'feature');

        const heatmap = featureCharts.find((chart) => chart?.type === 'heatmap') || allHeatmaps[0] || null;
        const nonHeatFeature = featureCharts.filter((chart) => chart?.type !== 'heatmap');

        // Prefer feature importance and distribution charts first in feature view.
        nonHeatFeature.sort((a, b) => {
            const at = lowerTitle(a);
            const bt = lowerTitle(b);
            const aScore = (at.includes('feature importance') ? 3 : 0) + (at.includes('distribution') ? 2 : 0);
            const bScore = (bt.includes('feature importance') ? 3 : 0) + (bt.includes('distribution') ? 2 : 0);
            return bScore - aScore;
        });

        const selected = nonHeatFeature.slice(0, 4);

        if (selected.length < 4) {
            const extra = allCharts.filter(
                (chart) => chart?.type !== 'heatmap' && !selected.includes(chart) && chart !== heatmap
            );
            selected.push(...extra.slice(0, 4 - selected.length));
        }

        if (heatmap) {
            selected.push(heatmap);
        }

        return selected.slice(0, 5);
    }

    if (view === 'quality') {
        const qualityCharts = allCharts.filter((chart) => _isQualityOnlyChart(chart) && chart?.type !== 'heatmap');
        return qualityCharts.slice(0, 2);
    }

    const filtered = allCharts.filter((chart) => _classifyChart(chart) === view);
    return filtered.length ? filtered : allCharts;
}

function _setNavActive(view) {
    document.querySelectorAll('.nav-item[data-view]').forEach((item) => {
        item.classList.toggle('active', item.dataset.view === view);
    });
}

function _plotChart(chart, idx) {
    const chartId = `full-dashboard-chart-${idx}`;
    const isPieChart = chart?.type === 'pie';
    const layout = {
        margin: isPieChart ? { t: 10, r: 10, b: 10, l: 10 } : { t: 4, r: 6, b: 24, l: 28 },
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        xaxis: isPieChart ? {} : { title: chart.x_title || '', tickfont: { size: 9 }, titlefont: { size: 9 }, showgrid: false, zeroline: false },
        yaxis: isPieChart ? {} : { title: chart.y_title || '', tickfont: { size: 9 }, titlefont: { size: 9 }, showgrid: false, zeroline: false },
        font: { size: 9, color: '#4b5563' },
        showlegend: false
    };

    const colorSet = ['#0071e3', '#4f84c4', '#7fa8d9', '#6e6e73', '#9ca3af', '#4b5563'];
    const color = colorSet[idx % colorSet.length];

    let data = [];
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
        const isDataCompleteness = (chart?.id === 'missing_data') || ((chart?.title || '').toLowerCase().includes('data completeness'));
        const pieDomain = isDataCompleteness
            ? { x: [0.10, 0.90], y: [0.16, 0.90] }
            : { x: [0.06, 0.94], y: [0.06, 0.94] };
        data = [{
            type: 'pie',
            labels: chart.labels || [],
            values: chart.values || [],
            // Keep a donut shape for completeness chart to avoid a fully-filled circle visual.
            hole: isDataCompleteness ? 0.5 : 0.38,
            domain: pieDomain,
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
}

function _getChartSpanClass(chart, view) {
    const title = (chart?.title || '').toLowerCase();

    if (view === 'feature' && chart?.type === 'heatmap') {
        return 'span-2';
    }

    if (view === 'model' && title.includes('model comparison')) {
        return 'span-4';
    }

    if (view === 'all') {
        if (title.includes('correlation heatmap')) {
            return 'span-2';
        }
        if (title.includes('data completeness')) {
            return '';
        }
    }

    return '';
}

function _renderCharts(schema) {
    const grid = document.getElementById('dashboardChartsGrid');
    if (!grid) return;

    const charts = _getChartsByView(schema, currentView);
    grid.classList.remove('tight-last-row-1', 'tight-last-row-2', 'tight-last-row-3');

    const spanClasses = charts.map((chart) => _getChartSpanClass(chart, currentView));
    const hasCustomSpan = spanClasses.some((spanClass) => spanClass.length > 0);

    if (!hasCustomSpan) {
        const remainder = charts.length % 4;
        if (charts.length > 4 && remainder > 0) {
            grid.classList.add(`tight-last-row-${remainder}`);
        }
    }

    grid.innerHTML = charts.map((chart, idx) => `
        <div class="chart-card ${spanClasses[idx]}">
            <h4 class="chart-title">${chart.title || `Chart ${idx + 1}`}</h4>
            <div id="full-dashboard-chart-${idx}" class="chart-plot"></div>
        </div>
    `).join('');

    charts.forEach((chart, idx) => _plotChart(chart, idx));

    // Force a resize pass after layout paint so plots stay inside card bounds.
    requestAnimationFrame(() => {
        charts.forEach((_, idx) => {
            const node = document.getElementById(`full-dashboard-chart-${idx}`);
            if (node && typeof Plotly !== 'undefined') {
                Plotly.Plots.resize(node);
            }
        });
    });
}

function _formatStat(value, decimals = 4) {
    const n = Number(value);
    if (!Number.isFinite(n)) return 'N/A';
    return n.toFixed(decimals);
}

function _renderQualityNumericSummary(schema) {
    const box = document.getElementById('dashboardNumericSummary');
    if (!box) return;

    if (currentView !== 'quality') {
        box.style.display = 'none';
        box.innerHTML = '';
        return;
    }

    const summary = schema?.numeric_summary || {};
    box.style.display = 'block';

    if (!summary.available) {
        box.innerHTML = `
            <div class="quality-summary-title">Numeric Data Summary</div>
            <div class="quality-summary-value">${summary.message || 'No numeric summary available'}</div>
        `;
        return;
    }

    const items = [
        ['Numeric Columns', summary.numeric_columns ?? 'N/A'],
        ['Count', summary.count ?? 'N/A'],
        ['Mean', _formatStat(summary.mean)],
        ['Median', _formatStat(summary.median)],
        ['Std Dev', _formatStat(summary.std)],
        ['Min', _formatStat(summary.min)],
        ['Q1', _formatStat(summary.q1)],
        ['Q3', _formatStat(summary.q3)],
        ['Max', _formatStat(summary.max)]
    ];

    box.innerHTML = `
        <div class="quality-summary-title">Numeric Data Summary</div>
        <div class="quality-summary-grid">
            ${items.map(([label, value]) => `
                <div class="quality-summary-item">
                    <div class="quality-summary-label">${label}</div>
                    <div class="quality-summary-value">${value}</div>
                </div>
            `).join('')}
        </div>
    `;
}

function _renderView() {
    const status = document.getElementById('dashboardStatus');
    if (!currentSchema) return;

    _setNavActive(currentView);
    _renderKpis(currentSchema);
    _renderQualityNumericSummary(currentSchema);
    _renderCharts(currentSchema);

    if (status) {
        const chartsCount = _getChartsByView(currentSchema, currentView).length;
        status.textContent = `${VIEW_LABELS[currentView]} view | target: ${currentSchema.target || 'N/A'} (${currentSchema.task_type || 'unknown'}) | charts: ${chartsCount}`;
    }
}

function switchDashboardView(view) {
    currentView = view;
    _renderView();
}

async function loadDashboard(refresh = false) {
    const status = document.getElementById('dashboardStatus');
    if (!status) return;

    if (!datasetId) {
        status.textContent = 'No dataset selected. Go back and train models first.';
        return;
    }

    status.textContent = 'Loading interactive dashboard...';

    try {
        const url = `${API_BASE}/dashboard-schema/${datasetId}?refresh=${refresh ? 1 : 0}`;
        const response = await fetch(url);
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to load dashboard schema');
        }

        const payload = await response.json();
        currentSchema = payload.dashboard_schema;
        _renderView();
    } catch (err) {
        status.textContent = `Dashboard load failed: ${err.message}`;
    }
}

function refreshDashboard() {
    loadDashboard(true);
}

async function downloadDashboardImage() {
    const root = document.querySelector('.main');
    if (!root) return;

    try {
        const canvas = await html2canvas(root, { scale: 2, backgroundColor: '#f5f5f7' });
        const link = document.createElement('a');
        link.href = canvas.toDataURL('image/png');
        link.download = `interactive_dashboard_${Date.now()}.png`;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    } catch (err) {
        alert(`Unable to download dashboard image: ${err.message}`);
    }
}

window.addEventListener('DOMContentLoaded', () => {
    document.querySelectorAll('.nav-item[data-view]').forEach((item) => {
        item.addEventListener('click', () => switchDashboardView(item.dataset.view));
    });
    loadDashboard(false);
});
