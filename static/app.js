// LLM Tuner - Frontend Application

class LLMTunerApp {
    constructor() {
        this.apiBase = '';
        this.currentBenchmark = null;
        this.pollInterval = null;
        this.charts = {};
        
        this.init();
    }

    async init() {
        await this.loadHardware();
        await this.loadEngines();
        await this.loadHistory();
        this.setupEventListeners();
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('benchmark-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.startBenchmark();
        });

        // Benchmark type change - show/hide options
        document.getElementById('benchmark-type').addEventListener('change', (e) => {
            const aiTuneOptions = document.getElementById('ai-tune-options');
            if (e.target.value === 'ai_tune') {
                aiTuneOptions.classList.remove('hidden');
            } else {
                aiTuneOptions.classList.add('hidden');
            }
        });
    }

    async loadHardware() {
        try {
            const response = await fetch(`${this.apiBase}/api/hardware`);
            const hardware = await response.json();
            
            this.renderHardware(hardware);
        } catch (error) {
            console.error('Failed to load hardware:', error);
        }
    }

    renderHardware(hw) {
        const bar = document.getElementById('hardware-bar');
        bar.classList.remove('hidden');

        // GPU vendor badge
        const vendorBadge = document.getElementById('gpu-vendor');
        vendorBadge.textContent = hw.vendor?.toUpperCase() || 'UNKNOWN';
        
        if (hw.vendor === 'rocm') {
            vendorBadge.style.background = '#f97316'; // Orange for AMD
        } else if (hw.vendor === 'cuda') {
            vendorBadge.style.background = '#7c3aed'; // Purple for NVIDIA
        }

        document.getElementById('gpu-name').textContent = hw.gpu?.name || '';
        
        const vramTotal = hw.gpu?.vram_total_mb ? `${Math.round(hw.gpu.vram_total_mb / 1024)}GB` : 'N/A';
        const vramUsed = hw.gpu?.vram_used_mb ? ` (${Math.round(hw.gpu.vram_used_mb / 1024 * 10) / 10}GB used)` : '';
        document.getElementById('vram-info').textContent = `${vramTotal}${vramUsed}`;

        const cpuName = hw.cpu?.name || 'Unknown CPU';
        const cores = hw.cpu?.cores ? ` (${hw.cpu.cores} cores)` : '';
        document.getElementById('cpu-info').textContent = `${cpuName}${cores}`;
    }

    async loadEngines() {
        try {
            const response = await fetch(`${this.apiBase}/api/engines`);
            const engines = await response.json();
            
            const select = document.getElementById('engine-select');
            select.innerHTML = '';
            
            engines.forEach(engine => {
                const option = document.createElement('option');
                option.value = engine.id;
                option.textContent = `${engine.name} ${engine.status !== 'available' ? `(${engine.status})` : ''}`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load engines:', error);
        }
    }

    async startBenchmark() {
        const form = document.getElementById('benchmark-form');
        const startBtn = document.getElementById('start-btn');
        
        // Disable button during submission
        startBtn.disabled = true;
        startBtn.textContent = 'Starting...';

        try {
            const requestData = {
                model_path: document.getElementById('model-path').value,
                engine: document.getElementById('engine-select').value,
                benchmark_type: document.getElementById('benchmark-type').value,
                context_lengths: this.parseContextLengths(),
                max_tokens: parseInt(document.getElementById('max-tokens').value),
                temperature: 0.7,
                rounds: parseInt(document.getElementById('tune-rounds').value)
            };

            const response = await fetch(`${this.apiBase}/api/benchmarks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            
            this.currentBenchmark = result.id;
            this.showResultsPanel();
            this.startPolling(result.id);
            
        } catch (error) {
            console.error('Failed to start benchmark:', error);
            alert(`Error: ${error.message}`);
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Benchmark';
        }
    }

    parseContextLengths() {
        const input = document.getElementById('context-lengths').value;
        return input.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
    }

    showResultsPanel() {
        document.getElementById('results-panel').classList.remove('hidden');
    }

    startPolling(benchmarkId) {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
        }

        this.pollInterval = setInterval(async () => {
            await this.checkBenchmarkStatus(benchmarkId);
        }, 2000); // Poll every 2 seconds
    }

    async checkBenchmarkStatus(id) {
        try {
            const response = await fetch(`${this.apiBase}/api/benchmarks/${id}`);
            const benchmark = await response.json();

            this.updateStatusBar(benchmark.status);

            if (benchmark.status === 'completed' && benchmark.results) {
                clearInterval(this.pollInterval);
                this.renderResults(benchmark);
                await this.loadHistory(); // Refresh history
            } else if (benchmark.status === 'failed') {
                clearInterval(this.pollInterval);
                alert(`Benchmark failed: ${benchmark.error}`);
            }

        } catch (error) {
            console.error('Failed to check status:', error);
        }
    }

    updateStatusBar(status) {
        const statusBar = document.getElementById('benchmark-status');
        statusBar.className = `status-bar status-${status}`;
        
        switch (status) {
            case 'pending':
                statusBar.textContent = '⏳ Benchmark queued...';
                break;
            case 'running':
                statusBar.textContent = '🔄 Running benchmark... This may take a few minutes.';
                break;
            case 'completed':
                statusBar.textContent = '✅ Benchmark completed!';
                break;
            case 'failed':
                statusBar.textContent = '❌ Benchmark failed.';
                break;
        }
    }

    renderResults(benchmark) {
        const results = benchmark.results;
        
        // Render throughput chart
        this.renderThroughputChart(results);
        
        // Show AI tune chart if applicable
        if (benchmark.benchmark_type === 'ai_tune' && results.rounds_history) {
            document.getElementById('ai-tune-chart-card').classList.remove('hidden');
            this.renderAiTuneChart(results.rounds_history);
        }

        // Render results table
        this.renderResultsTable(benchmark, results);
    }

    renderThroughputChart(results) {
        const ctx = document.getElementById('throughput-chart').getContext('2d');
        
        if (this.charts.throughput) {
            this.charts.throughout.destroy();
        }

        // Handle different result formats
        let labels, prefillData, decodeData;

        if (typeof results === 'object' && !Array.isArray(results)) {
            // Full benchmark with context lengths as keys
            labels = Object.keys(results).map(k => `ctx ${k}`);
            prefillData = Object.values(results).map(r => r.prefill_tps || 0);
            decodeData = Object.values(results).map(r => r.decode_tps || 0);
        } else {
            // Single result or other format
            labels = ['Test'];
            prefillData = [results.prefill_tps || 0];
            decodeData = [results.decode_tps || 0];
        }

        this.charts.throughput = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Prefill (tok/s)',
                        data: prefillData,
                        backgroundColor: '#3b82f6'
                    },
                    {
                        label: 'Decode (tok/s)',
                        data: decodeData,
                        backgroundColor: '#06b6d4'
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Tokens/Second' }
                    }
                }
            }
        });
    }

    renderAiTuneChart(history) {
        const ctx = document.getElementById('ai-tune-chart').getContext('2d');
        
        if (this.charts.aiTune) {
            this.charts.aiTune.destroy();
        }

        const labels = history.map(h => `Round ${h.round}`);
        const tpsData = history.map(h => h.tps || 0);

        this.charts.aiTune = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Decode TPS',
                    data: tpsData,
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    fill: true,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: { display: true, text: 'Tokens/Second' }
                    }
                }
            }
        });
    }

    renderResultsTable(benchmark, results) {
        const tbody = document.querySelector('#results-table tbody');
        tbody.innerHTML = '';

        // Add key metrics
        this.addTableRow(tbody, 'Engine', benchmark.engine);
        this.addTableRow(tbody, 'Type', benchmark.benchmark_type.replace('_', ' ').toUpperCase());
        
        if (results.best_tps) {
            this.addTableRow(tbody, 'Best TPS', results.best_tps.toFixed(2));
        }
        if (results.baseline_tps) {
            this.addTableRow(tbody, 'Baseline TPS', results.baseline_tps.toFixed(2));
        }
        if (results.improvement_pct) {
            this.addTableRow(tbody, 'Improvement', `${results.improvement_pct.toFixed(1)}%`);
        }

        // Add context-specific metrics
        if (typeof results === 'object' && !Array.isArray(results)) {
            Object.entries(results).forEach(([ctx, metrics]) => {
                if (typeof metrics === 'object') {
                    this.addTableRow(tbody, `Context ${ctx} - Prefill`, `${metrics.prefill_tps || 0} tok/s`);
                    this.addTableRow(tbody, `Context ${ctx} - Decode`, `${metrics.decode_tps || 0} tok/s`);
                }
            });
        }
    }

    addTableRow(tbody, metric, value) {
        const row = document.createElement('tr');
        row.innerHTML = `<td>${metric}</td><td>${value}</td>`;
        tbody.appendChild(row);
    }

    async loadHistory() {
        try {
            const response = await fetch(`${this.apiBase}/api/history?limit=20`);
            const history = await response.json();
            
            this.renderHistory(history);
        } catch (error) {
            console.error('Failed to load history:', error);
        }
    }

    renderHistory(benchmarks) {
        const container = document.getElementById('history-list');
        
        if (!benchmarks.length) {
            container.innerHTML = '<p style="color: var(--text-secondary)">No benchmarks yet.</p>';
            return;
        }

        container.innerHTML = '';
        
        benchmarks.forEach(bm => {
            const item = document.createElement('div');
            item.className = 'history-item';
            
            const statusClass = `status-${bm.status}`;
            
            item.innerHTML = `
                <div class="history-meta">
                    <span class="history-id">${bm.id}</span>
                    <span class="history-status ${statusClass}">${bm.status}</span>
                </div>
                <div>${bm.model_path.split('/').pop()} | ${bm.engine} | ${bm.benchmark_type}</div>
            `;

            item.addEventListener('click', () => {
                this.loadBenchmarkById(bm.id);
            });

            container.appendChild(item);
        });
    }

    async loadBenchmarkById(id) {
        try {
            const response = await fetch(`${this.apiBase}/api/benchmarks/${id}`);
            const benchmark = await response.json();
            
            if (benchmark.results) {
                this.renderResults(benchmark);
                document.getElementById('results-panel').classList.remove('hidden');
            }
        } catch (error) {
            console.error('Failed to load benchmark:', error);
        }
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LLMTunerApp();
});
