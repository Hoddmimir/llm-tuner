// LLM Tuner - Frontend Application

class LLMTunerApp {
    constructor() {
        this.apiBase = '';
        this.currentBenchmark = null;
        this.pollInterval = null;
        this.gpuPollInterval = null;
        this.charts = {};
        this.eventSource = null;
        this.models = [];

        this.init();
    }

    async init() {
        await this.loadSettings();
        this.setupTheme();
        await this.loadHardware();
        await this.loadEngines();
        await this.loadHistory();
        await this.loadModels();
        this.setupEventListeners();
        
        // Initialize benchmark description
        this.updateBenchDescription();
        
        // GPU graph history (60 points = 60 seconds at 1Hz)
        this.gpuHistory = { gpu: [], mem: [] };
        this.maxGraphPoints = 60;

        // Start continuous GPU polling every second
        this.startGpuPolling();
    }

    /* --- Benchmark Type Descriptions --- */

    updateBenchDescription() {
        const type = document.getElementById('benchmark-type').value;
        const descEl = document.getElementById('bench-description');
        if (!descEl) return;
        
        const descriptions = {
            'quick': 'Runs a single-context benchmark (2048 tokens). Fast test to check basic inference speed. Good for quick model comparisons.',
            'full': 'Runs a full benchmark across multiple context lengths (512, 1024, 2048, 4096). Measures prefill and decode throughput at each length. Best for general performance profiling.',
            'ai_tune': 'Uses the model itself as an optimizer - it proposes inference flag combinations to maximize throughput. Runs baseline first, then iteratively tests AI-suggested configs. Each round restarts llama-server with new flags.',
            'grid_search': 'Systematically tests every combination of threads (4/8/16), batch size (512-4096), and cache type (q4_0/f16). Tests 72 total configurations. Each config starts a fresh llama-server instance. Slow but thorough.'
        };
        
        descEl.textContent = descriptions[type] || '';
    }

    /* --- GPU Polling --- */

    startGpuPolling() {
        if (this.gpuPollInterval) {
            clearInterval(this.gpuPollInterval);
        }
        this.gpuPollInterval = setInterval(() => {
            this.pollGpuStats();
        }, 1000); // Every second
    }

    async pollGpuStats() {
        try {
            const response = await fetch(`${this.apiBase}/api/gpu`);
            if (!response.ok) return;
            const data = await response.json();

            // Update stats labels
            const tempLabel = document.getElementById('gpu-temp');
            const utilLabel = document.getElementById('gpu-util-label');
            const memLabel = document.getElementById('gpu-mem-label');

            if (data.gpu_utilization_pct !== undefined && utilLabel) {
                utilLabel.textContent = data.gpu_utilization_pct + '%';
            }

            if (data.gpu_memory_used_mb !== undefined && data.gpu_memory_total_mb && memLabel) {
                const usedGB = (data.gpu_memory_used_mb / 1024).toFixed(1);
                const totalGB = (data.gpu_memory_total_mb / 1024).toFixed(1);
                memLabel.textContent = usedGB + ' / ' + totalGB + ' GB';

                // Track memory % for graph
                const memPct = Math.min((data.gpu_memory_used_mb / data.gpu_memory_total_mb) * 100, 100);
                this.gpuHistory.mem.push(memPct);
            } else if (memLabel && !this.gpuHistory.mem.length) {
                this.gpuHistory.mem.push(0);
            }

            // Track GPU util for graph
            const gpuVal = data.gpu_utilization_pct !== undefined ? Math.min(data.gpu_utilization_pct, 100) : 0;
            this.gpuHistory.gpu.push(gpuVal);

            // Trim history
            if (this.gpuHistory.gpu.length > this.maxGraphPoints) {
                this.gpuHistory.gpu.shift();
            }
            if (this.gpuHistory.mem.length > this.maxGraphPoints) {
                this.gpuHistory.mem.shift();
            }

            // Update timestamp
            const tsEl = document.getElementById('monitor-timestamp');
            if (tsEl) tsEl.textContent = new Date().toLocaleTimeString();

            // Draw canvas graph
            this.drawGpuGraph();
        } catch (e) {
            console.error('GPU poll error:', e);
        }
    }

    /* --- Cancel Benchmark --- */

    async cancelBenchmark() {
        if (!this.currentBenchmark) return;
        
        try {
            const response = await fetch(`${this.apiBase}/api/benchmarks/${this.currentBenchmark}/cancel`, {
                method: 'POST'
            });
            const result = await response.json();
            
            this.addStatusMessage(`⚠️ ${result.message}`, 'warning');
            
            document.getElementById('cancel-btn').classList.add('hidden');
        } catch (e) {
            console.error('Failed to cancel:', e);
            this.addStatusMessage('❌ Failed to cancel benchmark', 'error');
        }
    }

    /* --- Settings & Theme --- */

    async loadSettings() {
        try {
            const response = await fetch(`${this.apiBase}/api/settings`);
            this.settings = await response.json();
        } catch (e) {
            this.settings = { models_directory: "", theme: "light" };
        }
    }

    setupTheme() {
        const theme = this.settings.theme || 'light';
        document.documentElement.setAttribute('data-theme', theme);
        this.updateThemeButtons(theme);
    }

    updateThemeButtons(theme) {
        document.getElementById('theme-light').classList.toggle('active', theme === 'light');
        document.getElementById('theme-dark').classList.toggle('active', theme === 'dark');
    }

    async setTheme(theme) {
        document.documentElement.setAttribute('data-theme', theme);
        this.updateThemeButtons(theme);
        this.settings.theme = theme;
        try {
            await fetch(`${this.apiBase}/api/settings`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.settings)
            });
        } catch (e) { console.error('Failed to save theme:', e); }
    }

    /* --- Model Discovery --- */

    async loadModels() {
        try {
            const response = await fetch(`${this.apiBase}/api/models`);
            const data = await response.json();
            this.models = data.models || [];
            
            // Populate settings modal dropdown
            const selectGroup = document.getElementById('model-select-group');
            const select = document.getElementById('model-select');
            
            if (this.models.length > 0) {
                selectGroup.classList.remove('hidden');
                select.innerHTML = '';
                this.models.forEach(m => {
                    const option = document.createElement('option');
                    option.value = m.path;
                    option.textContent = `${m.name} (${m.size_mb} MB)`;
                    select.appendChild(option);
                });
            } else {
                if (data.directory) {
                    selectGroup.classList.remove('hidden');
                    select.innerHTML = '<option value="">-- No .gguf files found --</option>';
                } else {
                    selectGroup.classList.add('hidden');
                }
            }

            // Populate form dropdown
            this.populateFormDropdown();

        } catch (e) { console.error('Failed to load models:', e); }
    }

    populateFormDropdown() {
        const dropdown = document.getElementById('model-dropdown');
        
        if (this.models.length === 0) {
            dropdown.classList.add('hidden');
            return;
        }
        
        dropdown.classList.remove('hidden');
        dropdown.innerHTML = '';
        
        this.models.forEach(m => {
            const option = document.createElement('option');
            option.value = m.path;
            option.textContent = `${m.name} (${m.size_mb} MB)`;
            dropdown.appendChild(option);
        });

        // Click to set model path
        dropdown.addEventListener('change', () => {
            if (dropdown.value) {
                document.getElementById('model-path').value = dropdown.value;
            }
        });
    }

    async scanModels() {
        const dirInput = document.getElementById('models-dir');
        const dirPath = dirInput.value.trim();
        
        this.settings.models_directory = dirPath;
        try {
            await fetch(`${this.apiBase}/api/settings`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(this.settings)
            });
        } catch (e) { console.error('Failed to save directory:', e); }
        
        await this.loadModels();
    }

    /* --- Modal --- */

    openSettings() {
        const modal = document.getElementById('settings-modal');
        modal.classList.remove('hidden');
        document.getElementById('models-dir').value = this.settings.models_directory || '';
    }

    closeSettings() {
        document.getElementById('settings-modal').classList.add('hidden');
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

        // Settings modal
        document.getElementById('settings-btn').addEventListener('click', () => this.openSettings());
        document.getElementById('close-settings').addEventListener('click', () => this.closeSettings());
        document.getElementById('modal-backdrop').addEventListener('click', () => this.closeSettings());

        // Theme toggle
        document.getElementById('theme-light').addEventListener('click', () => this.setTheme('light'));
        document.getElementById('theme-dark').addEventListener('click', () => this.setTheme('dark'));

        // Models directory scan
        document.getElementById('scan-models-btn').addEventListener('click', () => this.scanModels());

        // Use model from settings modal
        document.getElementById('use-model-btn').addEventListener('click', () => {
            const select = document.getElementById('model-select');
            if (select.value) {
                document.getElementById('model-path').value = select.value;
                
                // Also update form dropdown selection
                const formDropdown = document.getElementById('model-dropdown');
                for (let i = 0; i < formDropdown.options.length; i++) {
                    if (formDropdown.options[i].value === select.value) {
                        formDropdown.selectedIndex = i;
                        break;
                    }
                }
                
                this.closeSettings();
            }
        });

        // Detailed results toggle
        const toggleBtn = document.getElementById('toggle-details-btn');
        if (toggleBtn) {
            toggleBtn.addEventListener('click', () => {
                const panel = document.getElementById('details-panel');
                const isHidden = panel.classList.contains('hidden');
                panel.classList.toggle('hidden');
                toggleBtn.textContent = isHidden ? 'Hide Detailed Breakdown ▲' : 'Show Detailed Breakdown ▼';
            });
        }

        // Keyboard shortcut - Escape to close modal
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                this.closeSettings();
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
        
        // Also update GPU name in always-visible monitor panel
        const gpuNameLabel = document.getElementById('gpu-name-label');
        if (gpuNameLabel) {
            gpuNameLabel.textContent = hw.gpu?.name || 'Unknown GPU';
        }
        
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
                rounds: parseInt(document.getElementById('tune-rounds').value),
                custom_flags: document.getElementById('custom-flags')?.value || ''
            };

            const response = await fetch(`${this.apiBase}/api/benchmarks`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(requestData)
            });

            const result = await response.json();
            
            this.currentBenchmark = result.id;
            this.showResultsPanel();
            this.showBenchmarkControls();
            this.clearStatusLog();
            this.hideViewLogsButton();
            this.addStatusMessage('⏳ Benchmark queued...', 'info');
            
            // Start SSE connection for real-time updates
            this.connectSSE(result.id);
            
            // Also start polling as fallback
            this.startPolling(result.id);
            
        } catch (error) {
            console.error('Failed to start benchmark:', error);
            alert(`Error: ${error.message}`);
        } finally {
            startBtn.disabled = false;
            startBtn.textContent = 'Start Benchmark';
        }
    }

    connectSSE(benchmarkId) {
        // Close existing connection
        if (this.eventSource) {
            this.eventSource.close();
        }
        
        const url = `${this.apiBase}/api/benchmarks/${benchmarkId}/stream`;
        this.eventSource = new EventSource(url);
        
        this.eventSource.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                
                if (data.type === 'progress') {
                    // Progress update
                    this.updateProgress(data.value, data.status);
                    
                    // Update GPU monitor with live stats
                    if (data.gpu_memory_used_mb !== undefined) {
                        this.updateGpuMonitor(data);
                    }
                    
                    if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
                        this.eventSource.close();
                        this.hideBenchmarkControls();
                    }
                } else if (data.message) {
                    // Status message
                    const icon = this.getStatusIcon(data.level);
                    this.addStatusMessage(`${icon} ${data.message}`, data.level);
                    
                    // Update status bar based on level
                    if (data.level === 'success') {
                        this.updateStatusBar('completed');
                    } else if (data.level === 'error') {
                        this.updateStatusBar('failed');
                    }
                }
            } catch (e) {
                console.error('SSE parse error:', e);
            }
        };
        
        this.eventSource.onerror = (err) => {
            console.log('SSE connection closed');
            this.eventSource.close();
        };
    }

    getStatusIcon(level) {
        switch (level) {
            case 'success': return '✅';
            case 'error': return '❌';
            case 'warning': return '⚠️';
            default: return 'ℹ️';
        }
    }

    clearStatusLog() {
        const log = document.getElementById('status-log');
        if (log) {
            log.innerHTML = '';
        }
    }

    addStatusMessage(message, level = 'info') {
        let log = document.getElementById('status-log');
        
        if (!log) {
            const statusBar = document.getElementById('benchmark-status');
            log = document.createElement('div');
            log.id = 'status-log';
            log.className = 'terminal-log';
            statusBar.parentNode.insertBefore(log, statusBar.nextSibling);
        }
        
        const entry = document.createElement('div');
        entry.className = 'log-entry log-' + level;
        
        const time = new Date().toLocaleTimeString();
        entry.innerHTML = '<span class="log-time">[' + time + ']</span> <span class="log-msg">' + message + '</span>';
        
        log.appendChild(entry);
        
        // Auto-scroll to bottom
        log.scrollTop = log.scrollHeight;
    }

    updateProgress(progress, status) {
        const statusBar = document.getElementById('benchmark-status');
        
        if (progress > 0 && progress <= 100) {
            statusBar.textContent = `🔄 Running benchmark... ${progress}% complete`;
            
            // Update any progress bar
            const progressBar = document.querySelector('.progress-bar-fill');
            if (progressBar) {
                progressBar.style.width = `${progress}%`;
            }
        }
        
        this.updateStatusBar(status);
    }

    parseContextLengths() {
        const input = document.getElementById('context-lengths').value;
        return input.split(',').map(x => parseInt(x.trim())).filter(x => !isNaN(x));
    }

    showResultsPanel() {
        document.getElementById('results-panel').classList.remove('hidden');
    }

    showBenchmarkControls() {
        const controls = document.getElementById('benchmark-controls');
        const progressContainer = document.getElementById('benchmark-progress-container');
        if (controls) controls.classList.remove('hidden');
        if (progressContainer) progressContainer.classList.remove('hidden');
    }

    hideBenchmarkControls() {
        const controls = document.getElementById('benchmark-controls');
        const progressContainer = document.getElementById('benchmark-progress-container');
        if (controls) controls.classList.add('hidden');
        if (progressContainer) progressContainer.classList.add('hidden');
        
        // Show "View Logs" button after benchmark completes/fails
        this.showViewLogsButton();
    }

    showViewLogsButton() {
        let logBtn = document.getElementById('view-logs-btn');
        if (!logBtn) {
            const statusLog = document.getElementById('status-log');
            if (statusLog) {
                logBtn = document.createElement('button');
                logBtn.id = 'view-logs-btn';
                logBtn.className = 'btn btn-secondary';
                logBtn.textContent = '📋 View Full Logs';
                logBtn.onclick = () => this.showLogsForBenchmark(this.currentBenchmark);
                statusLog.parentNode.insertBefore(logBtn, statusLog.nextSibling);
            }
        } else {
            logBtn.classList.remove('hidden');
        }
    }

    hideViewLogsButton() {
        const logBtn = document.getElementById('view-logs-btn');
        if (logBtn) logBtn.classList.add('hidden');
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

            if (benchmark.live && benchmark.live.messages) {
                // Update from live status
                benchmark.live.messages.forEach(msg => {
                    const icon = this.getStatusIcon(msg.level);
                    this.addStatusMessage(`${icon} ${msg.message}`, msg.level);
                });
                
                if (benchmark.live.progress > 0) {
                    this.updateProgress(benchmark.live.progress, benchmark.status);
                }
            }

            if (benchmark.status === 'completed' && benchmark.results) {
                clearInterval(this.pollInterval);
                if (this.eventSource) this.eventSource.close();
                this.hideBenchmarkControls();
                this.renderResults(benchmark);
                await this.loadHistory(); // Refresh history
            } else if (benchmark.status === 'failed' || benchmark.status === 'cancelled') {
                clearInterval(this.pollInterval);
                if (this.eventSource) this.eventSource.close();
                this.hideBenchmarkControls();
                
                const errorMsg = benchmark.error || 'Unknown error';
                this.addStatusMessage(`❌ Benchmark failed: ${errorMsg}`, 'error');
            }

        } catch (error) {
            console.error('Failed to check status:', error);
        }
    }

    updateStatusBar(status) {
        const statusBar = document.getElementById('benchmark-status');
        if (!statusBar) return;
        
        statusBar.className = `status-bar status-${status}`;
        
        switch (status) {
            case 'pending':
                statusBar.textContent = '⏳ Benchmark queued...';
                break;
            case 'running':
                statusBar.textContent = '🔄 Running benchmark... Check log below for details.';
                break;
            case 'completed':
                statusBar.textContent = '✅ Benchmark completed!';
                break;
            case 'failed':
                statusBar.textContent = '❌ Benchmark failed. Check log for error details.';
                break;
        }
    }

    updateGpuMonitor(data) {
        // Update stats labels
        const utilLabel = document.getElementById('gpu-util-label');
        const memLabel = document.getElementById('gpu-mem-label');
        
        if (data.gpu_utilization_pct !== undefined && utilLabel) {
            utilLabel.textContent = data.gpu_utilization_pct + '%';
        }
        
        if (data.gpu_memory_used_mb !== undefined && data.gpu_memory_total_mb && memLabel) {
            const usedGB = (data.gpu_memory_used_mb / 1024).toFixed(1);
            const totalGB = (data.gpu_memory_total_mb / 1024).toFixed(1);
            memLabel.textContent = usedGB + ' / ' + totalGB + ' GB';
            
            // Track memory % for graph
            const memPct = Math.min((data.gpu_memory_used_mb / data.gpu_memory_total_mb) * 100, 100);
            this.gpuHistory.mem.push(memPct);
        } else if (memLabel && !this.gpuHistory.mem.length) {
            this.gpuHistory.mem.push(0);
        }
        
        // Track GPU util for graph
        const gpuVal = data.gpu_utilization_pct !== undefined ? Math.min(data.gpu_utilization_pct, 100) : 0;
        this.gpuHistory.gpu.push(gpuVal);
        
        // Trim history
        if (this.gpuHistory.gpu.length > this.maxGraphPoints) {
            this.gpuHistory.gpu.shift();
        }
        if (this.gpuHistory.mem.length > this.maxGraphPoints) {
            this.gpuHistory.mem.shift();
        }
        
        // Draw canvas graph
        this.drawGpuGraph();
        
        // Update progress
        const progressText = document.getElementById('gpu-progress-text');
        const progressBar = document.getElementById('gpu-progress-bar');
        if (data.current_context && progressText) {
            progressText.textContent = data.current_context;
        }
        if (data.value !== undefined && progressBar) {
            progressBar.style.width = data.value + '%';
        }
        
        // Update timestamp
        const tsEl = document.getElementById('monitor-timestamp');
        if (tsEl) tsEl.textContent = new Date().toLocaleTimeString();
    }

    drawGpuGraph() {
        const canvas = document.getElementById('gpu-graph-canvas');
        if (!canvas) return;

        // Use actual rendered dimensions with device pixel ratio for sharp rendering
        const dpr = window.devicePixelRatio || 1;
        const displayWidth = canvas.clientWidth;
        const displayHeight = canvas.clientHeight;

        // Set internal canvas resolution to match display * DPR
        if (canvas.width !== Math.round(displayWidth * dpr) || canvas.height !== Math.round(displayHeight * dpr)) {
            canvas.width = Math.round(displayWidth * dpr);
            canvas.height = Math.round(displayHeight * dpr);
        }

        const ctx = canvas.getContext('2d');
        // Scale context to DPR so drawing uses logical coordinates
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

        const width = displayWidth;
        const height = displayHeight;

        // Clear
        ctx.clearRect(0, 0, width, height);
        
        // Draw subtle grid lines (every 25%)
        ctx.strokeStyle = 'rgba(0,0,0,0.08)';
        ctx.lineWidth = 0.5;
        for (let i = 1; i < 4; i++) {
            const y = (height * i) / 4;
            ctx.beginPath();
            ctx.moveTo(0, y);
            ctx.lineTo(width, y);
            ctx.stroke();
        }
        
        // Draw GPU utilization line (steel blue)
        this.drawLineGraph(ctx, this.gpuHistory.gpu, width, height, '#475569');
        
        // Draw memory utilization line (cyan) on top
        this.drawLineGraph(ctx, this.gpuHistory.mem, width, height, '#06b6d4');
    }

    drawLineGraph(ctx, points, width, height, color) {
        if (points.length < 2) return;
        
        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';
        
        // Draw the line
        ctx.beginPath();
        points.forEach((value, idx) => {
            const x = (idx / (this.maxGraphPoints - 1)) * width;
            const y = height - (value / 100) * height;
            
            if (idx === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
        
        // Draw subtle gradient fill under the line
        ctx.fillStyle = color + '15';
        ctx.beginPath();
        points.forEach((value, idx) => {
            const x = (idx / (this.maxGraphPoints - 1)) * width;
            const y = height - (value / 100) * height;
            
            if (idx === 0) {
                ctx.moveTo(x, height);
                ctx.lineTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        for (let i = points.length - 1; i >= 0; i--) {
            const x = (i / (this.maxGraphPoints - 1)) * width;
            ctx.lineTo(x, height);
        }
        ctx.fill();
    }

    renderResults(benchmark) {
        const results = benchmark.results;
        
        this.renderThroughputChart(results);
        
        if (benchmark.benchmark_type === 'ai_tune' && results.rounds_history) {
            document.getElementById('ai-tune-chart-card').classList.remove('hidden');
            this.renderAiTuneChart(results.rounds_history);
        }

        if (benchmark.benchmark_type === 'full' || benchmark.benchmark_type === 'quick') {
            this.renderLatencyBreakdown(results);
        }

        if (benchmark.benchmark_type === 'grid_search' && results.all_results) {
            this.renderGridSearchResults(results);
        }

        this.renderResultsTable(benchmark, results);
    }

    renderLatencyBreakdown(results) {
        let card = document.getElementById('latency-chart-card');
        if (!card) {
            card = document.createElement('div');
            card.id = 'latency-chart-card';
            card.className = 'chart-card';
            card.innerHTML = '<h3>Latency Breakdown (ms/token)</h3><canvas id="latency-chart"></canvas>';
            const container = document.querySelector('.charts-container');
            if (container) container.appendChild(card);
        } else {
            card.classList.remove('hidden');
        }
        
        const canvasCtx = document.getElementById('latency-chart').getContext('2d');
        if (this.charts.latency) this.charts.latency.destroy();

        let labels, prefillData, decodeData;
        if (typeof results === 'object' && !Array.isArray(results)) {
            labels = Object.keys(results).map(k => `ctx ${k}`);
            const vals = Object.values(results);
            prefillData = vals.map(r => r.prefill_tps ? (1000 / r.prefill_tps) : 0);
            decodeData = vals.map(r => r.decode_tps ? (1000 / r.decode_tps) : 0);
        } else {
            labels = ['Test'];
            prefillData = results.prefill_tps ? [1000 / results.prefill_tps] : [0];
            decodeData = results.decode_tps ? [1000 / results.decode_tps] : [0];
        }

        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
        const textColor = isDark ? '#9ca3af' : '#6b7280';

        this.charts.latency = new Chart(canvasCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    { label: 'Prefill ms/tok', data: prefillData, backgroundColor: isDark ? 'rgba(251, 146, 60, 0.8)' : 'rgba(249, 115, 22, 0.8)', borderColor: '#f97316', borderWidth: 1, borderRadius: 3, borderSkipped: false },
                    { label: 'Decode ms/tok', data: decodeData, backgroundColor: isDark ? 'rgba(244, 63, 94, 0.8)' : 'rgba(225, 29, 72, 0.8)', borderColor: '#e11d48', borderWidth: 1, borderRadius: 3, borderSkipped: false }
                ]
            },
            options: {
                indexAxis: 'y', responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: textColor, font: { size: 11 }, usePointStyle: true, pointStyle: 'rectRounded', padding: 16 } },
                    tooltip: { backgroundColor: isDark ? '#2d2d2d' : '#ffffff', titleColor: isDark ? '#e5e5e5' : '#1a1a1a', bodyColor: textColor, borderColor: isDark ? '#404040' : '#e5e7eb', borderWidth: 1, cornerRadius: 6, padding: 8, callbacks: { label: c => ` ${c.dataset.label}: ${c.parsed.x.toFixed(2)} ms/tok` } }
                },
                scales: { x: { beginAtZero: true, grid: { color: gridColor, drawBorder: false }, ticks: { color: textColor, font: { size: 10 }, callback: v => v.toFixed(1) + 'ms' } }, y: { grid: { display: false }, ticks: { color: textColor, font: { size: 11 } } } }
            }
        });
    }

    renderGridSearchResults(results) {
        const card = document.getElementById('grid-chart-card');
        if (!card) return;
        card.classList.remove('hidden');
        const ctx = document.getElementById('grid-search-chart').getContext('2d');
        if (this.charts.gridSearch) this.charts.gridSearch.destroy();

        const allResults = results.all_results || [];
        const datasets = {};
        
        allResults.forEach(r => {
            const key = r.config.cache_type;
            if (!datasets[key]) datasets[key] = [];
            datasets[key].push({ x: r.config.threads, y: r.config.batch_size, tps: r.tps });
        });

        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const textColor = isDark ? '#9ca3af' : '#6b7280';
        
        this.charts.gridSearch = new Chart(ctx, {
            type: 'scatter',
            data: {
                datasets: Object.entries(datasets).map(([key, vals], i) => ({
                    label: `cache=${key}`,
                    data: vals.map(d => ({ x: d.x, y: d.y, r: Math.max(d.tps / 3, 4), tps: d.tps })),
                    backgroundColor: i === 0 ? 'rgba(59, 130, 246, 0.7)' : 'rgba(6, 182, 212, 0.7)',
                    borderColor: i === 0 ? '#3b82f6' : '#06b6d4', borderWidth: 1
                }))
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: textColor, font: { size: 11 }, usePointStyle: true, padding: 16 } },
                    tooltip: { backgroundColor: isDark ? '#2d2d2d' : '#ffffff', titleColor: isDark ? '#e5e5e5' : '#1a1a1a', bodyColor: textColor, borderColor: isDark ? '#404040' : '#e5e7eb', borderWidth: 1, cornerRadius: 6, padding: 8, callbacks: { label: c => ` threads=${c.raw.x}, batch=${c.raw.y} -> ${c.raw.tps?.toFixed(1) || '?'} tok/s` } }
                },
                scales: { x: { type: 'category', labels: [4, 8, 16], title: { display: true, text: 'Threads', color: textColor }, grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' } }, y: { type: 'category', labels: [512, 1024, 2048, 4096], title: { display: true, text: 'Batch Size', color: textColor }, grid: { color: isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)' } } }
            }
        });
    }

    renderThroughputChart(results) {
        const ctx = document.getElementById('throughput-chart').getContext('2d');
        
        if (this.charts.throughput) {
            this.charts.throughput.destroy();
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

        // Get theme colors for charts
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
        const textColor = isDark ? '#9ca3af' : '#6b7280';

        this.charts.throughput = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [
                    {
                        label: 'Prefill (tok/s)',
                        data: prefillData,
                        backgroundColor: isDark ? 'rgba(96, 165, 250, 0.8)' : 'rgba(59, 130, 246, 0.8)',
                        borderColor: isDark ? '#60a5fa' : '#3b82f6',
                        borderWidth: 1,
                        borderRadius: 4,
                        borderSkipped: false
                    },
                    {
                        label: 'Decode (tok/s)',
                        data: decodeData,
                        backgroundColor: isDark ? 'rgba(34, 211, 238, 0.8)' : 'rgba(6, 182, 212, 0.8)',
                        borderColor: isDark ? '#22d3ee' : '#06b6d4',
                        borderWidth: 1,
                        borderRadius: 4,
                        borderSkipped: false
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: textColor,
                            font: { size: 12, family: '-apple-system, BlinkMacSystemFont, sans-serif' },
                            usePointStyle: true,
                            pointStyle: 'rectRounded',
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: isDark ? '#2d2d2d' : '#ffffff',
                        titleColor: isDark ? '#e5e5e5' : '#1a1a1a',
                        bodyColor: isDark ? '#9ca3af' : '#6b7280',
                        borderColor: isDark ? '#404040' : '#e5e7eb',
                        borderWidth: 1,
                        cornerRadius: 6,
                        padding: 10,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return ` ${context.dataset.label}: ${context.parsed.y.toFixed(1)} tok/s`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: textColor,
                            font: { size: 11 }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            font: { size: 11 },
                            callback: (v) => v.toFixed(0) + ' tok/s'
                        }
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

        // Get theme colors for charts
        const isDark = document.documentElement.getAttribute('data-theme') === 'dark';
        const gridColor = isDark ? 'rgba(255,255,255,0.06)' : 'rgba(0,0,0,0.06)';
        const textColor = isDark ? '#9ca3af' : '#6b7280';

        this.charts.aiTune = new Chart(ctx, {
            type: 'line',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Decode TPS',
                    data: tpsData,
                    borderColor: isDark ? '#34d399' : '#10b981',
                    backgroundColor: isDark ? 'rgba(52, 211, 153, 0.1)' : 'rgba(16, 185, 129, 0.08)',
                    fill: true,
                    tension: 0.3,
                    pointRadius: 4,
                    pointHoverRadius: 6,
                    pointBackgroundColor: isDark ? '#34d399' : '#10b981',
                    pointBorderColor: isDark ? '#1e1e1e' : '#ffffff',
                    pointBorderWidth: 2,
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        labels: {
                            color: textColor,
                            font: { size: 12, family: '-apple-system, BlinkMacSystemFont, sans-serif' },
                            usePointStyle: true,
                            pointStyle: 'circle',
                            padding: 20
                        }
                    },
                    tooltip: {
                        backgroundColor: isDark ? '#2d2d2d' : '#ffffff',
                        titleColor: isDark ? '#e5e5e5' : '#1a1a1a',
                        bodyColor: isDark ? '#9ca3af' : '#6b7280',
                        borderColor: isDark ? '#404040' : '#e5e7eb',
                        borderWidth: 1,
                        cornerRadius: 6,
                        padding: 10,
                        displayColors: true,
                        callbacks: {
                            label: function(context) {
                                return ` TPS: ${context.parsed.y.toFixed(2)} tok/s`;
                            }
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { display: false },
                        ticks: {
                            color: textColor,
                            font: { size: 11 }
                        }
                    },
                    y: {
                        beginAtZero: true,
                        grid: {
                            color: gridColor,
                            drawBorder: false
                        },
                        ticks: {
                            color: textColor,
                            font: { size: 11 },
                            callback: (v) => v.toFixed(0) + ' tok/s'
                        }
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

        // Add detailed breakdown table for context-length results
        if (typeof results === 'object' && !Array.isArray(results)) {
            const detailEntries = Object.entries(results).filter(([_, v]) => typeof v === 'object');
            
            if (detailEntries.length > 0) {
                const toggleContainer = document.getElementById('detail-toggle-container');
                const detailsPanel = document.getElementById('details-panel');
                const toggleBtn = document.getElementById('toggle-details-btn');
                const detailsTbody = document.querySelector('#details-table tbody');
                
                if (toggleContainer && detailsPanel && toggleBtn && detailsTbody) {
                    toggleContainer.classList.remove('hidden');
                    detailsTbody.innerHTML = '';
                    
                    detailEntries.forEach(([ctx, metrics]) => {
                        const row = document.createElement('tr');
                        const prefill = metrics.prefill_tps || 0;
                        const decode = metrics.decode_tps || 0;
                        const prompt_ms = metrics.prompt_ms_per_tok || 0;
                        const decode_ms = metrics.decode_ms_per_tok || 0;
                        
                        row.innerHTML = `
                            <td>${ctx}</td>
                            <td>${prefill.toFixed(1)}</td>
                            <td>${decode.toFixed(1)}</td>
                            <td>${prompt_ms.toFixed(1)}</td>
                            <td>${decode_ms.toFixed(1)}</td>
                        `;
                        detailsTbody.appendChild(row);
                    });
                    
                    // Toggle button handler
                    toggleBtn.onclick = () => {
                        const isHidden = detailsPanel.classList.contains('hidden');
                        detailsPanel.classList.toggle('hidden');
                        toggleBtn.textContent = isHidden ? 'Hide Detailed Breakdown ▲' : 'Show Detailed Breakdown ▼';
                    };
                }
            }
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

    /* --- Log Viewer --- */

    async refreshLogsList() {
        try {
            const response = await fetch(`${this.apiBase}/api/logs`);
            const logs = await response.json();
            
            const select = document.getElementById('log-select');
            select.innerHTML = '<option value="">-- Select benchmark log --</option>';
            
            logs.forEach(log => {
                const option = document.createElement('option');
                option.value = log.id;
                const sizeKB = (log.size_bytes / 1024).toFixed(1);
                option.textContent = `${log.id} (${log.lines} lines, ${sizeKB} KB) - ${new Date(log.modified_at).toLocaleString()}`;
                select.appendChild(option);
            });
        } catch (error) {
            console.error('Failed to load logs list:', error);
        }
    }

    async loadLog(benchmarkId) {
        if (!benchmarkId) return;
        
        const logContent = document.getElementById('log-content');
        const logBenchId = document.getElementById('log-bench-id');
        
        logBenchId.textContent = `(${benchmarkId})`;
        logContent.innerHTML = '<div class="log-entry log-info">Loading logs...</div>';
        
        // Show the logs panel if hidden
        document.getElementById('logs-panel').classList.remove('hidden');
        
        try {
            const response = await fetch(`${this.apiBase}/api/benchmarks/${benchmarkId}/logs`);
            const data = await response.json();
            
            logContent.innerHTML = '';
            
            if (data.entries && data.entries.length > 0) {
                // Render structured entries
                data.entries.forEach(entry => {
                    const div = document.createElement('div');
                    div.className = `log-entry log-${entry.level || 'info'}`;
                    
                    const time = new Date(entry.ts).toLocaleTimeString();
                    const source = entry.source ? `[${entry.source}]` : '';
                    div.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-level">[${(entry.level || 'info').toUpperCase()}]</span> ${source} <span class="log-msg">${this.escapeHtml(entry.msg)}</span>`;
                    
                    logContent.appendChild(div);
                });
            } else if (data.file_content) {
                // Render raw file content
                const lines = data.file_content.split('\n').filter(l => l.trim());
                lines.forEach(line => {
                    const div = document.createElement('div');
                    div.className = 'log-entry log-info';
                    
                    // Try to parse structured format: [timestamp] [LEVEL] [source] message
                    const match = line.match(/^\[([^\]]+)\]\s*\[([^\]]+)\]\s*\[([^\]]+)\]\s*(.*)$/);
                    if (match) {
                        const [, time, level, source, msg] = match;
                        div.className = `log-entry log-${level.toLowerCase()}`;
                        div.innerHTML = `<span class="log-time">[${time}]</span> <span class="log-level">[${level.toUpperCase()}]</span> [${source}] <span class="log-msg">${this.escapeHtml(msg)}</span>`;
                    } else {
                        div.textContent = line;
                    }
                    
                    logContent.appendChild(div);
                });
            } else {
                logContent.innerHTML = '<div class="log-entry log-warning">No logs found for this benchmark.</div>';
            }
            
            // Auto-scroll to bottom
            if (document.getElementById('auto-scroll-log').checked) {
                logContent.scrollTop = logContent.scrollHeight;
            }
        } catch (error) {
            console.error('Failed to load logs:', error);
            logContent.innerHTML = `<div class="log-entry log-error">Error loading logs: ${this.escapeHtml(error.message)}</div>`;
        }
    }

    escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    showLogsForBenchmark(benchmarkId) {
        // Show logs panel and load the log for this benchmark
        document.getElementById('logs-panel').classList.remove('hidden');
        this.refreshLogsList().then(() => {
            const select = document.getElementById('log-select');
            select.value = benchmarkId;
            this.loadLog(benchmarkId);
        });
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    window.app = new LLMTunerApp();
});