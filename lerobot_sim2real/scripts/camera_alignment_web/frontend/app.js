class CameraAlignmentApp {
    constructor() {
        this.ws = null;
        this.canvas = document.getElementById('cameraCanvas');
        this.ctx = this.canvas.getContext('2d');
        this.connectionStatus = document.getElementById('connectionStatus');
        this.frameCount = document.getElementById('frameCount');
        this.loadingOverlay = document.getElementById('loadingOverlay');
        this.notification = document.getElementById('notification');
        
        // Position displays
        this.posX = document.getElementById('posX');
        this.posY = document.getElementById('posY');
        this.posZ = document.getElementById('posZ');
        this.fov = document.getElementById('fov');
        
        // Control state
        this.activeControls = new Set();
        this.isConnected = false;
        
        // Initialize
        this.init();
    }
    
    init() {
        this.setupWebSocket();
        this.setupEventListeners();
        this.setupKeyboardControls();
    }
    
    setupWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.hostname}:8000/ws`;
        
        this.ws = new WebSocket(wsUrl);
        
        this.ws.onopen = () => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.updateConnectionStatus(true);
            this.loadingOverlay.classList.add('hidden');
            
            // Request current configuration
            this.sendMessage({
                type: 'get_config'
            });
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.handleMessage(data);
        };
        
        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.showNotification('Connection error', 'error');
        };
        
        this.ws.onclose = () => {
            console.log('WebSocket disconnected');
            this.isConnected = false;
            this.updateConnectionStatus(false);
            this.loadingOverlay.classList.remove('hidden');
            
            // Attempt to reconnect after 2 seconds
            setTimeout(() => {
                if (!this.isConnected) {
                    this.setupWebSocket();
                }
            }, 2000);
        };
    }
    
    handleMessage(data) {
        switch (data.type) {
            case 'frame':
                this.updateFrame(data.frame);
                this.updateState(data.state);
                break;
            case 'config':
                this.handleConfig(data.config);
                break;
            case 'config_saved':
                this.showNotification('Configuration saved successfully', 'success');
                break;
        }
    }
    
    updateFrame(frameData) {
        const img = new Image();
        img.onload = () => {
            // Resize canvas to match image
            if (this.canvas.width !== img.width || this.canvas.height !== img.height) {
                this.canvas.width = img.width;
                this.canvas.height = img.height;
            }
            this.ctx.drawImage(img, 0, 0);
        };
        img.src = 'data:image/jpeg;base64,' + frameData;
    }
    
    updateState(state) {
        if (state.camera_position) {
            this.posX.textContent = state.camera_position[0].toFixed(3);
            this.posY.textContent = state.camera_position[1].toFixed(3);
            this.posZ.textContent = state.camera_position[2].toFixed(3);
        }
        if (state.fov !== undefined) {
            this.fov.textContent = state.fov.toFixed(3);
        }
        if (state.frame_count !== undefined) {
            this.frameCount.textContent = state.frame_count;
        }
    }
    
    handleConfig(config) {
        console.log('Loaded configuration:', config);
        this.showNotification('Configuration loaded', 'success');
    }
    
    updateConnectionStatus(connected) {
        const statusText = this.connectionStatus.querySelector('.status-text');
        if (connected) {
            this.connectionStatus.classList.add('connected');
            statusText.textContent = 'Connected';
        } else {
            this.connectionStatus.classList.remove('connected');
            statusText.textContent = 'Disconnected';
        }
    }
    
    setupEventListeners() {
        // Control buttons
        const controlButtons = document.querySelectorAll('.control-btn');
        controlButtons.forEach(button => {
            const control = button.dataset.control;
            
            // Mouse events
            button.addEventListener('mousedown', () => {
                this.startControl(control);
                button.classList.add('active');
            });
            
            button.addEventListener('mouseup', () => {
                this.stopControl(control);
                button.classList.remove('active');
            });
            
            button.addEventListener('mouseleave', () => {
                this.stopControl(control);
                button.classList.remove('active');
            });
            
            // Touch events for mobile
            button.addEventListener('touchstart', (e) => {
                e.preventDefault();
                this.startControl(control);
                button.classList.add('active');
            });
            
            button.addEventListener('touchend', (e) => {
                e.preventDefault();
                this.stopControl(control);
                button.classList.remove('active');
            });
        });
        
        // Save configuration button
        const saveBtn = document.getElementById('saveConfigBtn');
        saveBtn.addEventListener('click', () => {
            this.saveConfiguration();
        });
    }
    
    setupKeyboardControls() {
        const keyMap = {
            'w': 'forward',
            'W': 'forward',
            's': 'backward',
            'S': 'backward',
            'a': 'left',
            'A': 'left',
            'd': 'right',
            'D': 'right',
            'u': 'up',
            'U': 'up',
            'j': 'down',
            'J': 'down',
            ',': 'fov_decrease',
            '.': 'fov_increase',
            'p': 'save',
            'P': 'save',
            'Backspace': 'reset'
        };
        
        document.addEventListener('keydown', (e) => {
            const control = keyMap[e.key];
            if (control) {
                e.preventDefault();
                
                if (control === 'save') {
                    this.saveConfiguration();
                } else {
                    this.startControl(control);
                    
                    // Highlight corresponding button
                    const button = document.querySelector(`[data-control="${control}"]`);
                    if (button) {
                        button.classList.add('active');
                    }
                }
            }
        });
        
        document.addEventListener('keyup', (e) => {
            const control = keyMap[e.key];
            if (control && control !== 'save') {
                e.preventDefault();
                this.stopControl(control);
                
                // Remove highlight from button
                const button = document.querySelector(`[data-control="${control}"]`);
                if (button) {
                    button.classList.remove('active');
                }
            }
        });
    }
    
    startControl(control) {
        this.activeControls.add(control);
        this.sendControlUpdate();
    }
    
    stopControl(control) {
        this.activeControls.delete(control);
        this.sendControlUpdate();
    }
    
    sendControlUpdate() {
        if (!this.isConnected) return;
        
        const controls = {
            forward: this.activeControls.has('forward'),
            backward: this.activeControls.has('backward'),
            left: this.activeControls.has('left'),
            right: this.activeControls.has('right'),
            up: this.activeControls.has('up'),
            down: this.activeControls.has('down'),
            fov_decrease: this.activeControls.has('fov_decrease'),
            fov_increase: this.activeControls.has('fov_increase'),
            reset: this.activeControls.has('reset')
        };
        
        this.sendMessage({
            type: 'controls',
            controls: controls
        });
        
        // Clear reset after sending
        if (this.activeControls.has('reset')) {
            this.activeControls.delete('reset');
        }
    }
    
    saveConfiguration() {
        if (!this.isConnected) {
            this.showNotification('Not connected to server', 'error');
            return;
        }
        
        this.sendMessage({
            type: 'save_config'
        });
    }
    
    sendMessage(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        }
    }
    
    showNotification(message, type = 'info') {
        const notificationText = this.notification.querySelector('.notification-text');
        notificationText.textContent = message;
        
        // Remove all type classes
        this.notification.classList.remove('success', 'error', 'info');
        
        // Add appropriate type class
        if (type) {
            this.notification.classList.add(type);
        }
        
        // Show notification
        this.notification.classList.add('show');
        
        // Hide after 3 seconds
        setTimeout(() => {
            this.notification.classList.remove('show');
        }, 3000);
    }
}

// Initialize app when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new CameraAlignmentApp();
});