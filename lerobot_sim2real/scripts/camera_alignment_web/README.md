# Camera Alignment Web App

A web-based camera alignment tool for sim2real robot control, featuring real-time camera position adjustment and overlay visualization.

## Features

- **Real-time Camera Stream**: Live overlay of simulation and real camera feeds
- **Interactive Controls**: Adjust camera position and FOV using mouse, keyboard, or touch controls
- **Configuration Persistence**: Save and load camera alignment settings
- **WebSocket Communication**: Low-latency real-time updates
- **Responsive Design**: Works on desktop and mobile devices

## Architecture

```
camera_alignment_web/
├── backend/
│   └── main.py          # FastAPI server with WebSocket support
├── frontend/
│   ├── index.html       # Main HTML interface
│   ├── styles.css       # CSS styling
│   └── app.js          # JavaScript client application
└── run.sh              # Launch script
```

## Quick Start

### Prerequisites
Make sure you have installed the lerobot-sim2real package with its dependencies:
```bash
# From the root of lerobot-sim2real repository
pip install -e .
```

### Running the Web App

#### Option 1: Using the launch script
```bash
./run.sh
```

#### Option 2: Direct execution
```bash
cd backend
python main.py
```

Then open your browser and navigate to: http://localhost:8000

## Controls

### Keyboard Shortcuts
- **W/A/S/D**: Move camera forward/left/backward/right
- **U/J**: Move camera up/down
- **,/.**: Decrease/increase FOV
- **P**: Save current configuration
- **Backspace**: Reset camera to initial position

### Mouse/Touch
- Click and hold the on-screen buttons to control camera movement
- Touch controls supported on mobile devices

## API Endpoints

- `WebSocket /ws`: Real-time bidirectional communication
- `GET /api/status`: Get server status
- `GET /api/config`: Get current camera configuration
- `POST /api/config/save`: Save current configuration

## Configuration

Camera settings are saved to `camera_alignment_config.json` and include:
- Camera position offset (X, Y, Z)
- Field of view offset
- Timestamp of last save

## Development

The app follows a client-server architecture similar to the assembler0 simulator:

- **Backend**: FastAPI server handling robot control and camera streaming
- **Frontend**: Plain HTML/CSS/JavaScript for maximum compatibility
- **Communication**: WebSocket for real-time updates, REST API for configuration

## Requirements

- Python 3.8+
- FastAPI
- OpenCV
- PyTorch
- Gymnasium
- ManiSkill environment