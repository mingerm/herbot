#!/bin/bash
# Start Herbot Web UI

echo "🌿 Starting Herbot Web UI..."
echo ""
echo "The web interface will be available at:"
echo "  Local:   http://localhost:8501"
echo "  Network: http://$(hostname -I | awk '{print $1}'):8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd /home/pi/Ai-embed/projects/Herbot
streamlit run web_ui.py --server.port 8501 --server.address 0.0.0.0
