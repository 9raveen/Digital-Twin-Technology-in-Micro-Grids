#!/bin/bash
# Quick Start: Digital Twin Microgrid Simulation Dashboard

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/Scripts/activate

# Check dependencies
echo "Checking dependencies..."
python -m pip list | grep -E "streamlit|pandas|numpy|plotly|scikit|joblib" > /dev/null

# Run dashboard
echo ""
echo "================================================"
echo "Starting Simulation Dashboard..."
echo "================================================"
echo ""
echo "Dashboard URL: http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop"
echo ""

streamlit run simulation_dashboard.py --client.toolbarMode=minimal
