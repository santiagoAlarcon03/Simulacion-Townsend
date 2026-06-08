# Townsend Discharge 3D

Interactive 3D simulation of a simplified Townsend discharge using PyQt6, PyVista, and NumPy.

Features
- Stage selection and particle count control
- Start, pause or resume, and reset
- 3D particle visualization
- Basic plots of particle count over time

Run
1. Create and activate a Python environment
2. Install requirements: pip install -r requirements.txt
3. Run: python main.py

Debug (optional)
- Install debugpy: pip install debugpy
- Launch with debugpy: python -m debugpy --listen 5678 main.py
- Attach from VS Code: Run and Debug -> Python: Attach -> port 5678

Tests
- Run all tests: python -m unittest discover -s tests

Presentation checklist
- Use the controls to select the start stage and particle count
- Start, pause or resume, and reset to show stage transitions
- Show the 3D particle evolution and the time series plot
- Mention the model limitations listed in docs/limitations.md

Notes
- The physics model is simplified for academic use.
- Some modules are placeholders and should be replaced with full models as needed.
