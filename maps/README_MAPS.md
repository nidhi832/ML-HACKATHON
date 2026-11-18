# Folium Map Screenshots Guide for Hackathon Report

Interactive HTML maps have been generated in the `maps/` directory:
- Best Trip (00000052): `maps/student_test_best_00000052.html` (Accuracy: 95.04%)
- Typical Trip (00000056): `maps/student_test_typical_00000056.html` (Accuracy: 87.11%)
- Worst Trip (00000077): `maps/student_test_worst_00000077.html` (Accuracy: 76.99%)

### Visual Features:
1. **Raw GPS (Blue circles):** Every raw timestamped GPS observation point.
2. **Reference Route (Green solid polyline):** Ground-truth sequence of road arcs traversed.
3. **Predicted Route (Red dashed polyline):** Top-1 model candidate selection.
4. **Misclassifications (Orange circles):** Highlights points where the top-1 prediction diverged from ground truth.
5. **Road Network (Grey thin lines):** Surrounding directed road arcs.
6. **Interactive LayerControl (Top Right):** Toggle any layer on or off.

### Capturing PNG Screenshots for the Report / Presentation:
1. Double click any `.html` file above to open it in your browser (Chrome/Edge/Firefox).
2. Adjust the zoom level to show the full trajectory and road network clearly.
3. Use Windows Snipping Tool (`Win + Shift + S`) or browser full-page screenshot (`Ctrl + Shift + I` -> `Ctrl + Shift + P` -> "Capture full size screenshot").
4. Save screenshots to `report/figures/map_screenshot_<best|typical|worst>.png`.
