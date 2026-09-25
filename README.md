# TurbidityVision — Image Processing Based Turbidity Estimation

A software prototype for estimating **image blur caused by optical scattering** and using the resulting blur information to estimate turbidity and particle characteristics.

The project follows the intended pipeline:

```text
Projected Pattern
      ↓
Image Capture / Simulated Blur
      ↓
Image Blur / Sharpness Estimation
      ↓
m × n Blur Matrix
      ↓
Normalized Blur-Ratio Matrix (0–1)
      ↓
Scattering Angle
      ↓
Particle Type + Concentration
      ↓
Turbidity (NTU) + Particle Size
```

> **Important:** The current turbidity, particle-size, scattering-angle calibration, and particle classification values are software placeholders. They are suitable for testing the complete pipeline, but they are **not yet physically calibrated measurements** from the real turbidity rig.

---

## 1. Project Objective

The objective is to develop an embedded-friendly image-processing system that can:

- Project a known image pattern through a fluid.
- Capture the resulting image using a camera.
- Divide the captured image into small kernels/cells.
- Calculate a blur/sharpness value for every kernel.
- Generate a 2D blur matrix.
- Normalize the matrix to a 0–1 blur-ratio representation.
- Convert the normalized blur information into scattering-angle information.
- Estimate particle type and concentration.
- Estimate turbidity in NTU.
- Estimate particle size.
- Compare different projected patterns.
- Validate blur sensitivity using simulated Gaussian blur.
- Export the processed blur matrix as CSV.
- Run using a laptop/USB webcam now and support Raspberry Pi camera integration later.

---

## 2. Current Implementation

### 2.1 Projected Image Patterns

Six image patterns are implemented:

1. Checkerboard
2. Vertical stripes
3. Horizontal stripes
4. Dot grid
5. Concentric circles
6. Random binary pattern

The pattern generator creates BGR images that can be used as reference/projected patterns.

The application allows the feature size of the pattern to be changed.

---

## 3. Blur / Sharpness Algorithms Implemented

Three blur/sharpness estimation algorithms have been implemented.

### 3.1 Variance of Laplacian

```text
Laplacian → Variance
```

The image is converted to grayscale and the Laplacian operator is applied.

A higher variance indicates stronger edges and therefore a sharper image.

```python
lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
score = lap.var()
```

**Interpretation:**

```text
Higher score → sharper image
Lower score  → more blurred image
```

This is the default metric in the application.

---

### 3.2 Tenengrad

Tenengrad uses Sobel gradients in the horizontal and vertical directions.

```text
Sobel X
   +
Sobel Y
   ↓
Gradient magnitude²
   ↓
Mean gradient energy
```

Implementation:

```python
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

mag_sq = gx * gx + gy * gy
score = mag_sq.mean()
```

**Interpretation:**

```text
Higher score → sharper image
Lower score  → more blurred image
```

---

### 3.3 FFT High-Frequency Energy

A frequency-domain blur metric is also implemented.

The image is:

1. Windowed using a Hann window.
2. Converted using a 2D FFT.
3. Shifted so that low frequencies are at the center.
4. Converted to energy using squared magnitude.
5. Divided into low-frequency and high-frequency regions.
6. High-frequency energy is calculated as a percentage of total energy.

```text
Image
  ↓
Hann Window
  ↓
2D FFT
  ↓
FFT Shift
  ↓
Magnitude²
  ↓
High-Frequency Energy / Total Energy
```

A sharp pattern contains relatively more high-frequency energy.

```text
Higher high-frequency energy → sharper
Lower high-frequency energy  → blurrier
```

This also provides the DSP/FFT component of the implementation.

---

## 4. Blur Matrix Algorithm

The captured image is divided into an `m × n` grid of kernels.

For example:

```text
        Columns
     1  2  3  4  ... n
   ┌──┬──┬──┬──┬─────┐
 1 │  │  │  │  │     │
 2 │  │  │  │  │     │
 3 │  │  │  │  │     │
 . │  │  │  │  │     │
 m │  │  │  │  │     │
   └──┴──┴──┴──┴─────┘
```

For every kernel:

```text
Image
  ↓
Convert to grayscale
  ↓
Divide into cells
  ↓
Calculate sharpness for each cell
  ↓
Store result
```

The result is:

```text
B =
[b11 b12 b13 ... b1n
 b21 b22 b23 ... b2n
 ...
 bm1 bm2 bm3 ... bmn]
```

Each element represents the raw sharpness value of one image region.

The number of rows is automatically selected to approximately preserve the image aspect ratio.

---

## 5. Blur-Ratio Normalization

The raw matrix contains sharpness values, where:

```text
Higher value = sharper
Lower value  = blurrier
```

For visualization and downstream processing, the values are normalized to a **0–1 blur ratio**.

First:

```text
sharpness_norm = (value - minimum) / (maximum - minimum)
```

Then:

```text
blur_ratio = 1 - sharpness_norm
```

Therefore:

```text
0 → sharpest region
1 → blurriest region
```

Example:

```text
Raw sharpness:
[100 200 300
 150 250 350]

Normalized blur ratio:
[1.00 0.60 0.20
 0.83 0.40 0.00]
```

> The current normalization is **relative to the current frame**. It is not an absolute physical blur calibration.

---

## 6. Blur Heatmap

The normalized blur-ratio matrix is converted into a heatmap.

The heatmap represents:

```text
0 → sharp
1 → blurry
```

The matrix is resized back to the image dimensions and displayed with cell boundaries and optional numerical annotations.

This provides a visual representation of how blur/scattering varies spatially across the captured image.

---

## 7. Scattering Angle Calculation

The project implements the specified relationship:

```text
Scattering Angle = tan⁻¹(Blur Value)
```

The normalized blur-ratio matrix is used as the blur value:

```python
angle = np.degrees(np.arctan(blur_ratio))
```

Therefore, with a normalized input from 0 to 1:

```text
blur ratio = 0 → angle = 0°
blur ratio = 1 → angle = 45°
```

The application displays a scattering-angle heatmap and calculates:

- Mean scattering angle
- Minimum angle
- Maximum angle
- Standard deviation

> This is currently a **software placeholder mapping**. A physically accurate optical scattering angle must be obtained from calibration against the actual LED/tube/pattern/camera setup and known particle standards.

---

## 8. Particle Identification Algorithm

A lightweight particle-identification pipeline has been implemented.

### Feature extraction

The following features are extracted:

1. Mean blur ratio
2. Standard deviation of blur ratio
3. Mean scattering angle
4. Standard deviation of scattering angle
5. Relative spatial contrast of the raw blur matrix
6. FFT high-frequency score

Feature vector:

```text
[
 mean blur,
 blur variation,
 mean angle,
 angle variation,
 spatial contrast,
 FFT score
]
```

### Reference library

The current reference library contains three placeholder particle classes:

```text
fine_colloidal
medium_silt
coarse_sediment
```

Their current simulated blur-radius ranges are:

```text
fine_colloidal   : 1–3 px
medium_silt      : 4–7 px
coarse_sediment  : 8–13 px
```

These classes are generated using synthetic Gaussian blur.

---

## 9. Particle Classification Algorithm

The current classifier is a **nearest-centroid classifier in normalized feature space**.

For each class:

```text
z = (feature - centroid) / spread
```

Then the Euclidean distance is calculated.

The class with the smallest distance is selected.

```text
Input image
    ↓
Blur matrix
    ↓
Blur ratio
    ↓
Scattering angle
    ↓
Feature extraction
    ↓
Z-score normalization
    ↓
Euclidean distance to class centroids
    ↓
Nearest class
```

No machine-learning library is required for this classifier.

---

## 10. Concentration Estimation

A placeholder concentration-band algorithm is implemented.

The mean blur ratio is multiplied by a class-specific concentration gain.

The result is divided into:

```text
Low
Medium
High
```

Current thresholds:

```text
score < 0.33 → Low
score < 0.66 → Medium
otherwise    → High
```

This is an architectural/software placeholder.

For the final system, concentration must be calibrated using samples with known particle concentrations.

---

## 11. Turbidity Estimation

A calibration model is implemented for converting blur ratio into NTU.

Current relationship:

```text
NTU = slope × (blur_ratio × 100) + intercept
```

The software currently uses adjustable placeholder coefficients.

The calibration class also contains a least-squares fitting method:

```text
Input:
    known blur-ratio values
    known NTU values

Output:
    slope
    intercept
```

The fitting uses:

```python
np.linalg.lstsq(...)
```

### Required future calibration

Real samples should be collected using:

```text
Known turbidity standard
        ↓
Projected pattern
        ↓
Physical fluid rig
        ↓
Camera capture
        ↓
Blur matrix
        ↓
Mean/feature extraction
        ↓
Known NTU pairing
        ↓
Calibration curve
```

Only after this calibration should the system's NTU output be treated as a physical measurement.

---

## 12. Particle Size Estimation

A placeholder relationship is implemented:

```text
particle_size_µm = particle_k × blur_radius_px
```

`particle_k` is currently adjustable.

For the final implementation, `particle_k` should be obtained from experiments using particles with known physical sizes.

---

## 13. Synthetic Blur Simulation

Before the physical turbidity rig is fully calibrated, the project supports software-only testing.

Gaussian blur is used as a stand-in for increasing optical scattering:

```text
Sharp Pattern
     ↓
Gaussian Blur
     ↓
Blurred Pattern
```

The blur radius controls the simulated blur strength.

Optional additive noise can also be added.

This allows the complete pipeline to be tested without physical fluid samples.

---

## 14. Sensitivity Sweep

A sensitivity experiment is implemented in the GUI.

The system tests blur radii from:

```text
0 px → 12 px
```

For each radius:

```text
Generate blurred pattern
        ↓
Calculate sharpness score
        ↓
Store score
```

A line chart is generated showing:

```text
Blur Radius vs Sharpness
```

The expected behavior is a decreasing sharpness score as blur increases.

This can be used to compare the sensitivity of:

- Laplacian
- Tenengrad
- FFT

---

## 15. Pattern Comparison

The application compares the implemented patterns.

For each pattern:

```text
Sharp pattern
     ↓
Calculate Laplacian sharpness
     ↓
Apply 6 px Gaussian blur
     ↓
Calculate blurred sharpness
     ↓
Calculate sensitivity
```

Sensitivity is calculated as:

```text
Sensitivity =
(sharp_score - blurred_score) / sharp_score
```

The results are displayed as a bar chart.

Patterns implemented:

```text
Checkerboard
Vertical Stripes
Horizontal Stripes
Dot Grid
Concentric Circles
Random Binary
```

---

## 16. Camera Implementation

A unified camera interface is implemented.

### Laptop / USB webcam

OpenCV is used:

```python
cv2.VideoCapture(...)
```

### Raspberry Pi

The same interface can use:

```text
picamera2
```

The application automatically selects the Pi camera when `picamera2` is available, otherwise it uses the webcam.

Therefore, the downstream processing pipeline does not need to know which camera backend is being used.

---

## 17. Desktop GUI

A Tkinter-based desktop application has been implemented.

The GUI provides:

### Pattern controls

- Pattern selection
- Feature-size adjustment
- Reference-pattern generation

### Input controls

- Simulated blur
- Live camera
- Blur radius
- Noise percentage
- Camera capture

### Blur-matrix controls

- Grid-column selection
- Laplacian
- Tenengrad
- FFT

### Calibration controls

- Turbidity slope
- Particle-size coefficient

### Validation

- Sensitivity sweep
- Pattern comparison

### Export

- Blur matrix CSV
- Input snapshot
- Blur heatmap

---

## 18. Data Export

The application can export:

```text
row
col
sharpness_raw
blur_ratio
scattering_angle_deg
```

as a CSV file.

Example:

```text
row,col,sharpness_raw,blur_ratio,scattering_angle_deg
0,0,123.4567,0.8234,39.49
0,1,245.1234,0.5123,27.12
...
```

This exported data can later be used for:

- Calibration
- Regression
- Statistical analysis
- ML model development
- Experimental comparison

---

## 19. Current Algorithms Summary

| Component | Algorithm / Method | Status |
|---|---|---|
| Pattern generation | Checkerboard, stripes, dots, circles, random binary | Implemented |
| Blur estimation | Variance of Laplacian | Implemented |
| Blur estimation | Tenengrad / Sobel gradient energy | Implemented |
| Blur estimation | FFT high-frequency energy ratio | Implemented |
| Blur matrix | m × n kernel grid | Implemented |
| Normalization | Min-max + inversion to blur ratio | Implemented |
| Heatmap | OpenCV color mapping | Implemented |
| Scattering | `tan⁻¹(blur ratio)` | Implemented as placeholder |
| Particle features | Statistical + angle + FFT features | Implemented |
| Particle classification | Nearest centroid + z-score distance | Implemented as placeholder |
| Concentration | Low / Medium / High banding | Implemented as placeholder |
| Turbidity | Linear calibration model | Implemented as placeholder |
| Calibration fitting | Least-squares regression | Implemented |
| Particle size | Blur-radius scaling | Implemented as placeholder |
| Blur simulation | Gaussian blur + optional noise | Implemented |
| Sensitivity analysis | Blur-radius sweep | Implemented |
| Pattern comparison | Relative sensitivity | Implemented |
| Camera | OpenCV webcam | Implemented |
| Raspberry Pi camera | picamera2 backend | Supported |
| CSV export | Blur/scattering data | Implemented |
| GUI | Tkinter desktop application | Implemented |

---

## 20. Project Structure

```text
TurbidityVision/
│
├── app.py
├── blur.py
├── blur_matrix.py
├── scattering.py
├── calibration.py
├── particle_id.py
├── patterns.py
├── camera.py
├── utils.py
├── requirements.txt
│
├── assets/
│   └── generated pattern images
│
├── captured/
│   └── camera frames
│
└── results/
    ├── blur matrices
    ├── heatmaps
    ├── calibration files
    ├── sensitivity charts
    └── pattern comparison charts
```

---

# Installation

## 21. Requirements

Recommended:

```text
Python 3.10+
```

The current dependency file contains:

```text
numpy
opencv-python
Pillow
```

`picamera2` is optional and is only required when using a Raspberry Pi camera.

---

## 22. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

---

## 23. Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

---

## 24. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If you are installing manually:

```bash
pip install numpy opencv-python Pillow
```

---

## 25. Run the Application

```bash
python app.py
```

The GUI should open with:

```text
TURBIDITYVISION
software test rig
```

---

## 26. Generate Pattern Assets

```bash
python patterns.py
```

This generates the reference pattern images under:

```text
assets/
```

---

## 27. Test Blur Algorithms

```bash
python blur.py
```

Or provide an image:

```bash
python blur.py assets/checkerboard.png
```

The program prints:

```text
variance of Laplacian
Tenengrad
FFT high-frequency energy
```

---

## 28. Test Blur Matrix

```bash
python blur_matrix.py
```

Or:

```bash
python blur_matrix.py assets/checkerboard.png
```

The output is saved under:

```text
results/blur_matrix_demo.png
```

---

## 29. Test Scattering Calculation

```bash
python scattering.py
```

This calculates the scattering-angle statistics from the normalized blur matrix.

---

## 30. Test Calibration Module

```bash
python calibration.py
```

This currently demonstrates the placeholder calibration model and saves:

```text
results/calibration.json
```

---

## 31. Test Camera

For a laptop/USB webcam:

```bash
python camera.py
```

A test frame is saved as:

```text
captured/camera_test.png
```

---

# Raspberry Pi Setup

## 32. Install Raspberry Pi Camera Support

On Raspberry Pi, install `picamera2` using the Raspberry Pi OS package manager:

```bash
sudo apt update
sudo apt install -y python3-picamera2
```

Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

The camera module automatically uses `picamera2` when available.

---

# Git Commands

## 33. First-Time Git Setup

If the project is not already a Git repository:

```bash
cd /path/to/TurbidityVision
git init
```

Set your Git identity if required:

```bash
git config --global user.name "YOUR_NAME"
git config --global user.email "YOUR_EMAIL"
```

---

## 34. Create `.gitignore`

Create a `.gitignore` file:

```gitignore
.venv/
__pycache__/
*.pyc

captured/
results/

.DS_Store

.idea/
.vscode/

*.log
```

Do not commit generated camera captures, result files, virtual environments, or IDE files unless you specifically want them in the repository.

---

## 35. Add the Project to Git

```bash
git add .
```

Check what will be committed:

```bash
git status
```

Commit:

```bash
git commit -m "Initial turbidity image processing implementation"
```

---

## 36. Connect GitHub Repository

If you already created an empty GitHub repository:

```bash
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
```

Check:

```bash
git remote -v
```

---

## 37. Push to GitHub

For a new repository:

```bash
git branch -M main
git push -u origin main
```

For subsequent changes:

```bash
git add .
git commit -m "Update turbidity image processing algorithms"
git push
```

---

# Typical Development Workflow

```text
1. Generate pattern
        ↓
2. Project pattern / simulate blur
        ↓
3. Capture image
        ↓
4. Convert to grayscale
        ↓
5. Divide image into kernels
        ↓
6. Calculate sharpness per kernel
        ↓
7. Generate blur matrix
        ↓
8. Normalize to 0–1 blur ratio
        ↓
9. Generate heatmap
        ↓
10. Convert blur ratio to scattering angle
        ↓
11. Extract features
        ↓
12. Identify particle type
        ↓
13. Estimate concentration
        ↓
14. Estimate turbidity
        ↓
15. Estimate particle size
        ↓
16. Export results
```

---

# Important Limitations / Next Steps

The current software successfully demonstrates the complete image-processing architecture, but the following work is still required for a physically validated turbidity measurement system:

### 1. Real physical calibration

Collect images using:

- Known turbidity standards
- Known particle sizes
- Known particle concentrations
- The actual LED/tube/pattern/camera setup

Then fit the calibration model using real measurements.

### 2. Validate scattering-angle relationship

The current:

```text
angle = tan⁻¹(blur_ratio)
```

relationship is a software representation of the project concept. It must be experimentally calibrated before interpreting the result as a physical scattering angle.

### 3. Replace synthetic particle signatures

The current particle library is generated using synthetic Gaussian blur.

Replace it with real captured images from known particle standards.

### 4. Train a real regression model

After collecting sufficient experimental data, possible regression models can be evaluated using features such as:

```text
Blur matrix statistics
Mean blur
Blur standard deviation
Scattering-angle statistics
FFT energy
Pattern type
Pattern frequency
Spatial features
```

The final model can predict:

```text
Turbidity (NTU)
Particle size (µm)
```

### 5. Raspberry Pi deployment

Move the validated pipeline to Raspberry Pi and optimize:

- Processing time
- Memory usage
- Camera capture
- UI/device interface
- Continuous measurement
- Data logging

---

# Technology Stack

```text
Python
OpenCV
NumPy
Pillow
Tkinter
OpenCV FFT / NumPy FFT
picamera2 (optional Raspberry Pi)
Git / GitHub
```

---

# Current Status

### Completed

- [x] Pattern generation
- [x] Synthetic Gaussian blur
- [x] Noise simulation
- [x] Variance of Laplacian
- [x] Tenengrad
- [x] FFT high-frequency energy
- [x] Blur matrix
- [x] 0–1 blur-ratio normalization
- [x] Blur heatmap
- [x] Scattering-angle conversion
- [x] Particle feature extraction
- [x] Nearest-centroid particle identification
- [x] Concentration band estimation
- [x] Placeholder turbidity estimation
- [x] Placeholder particle-size estimation
- [x] Calibration least-squares fitting
- [x] Sensitivity sweep
- [x] Pattern comparison
- [x] Webcam capture
- [x] Raspberry Pi camera interface
- [x] CSV export
- [x] Tkinter GUI

### Pending for experimental validation

- [ ] Physical turbidity-rig calibration
- [ ] Real turbidity-standard dataset
- [ ] Real particle-size dataset
- [ ] Real particle-concentration dataset
- [ ] Experimental validation of scattering angle
- [ ] Regression model trained on real measurements
- [ ] Raspberry Pi field deployment
- [ ] Accuracy/error analysis against a reference turbidity meter

---

## License

Add your preferred license here, for example:

```text
MIT License
```

if the repository is intended to use the MIT license.
