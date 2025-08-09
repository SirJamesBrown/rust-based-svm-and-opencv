# Window and Door Detection with OpenCV and Rust

An intelligent computer vision application that detects windows, doors, and buildings in architectural images using OpenCV and Machine Learning (SVM) with Rust. The application features a modular architecture for maintainability and extensibility.

## Features

- **Multi-Object Detection**: Detects windows, doors, and building structures in architectural images
- **Machine Learning**: Uses Support Vector Machine (SVM) for intelligent classification with confidence scoring
- **Interactive Training**: Manual retraining system with visual feedback to improve detection accuracy
- **Modular Architecture**: Clean separation of concerns across multiple modules
- **Visual Display Mode**: Optional window display for real-time result viewing
- **Batch Processing**: Processes multiple image formats (JPG, PNG, WebP, AVIF)
- **Feature Analysis**: Analyzes geometric properties like size, aspect ratio, area, and position

## Project Structure

```
ocv/
├── src/
│   ├── main.rs                    # Application entry point and orchestration
│   ├── detection.rs               # WindowDoorDetector - core detection logic
│   ├── training.rs                # ModelTrainer - interactive training system
│   ├── models.rs                  # Data structures and type definitions
│   └── utils.rs                   # Common utilities and helper functions
├── images/                        # Sample images for testing
│   ├── image.jpg
│   ├── banner.webp
│   ├── Custom-Luxury-Home-Dallas.webp
│   ├── 3017-Creekhollow-Denton-TX-Model-3.avif
│   └── ...
├── training_data.csv              # Training dataset for the SVM model
├── svm_window_door_model.xml      # Trained SVM model file
├── Cargo.toml                     # Project dependencies
└── README.md                      # This file
```

## Prerequisites

Before building this project, you need to install OpenCV development libraries on your Linux system:

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install libopencv-dev clang libclang-dev pkg-config
```

### Fedora
```bash
sudo dnf install opencv-devel clang clang-devel pkg-config
```

### Arch Linux
```bash
sudo pacman -S opencv clang pkg-config
```

## Getting Started

### Building the Project

Navigate to the project directory and build:

```bash
cargo build --release
```

### Usage Modes

#### 1. Batch Detection Mode (Default)
Process all images in the `images/` directory:

```bash
cargo run
```

This will:
- Process all supported image formats in the `images/` folder
- Save annotated results as `result_001_filename.jpg`, etc.
- Display detection statistics and summary

#### 2. Visual Display Mode
View results in real-time windows (requires display):

```bash
cargo run -- --show
# or
cargo run -- -s
```

This enables:
- Interactive window display of detection results
- Press any key to advance to the next image
- Scaled images for optimal viewing (max 1000px)

#### 3. Interactive Model Retraining
Improve detection accuracy through manual classification:

```bash
cargo run -- --retrain
# or
cargo run -- -r
```

This will:
1. Process all images and extract potential features
2. Display each feature in a window for manual classification
3. Allow you to label features as:
   - **[W]** Window
   - **[D]** Door  
   - **[S]** Skip (not a window/door)
   - **[Q]** Quit retraining
4. Retrain the SVM model with combined old and new data
5. Save the improved model and updated training dataset

## Module Overview

### `detection.rs` - WindowDoorDetector
- **Core Detection Logic**: Edge detection, contour analysis, and feature extraction
- **SVM Integration**: Model loading and prediction with confidence scoring
- **Heuristic Fallback**: Geometric rules when SVM model is unavailable
- **Result Visualization**: Drawing bounding boxes, labels, and confidence scores
- **Building Detection**: Simplified building structure identification

### `training.rs` - ModelTrainer  
- **Interactive Training**: Visual feature classification interface
- **Data Management**: Loading/saving training data in CSV format
- **Model Training**: SVM configuration and training with feature vectors
- **Feature Extraction**: Converting image features to training samples

### `models.rs` - Data Structures
- **ProcessingResult**: Stores detection counts per image
- **TrainingData**: Training sample with features and labels
- **DetectionCandidate**: Individual detection with confidence and geometry

### `utils.rs` - Common Utilities
- **Image Scaling**: Automatic scaling for display windows
- **Building Detection**: Simplified building mask generation
- **File Utilities**: Supported format validation and helpers

## Detection Pipeline

### 1. Image Preprocessing
- Convert to grayscale for edge detection
- Apply building mask (excludes sky and ground regions)

### 2. Feature Detection
- Canny edge detection with optimal thresholds
- Contour extraction and geometric filtering
- Size and aspect ratio validation

### 3. Classification
- **SVM Prediction**: Uses trained model with 7-feature vectors
- **Heuristic Fallback**: Geometric rules for door/window characteristics
- **Confidence Scoring**: Color-coded results (green=high, orange=medium, red=low)

### 4. Result Generation
- Bounding box annotation with labels
- Building structure overlay
- Summary statistics and file output

## Training Data Format

The training data (`training_data.csv`) contains these features:

| Feature | Description |
|---------|-------------|
| Width | Feature width in pixels |
| Height | Feature height in pixels |
| Area | Calculated area (width × height) |
| AspectRatio | Width/height ratio |
| X | X-coordinate position |
| Y | Y-coordinate position |
| Confidence | Detection confidence score |
| Label | 0 for windows, 1 for doors |
| Type | Human-readable label ("Window" or "Door") |

## Example Output

```
OpenCV version: 4.8.0
Processing image: images/Custom-Luxury-Home-Dallas.webp
============================================================
Image loaded successfully! Size: 1920x1080
W1 (85%): 45x32 at (423, 234), ratio: 1.41, confidence: 85.0%
W2 (82%): 38x28 at (521, 245), ratio: 1.36, confidence: 82.0%
D1 (91%): 28x76 at (612, 298), ratio: 0.37, confidence: 91.0%
Building 1: Area: 156420.0 (13.42% of image), Bounds: 890x420 at (515, 180)
Processed image saved as: result_001_Custom-Luxury-Home-Dallas.webp

============================================================
PROCESSING SUMMARY
============================================================
Total images processed: 8
Total detections across all images:
  Windows: 42
  Doors: 12  
  Buildings: 15
  Image 1: Custom-Luxury-Home-Dallas.webp - W:8 D:2 B:1
  Image 2: banner.webp - W:12 D:3 B:2
  ...
```

## Supported Image Formats

- **JPEG** (.jpg, .jpeg)
- **PNG** (.png)  
- **WebP** (.webp)
- **AVIF** (.avif)

## Dependencies

```toml
[dependencies]
opencv = "0.91"  # Computer vision and machine learning
```

## Model Performance

- **Initial Heuristics**: Basic geometric classification rules
- **SVM Enhancement**: Machine learning with RBF kernel
- **Continuous Learning**: Interactive retraining improves accuracy
- **Feature Engineering**: 7-dimensional feature vectors for classification

## Contributing

To improve the detection system:

1. **Add Training Data**: 
   ```bash
   # Place new images in images/ folder
   cargo run -- --retrain
   ```

2. **Extend Detection**: Modify detection logic in `src/detection.rs`

3. **Add Features**: Enhance feature extraction in `src/training.rs`

4. **Improve Classification**: Adjust SVM parameters or add new features

## License

This project is licensed under the MIT License.