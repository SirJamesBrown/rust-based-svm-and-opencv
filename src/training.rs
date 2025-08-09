use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc, ml, highgui};
use std::fs::{read_dir, File};
use std::path::Path;
use std::io::{Write, BufRead, BufReader};

use crate::models::TrainingData;
use crate::utils::{scale_for_display, detect_buildings, get_supported_extensions};

pub struct ModelTrainer {
}

impl ModelTrainer {
    pub fn new() -> Self {
        ModelTrainer {}
    }

    /// Interactive model retraining mode
    pub fn retrain_model_interactive(&mut self) -> opencv::Result<()> {
        println!("=== INTERACTIVE MODEL RETRAINING ===");
        println!("This will process all images and allow you to correct classifications");
        
        // Check if images directory exists
        let images_dir = "images";
        if !Path::new(images_dir).exists() {
            println!("Error: '{}' directory not found", images_dir);
            return Ok(());
        }
        
        let mut training_data = Vec::new();
        let supported_extensions = get_supported_extensions();
        
        // Process each image for interactive training
        let entries = read_dir(images_dir).map_err(|e| {
            opencv::Error::new(opencv::core::StsError, format!("Failed to read directory: {}", e))
        })?;
        
        for entry in entries {
            let entry = entry.map_err(|e| {
                opencv::Error::new(opencv::core::StsError, format!("Failed to read directory entry: {}", e))
            })?;
            
            let path = entry.path();
            if let Some(extension) = path.extension() {
                let ext_str = extension.to_string_lossy().to_lowercase();
                if supported_extensions.contains(&ext_str.as_str()) {
                    let image_path = path.to_string_lossy();
                    println!("\n============================================================");
                    println!("RETRAINING: Processing image: {}", image_path);
                    println!("============================================================");
                    
                    let img = imgcodecs::imread(&image_path, imgcodecs::IMREAD_COLOR)?;
                    if img.empty() {
                        continue;
                    }
                    
                    let new_data = self.extract_features_for_retraining(&img, &image_path)?;
                    training_data.extend(new_data);
                }
            }
        }
        
        if training_data.is_empty() {
            println!("No training data extracted. Exiting.");
            return Ok(());
        }
        
        // Load existing training data if available
        let existing_data = self.load_existing_training_data()?;
        training_data.extend(existing_data);
        
        println!("\n=== TRAINING DATA SUMMARY ===");
        let window_count = training_data.iter().filter(|d| d.label == 0).count();
        let door_count = training_data.iter().filter(|d| d.label == 1).count();
        println!("Total samples: {}", training_data.len());
        println!("Windows: {}, Doors: {}", window_count, door_count);
        
        // Train new SVM model
        let model = self.train_svm_model(&training_data)?;
        
        // Save the model and updated training data
        model.save("svm_window_door_model.xml")?;
        self.save_training_data_to_csv(&training_data)?;
        
        println!("\n=== RETRAINING COMPLETE ===");
        println!("New model saved as 'svm_window_door_model.xml'");
        println!("Training data updated in 'training_data.csv'");
        
        Ok(())
    }

    /// Extract features from an image for retraining
    fn extract_features_for_retraining(&self, image: &Mat, image_path: &str) -> opencv::Result<Vec<TrainingData>> {
        let mut gray = Mat::default();
        imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        let building_mask = detect_buildings(image)?;
        
        let mut edges = Mat::default();
        imgproc::canny(&gray, &mut edges, 50.0, 150.0, 3, false)?;
        
        let mut masked_edges = Mat::default();
        core::bitwise_and(&edges, &edges, &mut masked_edges, &building_mask)?;
        
        let mut feature_contours = opencv::types::VectorOfVectorOfPoint::new();
        imgproc::find_contours(&masked_edges, &mut feature_contours, imgproc::RETR_EXTERNAL, 
                              imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
        
        let mut training_data = Vec::new();
        let filename = Path::new(image_path).file_name().unwrap_or_default().to_string_lossy();
        
        // Create a display image for interactive classification
        let display_image = image.clone();
        let scaled_display = scale_for_display(&display_image)?;
        
        println!("Found {} potential features in {}", feature_contours.len(), filename);
        
        for i in 0..feature_contours.len() {
            let contour = feature_contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            let rect = imgproc::bounding_rect(&contour)?;
            
            // Filter by size
            if area > 200.0 && area < 20000.0 && rect.width > 15 && rect.height > 15 {
                let aspect_ratio = rect.width as f64 / rect.height as f64;
                
                // Draw the current detection
                let mut temp_image = scaled_display.clone();
                let scale_factor = if image.cols() > 1000 { 1000.0 / image.cols() as f64 } else { 1.0 };
                let scaled_rect = core::Rect::new(
                    (rect.x as f64 * scale_factor) as i32,
                    (rect.y as f64 * scale_factor) as i32,
                    (rect.width as f64 * scale_factor) as i32,
                    (rect.height as f64 * scale_factor) as i32,
                );
                
                imgproc::rectangle(&mut temp_image, scaled_rect, 
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0), 3, imgproc::LINE_8, 0)?;
                
                imgproc::put_text(&mut temp_image, &format!("Feature {}", i + 1),
                    core::Point::new(scaled_rect.x, scaled_rect.y - 5),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.6,
                    core::Scalar::new(0.0, 255.0, 0.0, 0.0), 2, imgproc::LINE_8, false)?;
                
                // Show the detection and ask for classification
                let window_title = format!("Classify Feature - {}", filename);
                highgui::named_window(&window_title, highgui::WINDOW_AUTOSIZE)?;
                highgui::imshow(&window_title, &temp_image)?;
                
                println!("\nFeature {}: Size {}x{} at ({}, {}), Aspect Ratio: {:.2}, Area: {:.1}",
                         i + 1, rect.width, rect.height, rect.x, rect.y, aspect_ratio, area);
                println!("Classify this feature:");
                println!("  [W] Window");
                println!("  [D] Door");
                println!("  [S] Skip (not a window/door)");
                println!("  [Q] Quit retraining");
                
                let key = highgui::wait_key(0)?;
                highgui::destroy_window(&window_title)?;
                
                let label = match key as u8 as char {
                    'W' | 'w' => {
                        println!("  -> Classified as WINDOW");
                        Some(0)
                    },
                    'D' | 'd' => {
                        println!("  -> Classified as DOOR");
                        Some(1)
                    },
                    'S' | 's' => {
                        println!("  -> SKIPPED");
                        None
                    },
                    'Q' | 'q' => {
                        println!("  -> QUIT retraining");
                        return Ok(training_data);
                    },
                    _ => {
                        println!("  -> Invalid key, SKIPPING");
                        None
                    }
                };
                
                if let Some(label_value) = label {
                    training_data.push(TrainingData {
                        width: rect.width as f32,
                        height: rect.height as f32,
                        area: (rect.width * rect.height) as f32,
                        aspect_ratio: aspect_ratio as f32,
                        x: rect.x as f32,
                        y: rect.y as f32,
                        confidence: 1.0,
                        label: label_value,
                        type_name: if label_value == 0 { "Window".to_string() } else { "Door".to_string() },
                    });
                }
            }
        }
        
        println!("Added {} new training samples from {}", training_data.len(), filename);
        Ok(training_data)
    }

    /// Load existing training data from CSV
    fn load_existing_training_data(&self) -> opencv::Result<Vec<TrainingData>> {
        let mut training_data = Vec::new();
        
        if !Path::new("training_data.csv").exists() {
            println!("No existing training data found.");
            return Ok(training_data);
        }
        
        let file = File::open("training_data.csv").map_err(|e| {
            opencv::Error::new(opencv::core::StsError, format!("Failed to open training_data.csv: {}", e))
        })?;
        
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        
        // Skip header
        if let Some(Ok(_header)) = lines.next() {
            for line in lines {
                let line = line.map_err(|e| {
                    opencv::Error::new(opencv::core::StsError, format!("Failed to read line: {}", e))
                })?;
                
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 9 {
                    if let (Ok(width), Ok(height), Ok(area), Ok(aspect_ratio), Ok(x), Ok(y), Ok(confidence), Ok(label)) = (
                        parts[0].parse::<f32>(),
                        parts[1].parse::<f32>(),
                        parts[2].parse::<f32>(),
                        parts[3].parse::<f32>(),
                        parts[4].parse::<f32>(),
                        parts[5].parse::<f32>(),
                        parts[6].parse::<f32>(),
                        parts[7].parse::<i32>(),
                    ) {
                        training_data.push(TrainingData {
                            width,
                            height,
                            area,
                            aspect_ratio,
                            x,
                            y,
                            confidence,
                            label,
                            type_name: parts[8].to_string(),
                        });
                    }
                }
            }
        }
        
        println!("Loaded {} existing training samples", training_data.len());
        Ok(training_data)
    }

    /// Train SVM model with the training data
    fn train_svm_model(&self, training_data: &[TrainingData]) -> opencv::Result<opencv::core::Ptr<ml::SVM>> {
        println!("Training SVM model...");
        
        // Prepare feature matrix and labels
        let feature_count = 7; // width, height, area, aspect_ratio, x, y, confidence
        let sample_count = training_data.len();
        
        let mut features = Mat::zeros(sample_count as i32, feature_count, core::CV_32F)?.to_mat()?;
        let mut labels = Mat::zeros(sample_count as i32, 1, core::CV_32S)?.to_mat()?;
        
        for (i, data) in training_data.iter().enumerate() {
            let feature_row = [data.width, data.height, data.area, data.aspect_ratio, data.x, data.y, data.confidence];
            for (j, &value) in feature_row.iter().enumerate() {
                *features.at_2d_mut::<f32>(i as i32, j as i32)? = value;
            }
            *labels.at_2d_mut::<i32>(i as i32, 0)? = data.label;
        }
        
        // Create and configure SVM
        let mut svm = ml::SVM::create()?;
        svm.set_type(ml::SVM_Types::C_SVC as i32)?;
        svm.set_kernel(ml::SVM_KernelTypes::RBF as i32)?;
        svm.set_c(1.0)?;
        svm.set_gamma(0.5)?;
        
        // Train the model
        svm.train(&features, ml::ROW_SAMPLE, &labels)?;
        
        println!("SVM model training completed!");
        Ok(svm)
    }

    /// Save training data to CSV file
    fn save_training_data_to_csv(&self, training_data: &[TrainingData]) -> opencv::Result<()> {
        let mut file = File::create("training_data.csv").map_err(|e| {
            opencv::Error::new(opencv::core::StsError, format!("Failed to create training_data.csv: {}", e))
        })?;
        
        // Write header
        writeln!(file, "Width,Height,Area,AspectRatio,X,Y,Confidence,Label,Type").map_err(|e| {
            opencv::Error::new(opencv::core::StsError, format!("Failed to write header: {}", e))
        })?;
        
        // Write data
        for data in training_data {
            writeln!(file, "{},{},{},{},{},{},{},{},{}", 
                     data.width, data.height, data.area, data.aspect_ratio, 
                     data.x, data.y, data.confidence, data.label, data.type_name).map_err(|e| {
                opencv::Error::new(opencv::core::StsError, format!("Failed to write data: {}", e))
            })?;
        }
        
        println!("Training data saved to training_data.csv");
        Ok(())
    }
}