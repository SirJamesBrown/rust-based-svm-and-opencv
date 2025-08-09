use opencv::prelude::*;
use opencv::{core, imgcodecs, imgproc, ml, highgui};
use std::path::Path;

use crate::models::{ProcessingResult, DetectionCandidate};
use crate::utils::{scale_for_display, detect_buildings};

pub struct WindowDoorDetector {
    svm_model: Option<opencv::core::Ptr<ml::SVM>>,
}

impl WindowDoorDetector {
    pub fn new() -> opencv::Result<Self> {
        // Load the trained SVM model if it exists
        let svm_model = if Path::new("svm_window_door_model.xml").exists() {
            match ml::SVM::load("svm_window_door_model.xml") {
                Ok(model) => {
                    println!("Loaded existing SVM model for predictions");
                    Some(model)
                }
                Err(e) => {
                    println!("Warning: Could not load SVM model: {}", e);
                    None
                }
            }
        } else {
            None
        };

        Ok(WindowDoorDetector { svm_model })
    }

    pub fn process_image(
        &self,
        image: &Mat, 
        image_path: &str, 
        image_index: usize,
        show_windows: bool
    ) -> opencv::Result<ProcessingResult> {
        let filename = Path::new(image_path).file_name()
            .unwrap_or_default().to_string_lossy().to_string();
        
        // Convert to grayscale for processing
        let mut gray = Mat::default();
        imgproc::cvt_color(image, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;
        
        // Create building detection pipeline (simplified version)
        let building_mask = detect_buildings(image)?;
        
        // Detect features using edge detection
        let mut edges = Mat::default();
        imgproc::canny(&gray, &mut edges, 50.0, 150.0, 3, false)?;
        
        // Mask edges to only building regions
        let mut masked_edges = Mat::default();
        core::bitwise_and(&edges, &edges, &mut masked_edges, &building_mask)?;
        
        // Find contours for potential windows and doors
        let mut feature_contours = opencv::types::VectorOfVectorOfPoint::new();
        imgproc::find_contours(&masked_edges, &mut feature_contours, imgproc::RETR_EXTERNAL, 
                              imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
        
        // Analyze each contour and classify using SVM model if available
        let candidates = self.extract_candidates(&feature_contours)?;
        
        // Sort by confidence and separate into windows and doors
        let mut sorted_candidates = candidates;
        sorted_candidates.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        
        let mut result_image = image.clone();
        let (window_count, door_count) = self.draw_detections(&mut result_image, &sorted_candidates)?;
        
        // Detect building structures
        let building_count = self.detect_building_structures(&mut result_image, &building_mask)?;
        
        // Save the processed result
        let output_filename = format!("result_{:03}_{}", image_index + 1, filename);
        imgcodecs::imwrite(&output_filename, &result_image, &opencv::types::VectorOfi32::new())?;
        println!("Processed image saved as: {}", output_filename);
        
        // Show the result in a window if the flag is set
        if show_windows {
            self.show_result_window(&result_image, &filename)?;
        }
        
        Ok(ProcessingResult {
            filename,
            windows: window_count,
            doors: door_count,
            buildings: building_count,
        })
    }

    fn extract_candidates(&self, feature_contours: &opencv::types::VectorOfVectorOfPoint) -> opencv::Result<Vec<DetectionCandidate>> {
        let mut candidates = Vec::new();
        
        for i in 0..feature_contours.len() {
            let contour = feature_contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            let rect = imgproc::bounding_rect(&contour)?;
            
            // Filter by size
            if area > 200.0 && area < 20000.0 && rect.width > 15 && rect.height > 15 {
                let aspect_ratio = rect.width as f64 / rect.height as f64;
                
                // Create feature vector for SVM prediction
                let features = vec![
                    rect.width as f32,
                    rect.height as f32,
                    (rect.width * rect.height) as f32,
                    aspect_ratio as f32,
                    rect.x as f32,
                    rect.y as f32,
                    0.8, // Default confidence for new detections
                ];
                
                // Use SVM model for prediction if available
                let (predicted_class, confidence) = if let Some(ref model) = self.svm_model {
                    match self.predict_with_svm(model, &features) {
                        Ok((class, conf)) => (class, conf),
                        Err(_) => {
                            // Fallback to heuristic classification
                            self.heuristic_classification(rect, aspect_ratio, area)
                        }
                    }
                } else {
                    // Use heuristic classification
                    self.heuristic_classification(rect, aspect_ratio, area)
                };
                
                candidates.push(DetectionCandidate {
                    class: predicted_class,
                    confidence,
                    rect,
                    aspect_ratio,
                });
            }
        }
        
        Ok(candidates)
    }

    fn predict_with_svm(&self, model: &opencv::core::Ptr<ml::SVM>, features: &[f32]) -> opencv::Result<(i32, f32)> {
        let base_mat = Mat::from_slice(features)?;
        let feature_mat = base_mat.reshape(1, 1)?;
        let mut result = Mat::default();
        
        let prediction = model.predict(&feature_mat, &mut result, 0)?;
        
        // Convert prediction to confidence (simplified)
        let confidence = if prediction == 0.0 || prediction == 1.0 { 0.9 } else { 0.6 };
        
        Ok((prediction as i32, confidence))
    }

    fn heuristic_classification(&self, rect: core::Rect, aspect_ratio: f64, area: f64) -> (i32, f32) {
        // Door characteristics: tall and narrow
        let door_score = if aspect_ratio >= 0.2 && aspect_ratio <= 0.8 && rect.height > 80 && area > 2000.0 {
            0.8
        } else {
            0.2
        };
        
        // Window characteristics: more square/rectangular
        let window_score = if aspect_ratio >= 0.7 && aspect_ratio <= 2.0 && 
                              rect.width > 25 && rect.width < 150 && 
                              rect.height > 25 && rect.height < 100 && 
                              area > 500.0 && area < 8000.0 {
            0.8
        } else {
            0.3
        };
        
        if door_score > window_score {
            (1, door_score) // Door
        } else {
            (0, window_score) // Window
        }
    }

    fn draw_detections(&self, result_image: &mut Mat, candidates: &[DetectionCandidate]) -> opencv::Result<(usize, usize)> {
        let mut window_count = 0;
        let mut door_count = 0;

        for candidate in candidates.iter().take(20) {
            let color = if candidate.confidence > 0.7 {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0) // Green for high confidence
            } else if candidate.confidence > 0.5 {
                core::Scalar::new(0.0, 165.0, 255.0, 0.0) // Orange for medium confidence  
            } else {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0) // Red for low confidence
            };
            
            imgproc::rectangle(result_image, candidate.rect, color, 2, imgproc::LINE_8, 0)?;
            
            let label = if candidate.class == 0 {
                window_count += 1;
                format!("W{} ({:.0}%)", window_count, candidate.confidence * 100.0)
            } else {
                door_count += 1;
                format!("D{} ({:.0}%)", door_count, candidate.confidence * 100.0)
            };
            
            imgproc::put_text(result_image, &label,
                core::Point::new(candidate.rect.x, candidate.rect.y - 5),
                imgproc::FONT_HERSHEY_SIMPLEX, 0.4,
                color, 1, imgproc::LINE_8, false)?;
                
            println!("{}: {}x{} at ({}, {}), ratio: {:.2}, confidence: {:.1}%", 
                     label, candidate.rect.width, candidate.rect.height, 
                     candidate.rect.x, candidate.rect.y, candidate.aspect_ratio, candidate.confidence * 100.0);
        }

        Ok((window_count, door_count))
    }

    fn detect_building_structures(&self, result_image: &mut Mat, building_mask: &Mat) -> opencv::Result<usize> {
        let mut building_contours = opencv::types::VectorOfVectorOfPoint::new();
        imgproc::find_contours(building_mask, &mut building_contours, imgproc::RETR_EXTERNAL, 
                              imgproc::CHAIN_APPROX_SIMPLE, core::Point::new(0, 0))?;
        
        let image_area = (building_mask.rows() * building_mask.cols()) as f64;
        let mut building_count = 0;
        
        for i in 0..building_contours.len() {
            let contour = building_contours.get(i)?;
            let area = imgproc::contour_area(&contour, false)?;
            
            // Only consider large regions as buildings
            if area > image_area * 0.02 {
                building_count += 1;
                let rect = imgproc::bounding_rect(&contour)?;
                
                // Draw purple bounding box for buildings
                imgproc::rectangle(result_image, rect, 
                    core::Scalar::new(128.0, 0.0, 128.0, 0.0), 4, imgproc::LINE_8, 0)?;
                
                // Add building label
                let text_bg = core::Rect::new(rect.x + 5, rect.y + 5, 40, 20);
                imgproc::rectangle(result_image, text_bg,
                    core::Scalar::new(255.0, 255.0, 255.0, 0.0), -1, imgproc::LINE_8, 0)?;
                
                imgproc::put_text(result_image, &format!("B{}", building_count),
                    core::Point::new(rect.x + 8, rect.y + 18),
                    imgproc::FONT_HERSHEY_SIMPLEX, 0.5,
                    core::Scalar::new(128.0, 0.0, 128.0, 0.0), 1, imgproc::LINE_8, false)?;
                
                println!("Building {}: Area: {:.1} ({:.2}% of image), Bounds: {}x{} at ({}, {})", 
                         building_count, area, (area/image_area)*100.0, rect.width, rect.height, rect.x, rect.y);
            }
        }
        
        Ok(building_count)
    }

    fn show_result_window(&self, result_image: &Mat, filename: &str) -> opencv::Result<()> {
        // Scale the image for display (max 1000px)
        let scaled_image = scale_for_display(result_image)?;
        
        let window_title = format!("Detection Result - {}", filename);
        highgui::named_window(&window_title, highgui::WINDOW_AUTOSIZE)?;
        highgui::imshow(&window_title, &scaled_image)?;
        println!("Press any key to continue to next image...");
        highgui::wait_key(0)?;
        highgui::destroy_window(&window_title)?;
        
        Ok(())
    }
}