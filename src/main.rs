mod detection;
mod training;
mod models;
mod utils;

use opencv::prelude::*;
use opencv::imgcodecs;
use std::fs::read_dir;
use std::path::Path;
use std::env;

use detection::WindowDoorDetector;
use training::ModelTrainer;
use models::ProcessingResult;

fn main() -> opencv::Result<()> {
    println!("OpenCV version: {}", opencv::core::get_version_string()?);
    
    // Parse command line arguments
    let args: Vec<String> = env::args().collect();
    let show_windows = args.contains(&"--show".to_string()) || args.contains(&"-s".to_string());
    let retrain_mode = args.contains(&"--retrain".to_string()) || args.contains(&"-r".to_string());
    
    if retrain_mode {
        println!("RETRAINING MODE - Interactive classification and model retraining");
        let mut trainer = ModelTrainer::new();
        return trainer.retrain_model_interactive();
    } else if show_windows {
        println!("Window display mode enabled - results will be shown in windows");
    } else {
        println!("Use --show or -s to display results in windows");
        println!("Use --retrain or -r to retrain the model interactively");
    }
    
    // Initialize detector
    let detector = WindowDoorDetector::new()?;
    
    // Process all images in the images folder
    let images_dir = "images";
    if !Path::new(images_dir).exists() {
        println!("Error: '{}' directory not found", images_dir);
        return Ok(());
    }
    
    // Supported image formats
    let supported_extensions = vec!["jpg", "jpeg", "png", "webp", "avif"];
    
    // Read all files in the images directory
    let entries = read_dir(images_dir).map_err(|e| {
        opencv::Error::new(opencv::core::StsError, format!("Failed to read directory: {}", e))
    })?;
    
    let mut processed_count = 0;
    let mut all_results = Vec::new();
    
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
                println!("Processing image: {}", image_path);
                println!("============================================================");
                
                let img = imgcodecs::imread(&image_path, imgcodecs::IMREAD_COLOR)?;
                
                if img.empty() {
                    println!("Warning: Could not load image from {}", image_path);
                    continue;
                }
                
                println!("Image loaded successfully! Size: {}x{}", img.cols(), img.rows());
                
                // Process the image
                let result = detector.process_image(&img, &image_path, processed_count, show_windows)?;
                all_results.push(result);
                processed_count += 1;
            }
        }
    }
    
    // Print summary
    print_processing_summary(processed_count, &all_results, &supported_extensions);
    
    Ok(())
}

fn print_processing_summary(processed_count: usize, all_results: &[ProcessingResult], supported_extensions: &[&str]) {
    println!("\n============================================================");
    println!("PROCESSING SUMMARY");
    println!("============================================================");
    println!("Total images processed: {}", processed_count);
    
    let total_windows: usize = all_results.iter().map(|r| r.windows).sum();
    let total_doors: usize = all_results.iter().map(|r| r.doors).sum();
    let total_buildings: usize = all_results.iter().map(|r| r.buildings).sum();
    
    println!("Total detections across all images:");
    println!("  Windows: {}", total_windows);
    println!("  Doors: {}", total_doors);
    println!("  Buildings: {}", total_buildings);
    
    // Show individual results
    for (i, result) in all_results.iter().enumerate() {
        println!("  Image {}: {} - W:{} D:{} B:{}", 
                 i + 1, result.filename, result.windows, result.doors, result.buildings);
    }
    
    if processed_count == 0 {
        println!("No supported image files found in 'images' directory");
        println!("Supported formats: {}", supported_extensions.join(", "));
    }
}