use opencv::prelude::*;
use opencv::{core, imgproc};

/// Scale image for display in a window (max dimension 1000px)
pub fn scale_for_display(image: &Mat) -> opencv::Result<Mat> {
    let max_dimension = 1000.0;
    let height = image.rows() as f64;
    let width = image.cols() as f64;
    
    let scale = if width > height {
        if width > max_dimension { max_dimension / width } else { 1.0 }
    } else {
        if height > max_dimension { max_dimension / height } else { 1.0 }
    };
    
    if scale < 1.0 {
        let new_width = (width * scale) as i32;
        let new_height = (height * scale) as i32;
        let mut scaled = Mat::default();
        imgproc::resize(image, &mut scaled, core::Size::new(new_width, new_height), 0.0, 0.0, imgproc::INTER_AREA)?;
        println!("Scaled image for display: {}x{} -> {}x{} (scale: {:.2})", 
                 width as i32, height as i32, new_width, new_height, scale);
        Ok(scaled)
    } else {
        Ok(image.clone())
    }
}

/// Simplified building detection function
pub fn detect_buildings(image: &Mat) -> opencv::Result<Mat> {
    let height = image.rows();
    let width = image.cols();
    
    // Simple building mask: everything except top 30% (sky) and bottom 20% (ground)
    let mut building_mask = Mat::zeros(height, width, core::CV_8UC1)?.to_mat()?;
    let sky_end = (height as f32 * 0.3) as i32;
    let ground_start = (height as f32 * 0.8) as i32;
    
    for y in sky_end..ground_start {
        for x in 0..width {
            *building_mask.at_2d_mut::<u8>(y, x)? = 255;
        }
    }
    
    Ok(building_mask)
}

/// Get supported image file extensions
pub fn get_supported_extensions() -> Vec<&'static str> {
    vec!["jpg", "jpeg", "png", "webp", "avif"]
}

/// Check if a file has a supported image extension
pub fn is_supported_image(extension: &str) -> bool {
    let ext_lower = extension.to_lowercase();
    get_supported_extensions().contains(&ext_lower.as_str())
}