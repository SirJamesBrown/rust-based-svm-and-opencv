#[derive(Debug)]
pub struct ProcessingResult {
    pub filename: String,
    pub windows: usize,
    pub doors: usize,
    pub buildings: usize,
}

#[derive(Debug, Clone)]
pub struct TrainingData {
    pub width: f32,
    pub height: f32,
    pub area: f32,
    pub aspect_ratio: f32,
    pub x: f32,
    pub y: f32,
    pub confidence: f32,
    pub label: i32, // 0 for window, 1 for door
    pub type_name: String,
}

#[derive(Debug)]
pub struct DetectionCandidate {
    pub class: i32,
    pub confidence: f32,
    pub rect: opencv::core::Rect,
    pub aspect_ratio: f64,
}