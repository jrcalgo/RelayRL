use std::collections::HashMap;

pub struct EpochLogger {
    stored: HashMap<String, Vec<f32>>,
    fixed: HashMap<String, f32>,
    columns: Vec<String>,
    header_printed: bool,
}

impl Default for EpochLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl EpochLogger {
    pub fn new() -> Self {
        Self {
            stored: HashMap::new(),
            fixed: HashMap::new(),
            columns: Vec::new(),
            header_printed: false,
        }
    }

    pub fn store(&mut self, key: &str, value: f32) {
        self.stored.entry(key.to_string()).or_default().push(value);
    }

    pub fn log_tabular(&mut self, key: &str, value: Option<f32>) {
        if !self.columns.contains(&key.to_string()) {
            self.columns.push(key.to_string());
        }
        if let Some(v) = value {
            self.fixed.insert(key.to_string(), v);
        }
    }

    pub fn dump_tabular(&mut self) {
        if self.columns.is_empty() {
            return;
        }

        // Build rows
        let mut rows: Vec<(String, String)> = Vec::new();
        for col in &self.columns {
            let val_str = if let Some(&v) = self.fixed.get(col) {
                format!("{:.4}", v)
            } else if let Some(vals) = self.stored.get(col) {
                if vals.is_empty() {
                    "-".to_string()
                } else if vals.len() == 1 {
                    format!("{:.4}", vals[0])
                } else {
                    let mean = vals.iter().sum::<f32>() / vals.len() as f32;
                    let min = vals.iter().cloned().fold(f32::INFINITY, f32::min);
                    let max = vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    format!("{:.4} (min {:.4}, max {:.4})", mean, min, max)
                }
            } else {
                "-".to_string()
            };
            rows.push((col.clone(), val_str));
        }

        // Determine column widths
        let key_width = rows
            .iter()
            .map(|(k, _)| k.len())
            .max()
            .unwrap_or(10)
            .max(10);
        let val_width = rows
            .iter()
            .map(|(_, v)| v.len())
            .max()
            .unwrap_or(10)
            .max(10);
        let total_width = key_width + val_width + 5;

        let sep = "-".repeat(total_width);
        println!("{sep}");
        for (key, val) in &rows {
            println!("| {:<key_width$} | {:>val_width$} |", key, val);
        }
        println!("{sep}");

        // Clear state for next epoch
        self.stored.clear();
        self.fixed.clear();
        self.columns.clear();
        self.header_printed = false;
    }
}
