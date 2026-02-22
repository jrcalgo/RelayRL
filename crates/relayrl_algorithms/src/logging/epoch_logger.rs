#[derive(Default, Clone)]
pub struct EpochLogger;

impl EpochLogger {
    pub fn new() -> Self {
        Self
    }

    pub fn store(&mut self, _key: &str, _value: f32) {}

    pub fn log_tabular(&mut self, _key: &str, _value: Option<f32>) {}

    pub fn dump_tabular(&mut self) {}
}
