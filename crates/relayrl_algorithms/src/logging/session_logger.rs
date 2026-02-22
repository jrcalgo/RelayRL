#[derive(Default, Clone)]
pub struct SessionLogger;

impl SessionLogger {
    pub fn new() -> Self {
        Self
    }

    pub fn log_session<T>(&self, _algorithm: &T) -> Result<(), std::io::Error> {
        Ok(())
    }
}
