use std::time::{Duration, Instant};

pub struct SessionLogger {
    start_time: Instant,
}

impl Default for SessionLogger {
    fn default() -> Self {
        Self::new()
    }
}

impl SessionLogger {
    pub fn new() -> Self {
        Self {
            start_time: Instant::now(),
        }
    }

    pub fn log_session<T>(&self, _algorithm: &T) -> Result<(), std::io::Error> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        println!(
            "=== RelayRL Training Session ===\nStarted at UNIX time: {now}\nAlgorithm: {}\n================================",
            std::any::type_name::<T>()
        );
        Ok(())
    }

    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    pub fn log_elapsed(&self) {
        let elapsed = self.elapsed();
        println!("Session elapsed: {:.2}s", elapsed.as_secs_f64());
    }
}
