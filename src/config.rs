#[derive(Debug, Clone)]
pub struct FileProcessingConfig {
    pub file_write_timeout: u64,
    pub file_check_interval: u64,
    pub ignore_existing: bool,
    pub ollama_hosts: Vec<String>,
    pub unavailable_duration: u64,
    pub request_timeout: u64,
}

#[derive(Debug, Clone)]
pub struct MonitorConfig {
    pub file_write_timeout: u64,
    pub file_check_interval: u64,
    pub event_cooldown: u64,
    pub timeout: u64,
    pub lang: String,
    pub ignore_existing: bool,
    pub ollama_hosts: Vec<String>,
    pub unavailable_duration: u64,
}
