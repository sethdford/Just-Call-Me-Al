use safetensors::SafeTensorError;
use tch::TchError;

#[derive(Debug, thiserror::Error)]
pub enum TensorLoadError {
    #[error("Tch Error: {0}")]
    Tch(#[from] TchError),

    #[error("SafeTensors Error: {0}")]
    SafeTensors(#[from] SafeTensorError),

    #[error("IO Error: {0}")]
    Io(#[from] std::io::Error),

    #[error("VarStore variable not found: {0}")]
    VarNotFound(String),

    #[error("Shape mismatch for {0}: VarStore={1:?}, File={2:?}")]
    ShapeMismatch(String, Vec<i64>, Vec<usize>),
} 