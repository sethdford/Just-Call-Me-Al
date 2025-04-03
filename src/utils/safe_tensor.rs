use anyhow::Result;
use tch::{Device, Kind, Tensor};
use std::ops::{Add, Sub, Mul, Div};
use super::tensor::{tensor_to_vec_f32, tensor_to_vec_i64};
use std::sync::Arc;

#[derive(Clone)]
pub struct SafeTensor {
    tensor: Arc<Tensor>,
}

unsafe impl Send for SafeTensor {}
unsafe impl Sync for SafeTensor {}

impl SafeTensor {
    pub fn new(tensor: Tensor) -> Self {
        Self {
            tensor: Arc::new(tensor),
        }
    }

    pub fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub fn into_tensor(self) -> Tensor {
        Arc::try_unwrap(self.tensor)
            .unwrap_or_else(|arc| (*arc).shallow_clone())
    }

    pub fn size(&self) -> Vec<i64> {
        self.tensor.size()
    }

    pub fn reshape(&self, shape: &[i64]) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.reshape(shape)))
    }

    pub fn softmax(&self, dim: i64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.softmax(dim, Kind::Float)))
    }

    pub fn mean(&self) -> Result<f64> {
        Ok(self.tensor.mean(Kind::Float).double_value(&[]))
    }

    pub fn argmax(&self, dim: i64, keepdim: bool) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.argmax(dim, keepdim)))
    }

    pub fn to_vec_f32(&self) -> Result<Vec<f32>> {
        tensor_to_vec_f32(&self.tensor)
    }

    pub fn to_vec_i64(&self) -> Result<Vec<i64>> {
        tensor_to_vec_i64(&self.tensor)
    }

    pub fn map<F>(&self, f: F) -> Result<SafeTensor>
    where
        F: FnOnce(&Tensor) -> Result<Tensor>,
    {
        Ok(SafeTensor::new(f(&self.tensor)?))
    }

    pub fn dim(&self) -> i64 {
        self.tensor.dim() as i64
    }

    pub fn to_kind(&self, kind: Kind) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.to_kind(kind)))
    }

    pub fn to_device(&self, device: Device) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.to_device(device)))
    }

    pub fn clone(&self) -> Self {
        Self::new(self.tensor.shallow_clone())
    }

    pub fn device(&self) -> Device {
        self.tensor.device()
    }

    pub fn add(&self, other: &SafeTensor) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_add(&other.tensor)?))
    }

    pub fn sub(&self, other: &SafeTensor) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_sub(&other.tensor)?))
    }

    pub fn mul(&self, other: &SafeTensor) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_mul(&other.tensor)?))
    }

    pub fn div(&self, other: &SafeTensor) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_div(&other.tensor)?))
    }

    pub fn add_scalar(&self, scalar: f64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_add_scalar(scalar)?))
    }

    pub fn sub_scalar(&self, scalar: f64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_sub_scalar(scalar)?))
    }

    pub fn mul_scalar(&self, scalar: f64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_mul_scalar(scalar)?))
    }

    pub fn div_scalar(&self, scalar: f64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_div_scalar(scalar)?))
    }

    pub fn contiguous(&self) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.contiguous()))
    }

    pub fn squeeze(&self, dim: i64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.squeeze_dim(dim)))
    }

    pub fn transpose(&self, dim0: i64, dim1: i64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.transpose(dim0, dim1)))
    }

    pub fn permute(&self, dims: &[i64]) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.permute(dims)))
    }

    pub fn view(&self, shape: &[i64]) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.view(shape)))
    }

    pub fn chunk(&self, chunks: i64, dim: i64) -> Result<Vec<SafeTensor>> {
        Ok(self.tensor
            .chunk(chunks, dim)
            .into_iter()
            .map(SafeTensor::new)
            .collect())
    }

    pub fn split(&self, split_size: i64, dim: i64) -> Result<Vec<SafeTensor>> {
        Ok(self.tensor
            .split(split_size, dim)
            .into_iter()
            .map(SafeTensor::new)
            .collect())
    }

    pub fn cat(tensors: &[SafeTensor], dim: i64) -> Result<SafeTensor> {
        let tensor_refs: Vec<&Tensor> = tensors.iter().map(|t| &*t.tensor).collect();
        Ok(SafeTensor::new(Tensor::cat(&tensor_refs, dim)))
    }

    pub fn stack(tensors: &[SafeTensor], dim: i64) -> Result<SafeTensor> {
        let tensor_refs: Vec<Tensor> = tensors.iter()
            .map(|t| t.tensor.shallow_clone())
            .collect();
        Ok(SafeTensor::new(Tensor::stack(&tensor_refs, dim)))
    }

    pub fn matmul(&self, other: &SafeTensor) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.matmul(&other.tensor)))
    }

    pub fn pow(&self, exponent: f64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.pow_tensor_scalar(exponent)))
    }

    pub fn mean_dim(&self, dim: i64, keepdim: bool) -> Result<SafeTensor> {
        let dims = vec![dim];
        Ok(SafeTensor::new(self.tensor.mean_dim(&dims, keepdim, self.tensor.kind())))
    }

    pub fn sum_dim(&self, dim: i64, keepdim: bool) -> Result<SafeTensor> {
        let dims = vec![dim];
        Ok(SafeTensor::new(self.tensor.sum_dim_intlist(&dims, keepdim, self.tensor.kind())))
    }

    pub fn expand(&self, shape: &[i64]) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.expand(shape, false)))
    }

    pub fn unsqueeze(&self, dim: i64) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.unsqueeze(dim)))
    }

    pub fn f_contiguous(&self) -> Result<SafeTensor> {
        Ok(SafeTensor::new(self.tensor.f_contiguous()?))
    }

    pub fn f_to_vec<T: tch::kind::Element + From<f32> + From<i64>>(&self) -> Result<Vec<T>> {
        match T::KIND {
            Kind::Float => Ok(tensor_to_vec_f32(&self.tensor)?.into_iter().map(|x| T::from(x)).collect()),
            Kind::Int64 => Ok(tensor_to_vec_i64(&self.tensor)?.into_iter().map(|x| T::from(x)).collect()),
            _ => Err(anyhow::anyhow!("Unsupported tensor kind for conversion"))
        }
    }
}

impl Add for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn add(self, other: &SafeTensor) -> Self::Output {
        self.add(other)
    }
}

impl Sub for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn sub(self, other: &SafeTensor) -> Self::Output {
        self.sub(other)
    }
}

impl Mul for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn mul(self, other: &SafeTensor) -> Self::Output {
        self.mul(other)
    }
}

impl Div for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn div(self, other: &SafeTensor) -> Self::Output {
        self.div(other)
    }
}

impl Add<f64> for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn add(self, scalar: f64) -> Self::Output {
        Ok(SafeTensor::new(&*self.tensor * 1.0 + scalar))
    }
}

impl Sub<f64> for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn sub(self, scalar: f64) -> Self::Output {
        Ok(SafeTensor::new(&*self.tensor * 1.0 - scalar))
    }
}

impl Mul<f64> for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn mul(self, scalar: f64) -> Self::Output {
        Ok(SafeTensor::new(&*self.tensor * scalar))
    }
}

impl Div<f64> for &SafeTensor {
    type Output = Result<SafeTensor>;

    fn div(self, scalar: f64) -> Self::Output {
        Ok(SafeTensor::new(&*self.tensor / scalar))
    }
}

impl From<Tensor> for SafeTensor {
    fn from(tensor: Tensor) -> Self {
        Self::new(tensor)
    }
}

impl From<SafeTensor> for Tensor {
    fn from(safe_tensor: SafeTensor) -> Self {
        safe_tensor.into_tensor()
    }
}

impl std::fmt::Debug for SafeTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafeTensor")
            .field("size", &self.tensor.size())
            .field("device", &self.tensor.device())
            .field("kind", &self.tensor.kind())
            .finish()
    }
}

#[allow(dead_code)]
fn compute_distances(x: &Tensor, codebook: &Tensor) -> Result<Tensor> {
    let dims = vec![-1i64];
    let x_norm = x.pow_tensor_scalar(2.0).mean_dim(&dims, true, x.kind());
    let cb_norm = codebook.pow_tensor_scalar(2.0).mean_dim(&dims, true, x.kind());
    let prod = x.matmul(&codebook.transpose(-2, -1));
    
    Ok(x_norm + cb_norm.transpose(-2, -1) - 2.0 * prod)
}

#[allow(dead_code)]
fn normalize_tensor(x: &Tensor) -> Result<Tensor> {
    let dims = vec![1i64];
    let norm = x.mean_dim(&dims, true, x.kind());
    Ok(x / norm.clamp_min(1e-12))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safe_tensor_basics() -> Result<()> {
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let safe = SafeTensor::new(tensor);
        
        let vec = safe.to_vec_f32()?;
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        
        Ok(())
    }

    #[test]
    fn test_safe_tensor_ops() -> Result<()> {
        let t1 = SafeTensor::new(Tensor::from_slice(&[1.0f32, 2.0]));
        let t2 = SafeTensor::new(Tensor::from_slice(&[3.0f32, 4.0]));
        
        let sum = t1.add(&t2)?;
        let vec = sum.to_vec_f32()?;
        assert_eq!(vec, vec![4.0, 6.0]);
        
        Ok(())
    }

    #[test]
    fn test_safe_tensor_scalar_ops() -> Result<()> {
        let t = SafeTensor::new(Tensor::from_slice(&[1.0f32, 2.0]));
        
        let scaled = t.mul_scalar(2.0)?;
        let vec = scaled.to_vec_f32()?;
        assert_eq!(vec, vec![2.0, 4.0]);
        
        Ok(())
    }
} 