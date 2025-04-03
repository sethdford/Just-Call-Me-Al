use anyhow::Result;
use tch::{Device, Kind, Tensor};

// Core tensor utilities
pub fn tensor_to_vec_f32(tensor: &Tensor) -> Result<Vec<f32>> {
    let contiguous = tensor.to_kind(Kind::Float).contiguous();
    Ok(Vec::<f32>::try_from(&contiguous)?)
}

pub fn tensor_to_vec_i64(tensor: &Tensor) -> Result<Vec<i64>> {
    let contiguous = tensor.to_kind(Kind::Int64).contiguous();
    Ok(Vec::<i64>::try_from(&contiguous)?)
}

pub fn create_mask(lengths: &[i64], max_length: i64, device: Device) -> Result<Tensor> {
    let batch_size = lengths.len() as i64;
    
    // Create position indices
    let indices = Tensor::arange(max_length, (Kind::Int64, device));
    let indices = indices.unsqueeze(0).expand(&[batch_size, max_length], false);
    
    // Create expanded lengths
    let lengths_tensor = Tensor::from_slice(lengths).to_device(device);
    let lengths_expanded = lengths_tensor.unsqueeze(1).expand(&[batch_size, max_length], false);
    
    // Create mask where indices < lengths
    Ok(indices.lt_tensor(&lengths_expanded))
}

pub fn create_triu_mask(seq_len: i64, device: Device) -> Result<Tensor> {
    let mask = Tensor::ones(&[seq_len, seq_len], (Kind::Bool, device));
    let triu = mask.triu(1);
    Ok(triu.logical_not())
}

pub fn create_attention_mask(
    _batch_size: i64,
    seq_len: i64,
    lengths: &[i64],
    device: Device,
) -> Result<Tensor> {
    // 1. Create causal mask [seq_len, seq_len]
    let causal_mask = create_triu_mask(seq_len, device)?;
    // Expand to [1, seq_len, seq_len] for broadcasting
    let causal_mask_expanded = causal_mask.unsqueeze(0);
    
    // 2. Create padding mask [batch_size, seq_len]
    let padding_mask = create_mask(lengths, seq_len, device)?;
    // Expand for columns [batch_size, 1, seq_len]
    let padding_mask_cols = padding_mask.unsqueeze(1);
    // Expand for rows [batch_size, seq_len, 1]
    let padding_mask_rows = padding_mask.unsqueeze(2);
    
    // 3. Combine masks using logical_and and broadcasting
    // Causal     [1, T, S]
    // PaddingCols[B, 1, S]
    // PaddingRows[B, T, 1]
    // Result = Causal & PaddingCols & PaddingRows -> [B, T, S]
    Ok(causal_mask_expanded
        .logical_and(&padding_mask_cols)
        .logical_and(&padding_mask_rows))
}

pub fn pad_sequence(sequences: &[Tensor], padding_value: f64) -> Result<Tensor> {
    if sequences.is_empty() {
        return Ok(Tensor::empty(&[0], (Kind::Float, Device::Cpu)));
    }

    let batch_size = sequences.len() as i64;
    let max_len = sequences.iter()
        .map(|seq| seq.size()[0])
        .max()
        .unwrap_or(0);
    
    let device = sequences[0].device();
    let padded = Tensor::full(&[batch_size, max_len], padding_value, (Kind::Float, device));
    
    for (i, seq) in sequences.iter().enumerate() {
        let seq_len = seq.size()[0];
        padded.slice(0, i as i64, (i + 1) as i64, 1)
             .slice(1, 0, seq_len, 1)
             .copy_(&seq.slice(0, 0, seq_len, 1));
    }
    
    Ok(padded)
}

pub trait TensorExt {
    fn to_vec_f32(&self) -> Result<Vec<f32>>;
    fn to_vec_i64(&self) -> Result<Vec<i64>>;
}

impl TensorExt for Tensor {
    fn to_vec_f32(&self) -> Result<Vec<f32>> {
        tensor_to_vec_f32(self)
    }

    fn to_vec_i64(&self) -> Result<Vec<i64>> {
        tensor_to_vec_i64(self)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_to_vec_f32() -> Result<()> {
        let tensor = Tensor::from_slice(&[1.0f32, 2.0, 3.0]);
        let vec = tensor_to_vec_f32(&tensor)?;
        assert_eq!(vec, vec![1.0, 2.0, 3.0]);
        Ok(())
    }

    #[test]
    fn test_tensor_to_vec_i64() -> Result<()> {
        let tensor = Tensor::from_slice(&[1i64, 2, 3]);
        let vec = tensor_to_vec_i64(&tensor)?;
        assert_eq!(vec, vec![1, 2, 3]);
        Ok(())
    }

    #[test]
    fn test_create_mask() -> Result<()> {
        let lengths = vec![2i64, 3];
        let max_len = 4;
        let mask = create_mask(&lengths, max_len, Device::Cpu)?;
        
        let expected = Tensor::from_slice(&[
            true, true, false, false,
            true, true, true, false,
        ]).reshape(&[2, 4]);
        assert!(mask.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_create_triu_mask() -> Result<()> {
        let mask = create_triu_mask(3, Device::Cpu)?;
        let expected = Tensor::from_slice(&[
            true, false, false,
            true, true, false,
            true, true, true,
        ]).reshape(&[3, 3]);
        assert!(mask.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_create_attention_mask() -> Result<()> {
        let lengths = vec![2i64, 3];
        let batch_size = 2;
        let seq_len = 3;
        let mask = create_attention_mask(batch_size, seq_len, &lengths, Device::Cpu)?;
        
        let expected = Tensor::from_slice(&[
            true, false, false,
            true, true, false,
            false, false, false,
            true, false, false,
            true, true, false,
            true, true, true,
        ]).reshape(&[2, 3, 3]);
        assert!(mask.equal(&expected));
        Ok(())
    }

    #[test]
    fn test_pad_sequence() -> Result<()> {
        let seq1 = Tensor::from_slice(&[1.0f32, 2.0]);
        let seq2 = Tensor::from_slice(&[3.0f32]);
        let padded = pad_sequence(&[seq1, seq2], 0.0)?;
        
        let expected = Tensor::from_slice(&[
            1.0, 2.0,
            3.0, 0.0,
        ]).reshape(&[2, 2]);
        assert!(padded.equal(&expected));
        Ok(())
    }
} 