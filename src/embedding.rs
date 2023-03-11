use crate::tensor::Tensor;
use crate::unpickler;
use crate::unpickler::*;
use std::collections::BTreeMap;
use std::path::Path;

pub struct Embedding {
    wgts: BTreeMap<usize, Tensor>,
}

impl Embedding {
    pub fn from_unpickled<P: AsRef<Path>>(
        unpickled: &unpickler::Value,
        data_dir: P,
    ) -> Result<Self, UnpicklingError> {
        let data_dir: &Path = data_dir.as_ref();

        let val = match unpickled.get_str_key("tok_embeddings.weight") {
            Some(val) => val,
            None => {
                return Err(UnpicklingError::MissingField(
                    "tok_embeddings.weight".to_string(),
                ))
            }
        };
        let tensor = val
            .to_tensor_builder()
            .ok_or(UnpicklingError::InvalidTensorData)?;
        let tensor = tensor.load(data_dir)?;

        let num_embeddings = tensor.rows();

        let mut table: BTreeMap<usize, Tensor> = BTreeMap::new();
        for key in 0..num_embeddings {
            let row = tensor.row(key);
            table.insert(key as usize, row);
        }

        Ok(Self { wgts: table })
    }

    pub fn get_embedding(&self, idx: usize) -> &Tensor {
        self.wgts.get(&idx).unwrap()
    }
}
