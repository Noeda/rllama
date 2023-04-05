/*
 * Understands HuggingFace format for models, or well at least as much as we need to.
 */

use crate::unpickler;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum HugginfaceModelError {
    #[error("Error parsing JSON: {0}")]
    JSONError(#[from] serde_json::Error),
    #[error("IO error: {0}")]
    IOError(#[from] std::io::Error),
    #[error("ZIP error: {0}")]
    ZIPError(#[from] zip::result::ZipError),
    #[error("Unpickler error: {0}")]
    UnpicklingError(#[from] unpickler::UnpicklingError),
}

pub struct HugginfaceModel {
    pub(crate) unpickles: Vec<(unpickler::Value, PathBuf)>,
    // (path, files, tensors)
    pub(crate) zip_file_contents: Vec<(PathBuf, BTreeSet<String>, BTreeSet<String>)>,
    pub(crate) unpickles_flattened: unpickler::Value,
    pub(crate) index: HugginfaceIndex,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceConfig {
    vocab_size: usize,
    hidden_size: usize,
    intermediate_size: usize,
    num_hidden_layers: usize,
    num_attention_heads: usize,
    max_position_embeddings: usize,
    rms_norm_eps: f32,
    architectures: Vec<String>,

    bos_token_id: usize,
    eos_token_id: usize,

    torch_dtype: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceIndex {
    metadata: HugginfaceIndexMetadata,
    weight_map: BTreeMap<String, String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HugginfaceIndexMetadata {
    total_size: usize,
}

impl HugginfaceModel {
    pub fn unpickle<P: AsRef<Path>>(path: P) -> Result<Self, HugginfaceModelError> {
        let path: &Path = path.as_ref();

        let mut unpickles = vec![];

        // Read config,json
        let config_json_path: PathBuf = path.join("config.json");
        let config_json = std::fs::read_to_string(config_json_path)?;
        let config: HugginfaceConfig = serde_json::from_str(&config_json)?;

        let index_json_path: PathBuf = path.join("pytorch_model.bin.index.json");
        let index_json = std::fs::read_to_string(index_json_path)?;
        let index: HugginfaceIndex = serde_json::from_str(&index_json)?;

        // List all .bin files that contain the weights.
        let mut weight_files: Vec<PathBuf> = vec![];
        for entry in std::fs::read_dir(path)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().unwrap_or_default() == "bin" {
                weight_files.push(path);
            }
        }

        // List all files in said zips
        let mut unpickles2 = vec![];
        let mut zip_file_contents = vec![];
        for file in weight_files.iter() {
            let mut files_in_zip = BTreeSet::new();
            let mut tensors_in_zip = BTreeSet::new();
            let reader = std::io::BufReader::new(std::fs::File::open(file)?);
            let mut archive = zip::ZipArchive::new(reader)?;
            for i in 0..archive.len() {
                let mut file = archive.by_index(i)?;
                // Remove the first directory.
                let file2 = remove_first_directory(file.name());
                files_in_zip.insert(file2.to_str().unwrap().to_string());
                // data.pkl
                if file.name().ends_with("data.pkl") {
                    let mut data_unzipped: Vec<u8> = vec![];
                    file.read_to_end(&mut data_unzipped)?;
                    let unpickled = unpickler::unpickle(&data_unzipped)?;
                    for tensor in unpickled.keys() {
                        tensors_in_zip.insert(tensor.to_string());
                    }
                    unpickles2.push(unpickled.clone());
                    unpickles.push((unpickled, file.name().to_string().into()))
                }
            }
            zip_file_contents.push((file.clone(), files_in_zip, tensors_in_zip));
        }
        // Flatten unpickles.
        let unpickles_flattened = crate::unpickler::Value::merge_dicts(&unpickles2);

        Ok(HugginfaceModel {
            unpickles,
            unpickles_flattened,
            zip_file_contents,
            index,
        })
    }
}

pub fn remove_first_directory<P: AsRef<Path>>(path: P) -> PathBuf {
    let path = path.as_ref();
    let mut components = vec![];
    for component in path.components().skip(1) {
        components.push(component);
    }
    PathBuf::from(components.iter().collect::<PathBuf>())
}
