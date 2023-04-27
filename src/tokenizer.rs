use crate::protomodels::sentencepiece_model::model_proto::sentence_piece;
use crate::protomodels::sentencepiece_model::ModelProto;
use protobuf::Message;
use std::collections::BTreeMap;
use std::io::Read;
use std::path::Path;
use thiserror::Error;

pub type TokenId = i32;

#[derive(Clone, Debug)]
pub struct Tokenizer {
    pieces: BTreeMap<String, Piece>,
}

#[derive(Clone, Debug, Copy, Eq, Ord, PartialEq, PartialOrd)]
pub enum PieceType {
    Normal,
    Unknown,
    Control,
    UserDefined,
    Byte,
    Unused,
}

#[derive(Clone, Debug)]
pub struct Piece {
    _tp: PieceType,
    // piece: String   this is in the BTreeMap that holds the pieces
    _score: f32,
    idx: usize,
}

#[derive(Error, Debug)]
pub enum TokenizerError {
    #[error("IO error")]
    IoError(#[from] std::io::Error),
    #[error("Protobuf error")]
    ProtobufError(#[from] protobuf::Error),
    #[error("Unknown piece type")]
    UnknownPieceType(String),
}

impl Tokenizer {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Tokenizer, TokenizerError> {
        let mut fs = std::fs::File::open(path)?;
        let mut buffer = Vec::new();
        fs.read_to_end(&mut buffer)?;
        std::mem::drop(fs);
        let model = ModelProto::parse_from_bytes(&buffer)?;

        let mut pieces = BTreeMap::new();
        for (idx, piece) in model.pieces.iter().enumerate() {
            let piece_str = piece.piece.clone();
            if piece_str.is_none() {
                continue;
            }
            let piece_str = piece_str.unwrap();
            let piece_type = match piece.type_ {
                None => sentence_piece::Type::NORMAL,
                Some(v) => match v.enum_value() {
                    Err(_) => return Err(TokenizerError::UnknownPieceType(piece_str)),
                    Ok(v) => v,
                },
            };

            let score = piece.score.unwrap_or(0.0);
            let tp = if piece_type == sentence_piece::Type::NORMAL {
                PieceType::Normal
            } else if piece_type == sentence_piece::Type::UNKNOWN {
                PieceType::Unknown
            } else if piece_type == sentence_piece::Type::CONTROL {
                PieceType::Control
            } else if piece_type == sentence_piece::Type::USER_DEFINED {
                PieceType::UserDefined
            } else if piece_type == sentence_piece::Type::BYTE {
                PieceType::Byte
            } else if piece_type == sentence_piece::Type::UNUSED {
                PieceType::Unused
            } else {
                return Err(TokenizerError::UnknownPieceType(piece_str));
            };
            pieces.insert(
                piece_str,
                Piece {
                    _tp: tp,
                    _score: score,
                    idx,
                },
            );
        }

        Ok(Tokenizer { pieces })
    }

    // Gives a string for a token id.
    // Panics if the id is out of range.
    pub fn id_to_str(&self, id: i32) -> &str {
        let id = id as usize;
        for (piece_str, piece_info) in self.pieces.iter() {
            if piece_info.idx == id {
                return piece_str;
            }
        }
        panic!("id out of range");
    }

    // Tries to find a token from dictionary.
    pub fn str_to_id(&self, s: &str) -> Option<TokenId> {
        self.pieces.get(s).map(|piece_info| piece_info.idx as i32)
    }

    // Converts a string to a Vec<&str>
    // You may want to use tokenize_to_ids instead.
    //
    // This will not add start or end tokens; only the string is processed.
    //
    // I noticed LLaMa code adds an extra space character at the beginning of the string, this
    // function does not do that either.
    pub fn tokenize_to_pieces<S: AsRef<str>>(&self, s: S) -> Vec<&str> {
        let mut s: &str = s.as_ref();
        let mut result: Vec<&str> = Vec::new();

        // Very naive matching
        while !s.is_empty() {
            let mut best_candidate: &str = "";
            let mut best_candidate_len: usize = 0;
            let mut skip_s: &str = "";
            // Specially recognize newline. Otherwise it matches something we don't actually
            // want.
            if s.starts_with('\n') {
                if self.str_to_id("<0x0A>").is_some() {
                    best_candidate = "<0x0A>";
                    best_candidate_len = best_candidate.len();
                    skip_s = &s[1..];
                } else {
                    best_candidate = "\\n";
                }
            } else {
                for (piece_str, _piece_info) in self.pieces.iter() {
                    if s.starts_with(piece_str) && best_candidate_len < piece_str.len() {
                        best_candidate = piece_str;
                        best_candidate_len = piece_str.len();
                        skip_s = &s[piece_str.len()..];
                    }
                }
            }
            if best_candidate_len == 0 {
                // Skip token.
                s = s.get(1..).unwrap_or("");
            } else {
                result.push(best_candidate);
                s = skip_s;
            }
        }
        result
    }

    pub fn tokenize_to_ids<S: AsRef<str>>(&self, s: S) -> Vec<TokenId> {
        let mut s: String = format!("▁{}", s.as_ref());
        // Replace all space characters with a special token.
        s = s.replace(' ', "▁");

        let pieces = self.tokenize_to_pieces(s);
        let mut result = Vec::new();
        result.push(1); // start token
        for piece in pieces {
            let piece_info = self.pieces.get(piece).unwrap();
            result.push(piece_info.idx as i32);
        }
        result
    }
}
