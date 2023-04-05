use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;

pub struct Unpickler {}

use crate::tensor::{TensorBuilder, TensorDType, TensorError};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum UnpicklingError {
    #[error("Unpickling error: {0}")]
    UnpicklingError(String),
    #[error("UTF-8 decoding error")]
    Utf8Error(#[from] std::str::Utf8Error),
    #[error("Missing field")]
    MissingField(String),
    #[error("Tensor conversion operation failed")]
    TensorError(#[from] TensorError),
    #[error("Data has incorrect format to be converted to a tensor")]
    InvalidTensorData,
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
pub enum Value {
    Mark(usize),
    String(String),
    Global(String, String), // module name, attribute name
    Integer64(i64),
    Tuple(Vec<Value>),
    PersistentId(Box<Value>),
    Bool(bool),
    Reduce(Box<Value>, Box<Value>),
    Dict(BTreeMap<Value, Value>),
}

impl Value {
    // Gets a value from a dictionary, assuming Value is a dictionary.
    //
    // Returns None if the key is not found, or the value is not a dictionary.
    pub fn get(&self, key: &Value) -> Option<&Value> {
        match self {
            Value::Dict(d) => d.get(key),
            _ => None,
        }
    }

    // Same as get() but uses a string as key.
    pub fn get_str_key<S: AsRef<str>>(&self, key: S) -> Option<&Value> {
        self.get(&Value::String(key.as_ref().to_string()))
    }

    // Same as get_str_key but tries two keys, returning the first one that is found.
    pub fn get_str_key2<S: AsRef<str>, S2: AsRef<str>>(
        &self,
        key: S,
        key2: S2,
    ) -> Option<(String, &Value)> {
        let key = key.as_ref();
        let key2 = key2.as_ref();
        match self.get_str_key(key) {
            Some(v) => Some((key.to_string(), v)),
            None => match self.get_str_key(key2) {
                Some(v) => Some((key2.to_string(), v)),
                None => None,
            },
        }
    }

    // Returns all keys as a set of strings, if the value is a dictionary. Otherwise returns empty set.
    pub fn keys(&self) -> BTreeSet<String> {
        match self {
            Value::Dict(d) => {
                let mut result = BTreeSet::new();
                for (k, _v) in d.iter() {
                    match k {
                        Value::String(s) => {
                            result.insert(s.clone());
                        }
                        _ => {}
                    }
                }
                result
            }
            _ => BTreeSet::new(),
        }
    }

    // Merges value dictionaries together
    //
    // Panics if there are duplicate keys.
    pub fn merge_dicts(dicts: &[Self]) -> Self {
        if dicts.is_empty() {
            return Value::Dict(BTreeMap::new());
        }
        let mut result = dicts[0].clone();
        for dict in dicts.iter().skip(1) {
            match (result, dict) {
                (Value::Dict(mut d1), Value::Dict(d2)) => {
                    for (k, v) in d2 {
                        d1.insert(k.clone(), v.clone());
                    }
                    result = Value::Dict(d1);
                }
                _ => panic!("Can only merge dictionaries"),
            }
        }
        result
    }

    pub fn get_global(&self) -> Option<(&str, &str)> {
        match self {
            Value::Global(module_name, attribute_name) => Some((module_name, attribute_name)),
            _ => None,
        }
    }

    pub fn get_str(&self) -> Option<&str> {
        match self {
            Value::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn get_int64(&self) -> Option<i64> {
        match self {
            Value::Integer64(i) => Some(*i),
            _ => None,
        }
    }

    pub fn get_persistent_id(&self) -> Option<&Value> {
        match self {
            Value::PersistentId(v) => Some(v),
            _ => None,
        }
    }

    pub fn get_tuple(&self) -> Option<&[Value]> {
        match self {
            Value::Tuple(v) => Some(v),
            _ => None,
        }
    }

    // Assume that the value represents a tensor in PyTorch and return instructions how to actually
    // load the values.
    pub fn to_tensor_builder(&self, tensor_name: String) -> Option<TensorBuilder> {
        match self {
            Value::Reduce(call, args) => match **call {
                Value::Global(ref module_name, ref attribute_name) => {
                    if module_name == "torch._utils" && attribute_name == "_rebuild_tensor_v2" {
                        match **args {
                            Value::Tuple(ref args) => self.to_tensor_builder2(tensor_name, args),
                            _ => None,
                        }
                    } else {
                        None
                    }
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn to_tensor_builder2(&self, tensor_name: String, args: &[Value]) -> Option<TensorBuilder> {
        if args.len() == 6 {
            Self::to_tensor_builder2_6items(tensor_name, args)
        } else {
            None
        }
    }

    fn to_tensor_builder2_6items(tensor_name: String, args: &[Value]) -> Option<TensorBuilder> {
        let storagev: &Value = args[0].get_persistent_id()?;
        let storage_args: &[Value] = storagev.get_tuple()?;
        let storage_mark: &str = storage_args[0].get_str()?;
        if storage_mark != "storage" {
            return None;
        }

        let (storage_module, storage_type) = storage_args[1].get_global()?;
        if storage_module != "torch" {
            return None;
        }
        let dtype: TensorDType = match storage_type {
            "HalfStorage" => TensorDType::Float16,
            _ => {
                return None;
            }
        };
        let storage_filename: &str = storage_args[2].get_str()?;
        let nitems: i64 = storage_args[4].get_int64()?;

        let offset: i64 = args[1].get_int64()?;

        let shape: &[Value] = args[2].get_tuple()?;
        let stride: &[Value] = args[3].get_tuple()?;

        if shape.len() != 2 && shape.len() != 1 {
            return None;
        }
        if stride.len() != 2 && stride.len() != 1 {
            return None;
        }

        let (rows, cols) = if shape.len() == 2 {
            (shape[0].get_int64()?, shape[1].get_int64()?)
        } else {
            let cols = shape[0].get_int64()?;
            (1, cols)
        };

        let (row_stride, col_stride) = if stride.len() == 1 {
            let (r, c) = (stride[0].get_int64()?, 1);
            if r != 1 {
                return None;
            }
            (r, c)
        } else {
            (stride[0].get_int64()?, stride[1].get_int64()?)
        };

        if col_stride != 1 {
            return None;
        }
        if row_stride != cols && stride.len() == 2 {
            return None;
        }

        Some(TensorBuilder {
            src_path: PathBuf::from(storage_filename),
            tensor_name,
            dtype,
            stride: row_stride,
            rows,
            cols,
            nitems,
            offset,
        })

        /* Args should look like this (took random example from debug print) :
            0 PERSISTENT_ID
                TUPLE
                  STRING "storage"
                  GLOBAL "torch" "HalfStorage"
                  STRING "0"                    (filename)
                  STRING "cpu"
                  INTEGER 131072000             (number of items)
            1 INTEGER 0
            2 TUPLE
                INTEGER 32000
                INTEGER 4096
            3 TUPLE
                INTEGER 4096
                INTEGER 1
            4 BOOL false        (this is about gradient)
            5 REDUCE            (no idea why this is here)
                GLOBAL "collections" "OrderedDict"
                TUPLE

            Sometimes arguments 2 and 3 are missing.
        */
    }

    // Print a nice representation of the value to stdout. Used for good old printf debugging.
    pub fn debug_print(&self) {
        self.debug_print_go(0);
    }

    fn debug_print_go(&self, indent: usize) {
        if indent > 0 {
            print!("{:indent$}", "", indent = indent);
        }
        match self {
            Value::Mark(_) => {
                println!("MARK");
            }
            Value::String(s) => {
                println!("STRING {:?}", s);
            }
            Value::Global(module_name, attribute_name) => {
                println!("GLOBAL {:?} {:?}", module_name, attribute_name);
            }
            Value::Integer64(i) => {
                println!("INTEGER {:?}", i);
            }
            Value::Tuple(v) => {
                println!("TUPLE");
                for i in v {
                    i.debug_print_go(indent + 2);
                }
            }
            Value::PersistentId(v) => {
                println!("PERSISTENT_ID");
                v.debug_print_go(indent + 2);
            }
            Value::Bool(b) => {
                println!("BOOL {:?}", b);
            }
            Value::Reduce(v1, v2) => {
                println!("REDUCE");
                v1.debug_print_go(indent + 2);
                v2.debug_print_go(indent + 2);
            }
            Value::Dict(d) => {
                println!("DICT");
                for (k, v) in d {
                    k.debug_print_go(indent + 2);
                    v.debug_print_go(indent + 2);
                }
            }
        }
    }
}

pub fn unpickle(bytes: &[u8]) -> Result<Value, UnpicklingError> {
    // The LLaMA file is in pickle 2 format, check that header is there
    if bytes.len() < 2 {
        return Err(UnpicklingError::UnpicklingError(
            "Data is too short to be a pickle".to_string(),
        ));
    }

    if bytes[0] != 128 || bytes[1] != 2 {
        return Err(UnpicklingError::UnpicklingError(
            "No magic header using Pickle 2 protocol".to_string(),
        ));
    }

    let mut memo: BTreeMap<u32, Value> = BTreeMap::new();
    let mut stack: Vec<Value> = vec![];

    // Decode frames
    let mut bytes: &[u8] = &bytes[2..];
    while !bytes.is_empty() {
        let frame_opcode = bytes[0];
        if frame_opcode == 125 {
            // empty dict
            stack.push(Value::Dict(BTreeMap::new()));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 113 {
            // binput
            if bytes.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINPUT".to_string(),
                ));
            }
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling BINPUT".to_string(),
                ));
            }
            let key = bytes[1];
            memo.insert(key as u32, stack.last().unwrap().clone());
            bytes = &bytes[2..];
            continue;
        }
        if frame_opcode == 40 {
            // mark
            stack.push(Value::Mark(stack.len()));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 88 {
            // binunicode
            if bytes.len() < 5 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINUNICODE".to_string(),
                ));
            }
            let len = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
            if bytes.len() < 5 + len as usize {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINUNICODE".to_string(),
                ));
            }
            let string = std::str::from_utf8(&bytes[5..5 + len as usize])?;
            stack.push(Value::String(string.to_string()));
            bytes = &bytes[5 + len as usize..];
            continue;
        }
        if frame_opcode == 99 {
            // global
            // followed by newline terminated module name and attribute name
            bytes = &bytes[1..];
            let mut module_name = String::new();
            while !bytes.is_empty() && bytes[0] != 10 {
                module_name.push(bytes[0] as char);
                bytes = &bytes[1..];
                if bytes.is_empty() {
                    return Err(UnpicklingError::UnpicklingError(
                        "Unexpected end of data while handling GLOBAL".to_string(),
                    ));
                }
            }
            bytes = &bytes[1..];
            let mut attribute_name = String::new();
            while !bytes.is_empty() && bytes[0] != 10 {
                attribute_name.push(bytes[0] as char);
                bytes = &bytes[1..];
                if bytes.is_empty() {
                    return Err(UnpicklingError::UnpicklingError(
                        "Unexpected end of data while handling GLOBAL".to_string(),
                    ));
                }
            }
            bytes = &bytes[1..];
            stack.push(Value::Global(module_name, attribute_name));
            continue;
        }
        if frame_opcode == 74 {
            // binint
            if bytes.len() < 5 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BININT".to_string(),
                ));
            }
            let value = i32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
            stack.push(Value::Integer64(value as i64));
            bytes = &bytes[5..];
            continue;
        }
        if frame_opcode == 116 {
            // tuple
            let mut tuple = vec![];
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling TUPLE".to_string(),
                ));
            }
            let mut ok = false;
            while !stack.is_empty() {
                let top = stack.pop().unwrap();
                if let Value::Mark(_mark) = top {
                    tuple.reverse();
                    stack.push(Value::Tuple(tuple));
                    ok = true;
                    break;
                }
                tuple.push(top);
            }
            if !ok {
                return Err(UnpicklingError::UnpicklingError(
                    "No mark while handling TUPLE".to_string(),
                ));
            }
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 81 {
            // binpersid
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling BINPERSID".to_string(),
                ));
            }
            let top = stack.pop().unwrap();
            stack.push(Value::PersistentId(Box::new(top)));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 75 {
            // binint1
            if bytes.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BININT1".to_string(),
                ));
            }
            let value = bytes[1];
            stack.push(Value::Integer64(value as i64));
            bytes = &bytes[2..];
            continue;
        }
        if frame_opcode == 77 {
            // binint2
            if bytes.len() < 3 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BININT2".to_string(),
                ));
            }
            let value = i16::from_le_bytes([bytes[1], bytes[2]]);
            stack.push(Value::Integer64(value as i64));
            bytes = &bytes[3..];
            continue;
        }
        if frame_opcode == 134 {
            // tuple2
            let mut tuple = vec![];
            if stack.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack does not have enough items while handling TUPLE2".to_string(),
                ));
            }
            tuple.push(stack.pop().unwrap());
            tuple.push(stack.pop().unwrap());
            tuple.reverse();
            stack.push(Value::Tuple(tuple));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 137 {
            // newfalse
            stack.push(Value::Bool(false));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 41 {
            // empty tuple
            stack.push(Value::Tuple(vec![]));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 82 {
            // reduce
            if stack.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack does not have enough items while handling REDUCE".to_string(),
                ));
            }
            let arg_tuple = stack.pop().unwrap();
            let callable = stack.pop().unwrap();
            stack.push(Value::Reduce(Box::new(callable), Box::new(arg_tuple)));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 104 {
            // binget
            if bytes.len() < 2 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling BINGET".to_string(),
                ));
            }
            let idx = bytes[1];
            match memo.get(&(idx as u32)) {
                None => {
                    return Err(UnpicklingError::UnpicklingError(
                        "BINGET index out of range".to_string(),
                    ));
                }
                Some(memo_value) => {
                    stack.push(memo_value.clone());
                }
            }
            bytes = &bytes[2..];
            continue;
        }
        if frame_opcode == 133 {
            // tuple1
            let mut tuple = vec![];
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling TUPLE1".to_string(),
                ));
            }
            tuple.push(stack.pop().unwrap());
            stack.push(Value::Tuple(tuple));
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 114 {
            // long binput
            if bytes.len() < 5 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling LONG_BINPUT".to_string(),
                ));
            }
            let key = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling LONG_BINPUT".to_string(),
                ));
            }
            memo.insert(key, stack.last().unwrap().clone());
            bytes = &bytes[5..];
            continue;
        }
        if frame_opcode == 117 {
            // setitems
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling SETITEMS".to_string(),
                ));
            }
            let mut ok = false;
            let mut keyvalues: BTreeMap<Value, Value> = BTreeMap::new();
            while !stack.is_empty() {
                let value = stack.pop().unwrap();
                if let Value::Mark(_mark) = value {
                    ok = true;
                    break;
                }
                if stack.is_empty() {
                    return Err(UnpicklingError::UnpicklingError(
                        "Stack is empty while handling SETITEMS".to_string(),
                    ));
                }
                let key = stack.pop().unwrap();
                if let Value::Mark(_mark) = key {
                    return Err(UnpicklingError::UnpicklingError(
                        "Unexpected mark while handling SETITEMS".to_string(),
                    ));
                }
                keyvalues.insert(key, value);
            }
            if !ok {
                return Err(UnpicklingError::UnpicklingError(
                    "No mark while handling SETITEMS".to_string(),
                ));
            }
            if stack.is_empty() {
                return Err(UnpicklingError::UnpicklingError(
                    "Stack is empty while handling SETITEMS".to_string(),
                ));
            }
            let mut dict = stack.pop().unwrap();
            match dict {
                Value::Dict(ref mut dict) => {
                    for (key, value) in keyvalues {
                        dict.insert(key, value);
                    }
                }
                _ => {
                    return Err(UnpicklingError::UnpicklingError(
                        "SETITEMS on non-dict".to_string(),
                    ));
                }
            }
            stack.push(dict);
            bytes = &bytes[1..];
            continue;
        }
        if frame_opcode == 106 {
            // long_binget
            if bytes.len() < 5 {
                return Err(UnpicklingError::UnpicklingError(
                    "Unexpected end of data while handling LONG_BINGET".to_string(),
                ));
            }
            let idx = u32::from_le_bytes([bytes[1], bytes[2], bytes[3], bytes[4]]);
            match memo.get(&{ idx }) {
                None => {
                    return Err(UnpicklingError::UnpicklingError(
                        "LONG_BINGET index out of range".to_string(),
                    ));
                }
                Some(memo_value) => {
                    stack.push(memo_value.clone());
                }
            }
            bytes = &bytes[5..];
            continue;
        }
        if frame_opcode == 46 {
            // stop
            // bytes = &bytes[1..];
            break;
        }
        return Err(UnpicklingError::UnpicklingError(format!(
            "Unknown opcode: {}",
            frame_opcode
        )));
    }

    // Stack should have just one item, our final value
    if stack.len() != 1 {
        return Err(UnpicklingError::UnpicklingError(
            "Stack does not have exactly one item after unpickling".to_string(),
        ));
    }

    Ok(stack.pop().unwrap())
}
