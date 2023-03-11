fn main() {
    protobuf_codegen::Codegen::new()
        .pure()
        .out_dir("src/protomodels")
        .include("proto")
        .input("proto/sentencepiece_model.proto")
        .run()
        .unwrap();
}
