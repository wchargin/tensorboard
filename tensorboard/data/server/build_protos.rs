use std::env;
use std::fs;
use std::path::Path;
use std::process::exit;

fn main() -> std::io::Result<()> {
    if env::var_os("PROTOC").is_none() {
        eprintln!("must set PROTOC env var to path to protoc(1)");
        exit(1);
    }

    let args = env::args_os().skip(1).collect::<Vec<_>>();
    let out_path = args.get(0).expect("must give output file as first arg");
    let proto = args
        .get(1)
        .expect("must give main .proto file as second arg");

    let out_dir = Path::new("/tmp/gen_protos_out");
    fs::DirBuilder::new()
        .recursive(true)
        .create(&out_dir)
        .unwrap();
    prost_build::Config::new()
        .out_dir(&out_dir)
        .compile_protos(&[proto.into()], &[std::ffi::OsString::from(".")])
        .expect("compile_protos");
    fs::copy(out_dir.join("tensorboard.rs"), out_path)?;
    Ok(())
}
