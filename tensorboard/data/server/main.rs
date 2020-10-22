use byteorder::{ByteOrder, LittleEndian};

fn main() {
    let ptr = LittleEndian::read_u32(b"\x2e\x68\x63\x73"); // look, a dependency!
    assert_eq!(ptr, 0x7363682e);
    println!("Hello, server! 2 + 2 = {}", rustboard_core::add(2, 2)); // look, a sibling crate!
}
