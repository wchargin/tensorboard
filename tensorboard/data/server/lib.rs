/// Adds two integers together and returns the result. Must not overflow.
///
/// # Examples
///
/// ```
/// assert_eq!(rustboard_core::add(20, 22), 42);
/// ```
pub fn add(x: i32, y: i32) -> i32 {
    x + y
}

#[cfg(test)]
mod test {
    use super::add;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 4), 6);
        assert_eq!(add(999, 999), 1998);
    }
}
