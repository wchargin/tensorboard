//! Checksums as used by TFRecords.

use std::fmt::{self, Debug, Display};

/// A CRC-32-C (Castagnoli) checksum after a masking permutation.
///
/// This is the checksum format used by TFRecords. The masking permutation is [the same as in
/// LevelDB][ldb], which offers as motivation that "it is problematic to compute the CRC of a
/// string that contains embedded CRCs".
///
/// [ldb]: https://github.com/google/leveldb/blob/f67e15e50f392625b4097caf22e8be1b0fe96013/util/crc32c.h#L28-L30
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct MaskedCrc(pub u32);

impl Debug for MaskedCrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MaskedCrc({})", self)
    }
}

impl Display for MaskedCrc {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:#010x?}", self.0)
    }
}

const CRC_MASK_DELTA: u32 = 0xa282ead8;

/// Apply a masking operation to an unmasked CRC-32-C.
fn mask(crc: u32) -> MaskedCrc {
    MaskedCrc(((crc >> 15) | (crc << 17)).wrapping_add(CRC_MASK_DELTA))
}

impl MaskedCrc {
    /// Compute a `MaskedCrc` from a data buffer.
    ///
    /// # Examples
    ///
    /// ```
    /// use rustboard_core::masked_crc::MaskedCrc;
    ///
    /// // Checksum extracted from real TensorFlow event file with record:
    /// // tf.compat.v1.Event(file_version=b"CRC test, one two")
    /// let data = b"\x1a\x11CRC test, one two";
    /// assert_eq!(MaskedCrc::compute(data), MaskedCrc(0x5794d08a));
    /// ```
    pub fn compute(bytes: &[u8]) -> Self {
        mask(crc::crc32::checksum_castagnoli(bytes))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute() {
        // From RFC 3720 (iSCSI), section B.4 "CRC Examples".
        assert_eq!(MaskedCrc::compute(&[0; 32]), mask(0x8a9136aa));

        // From a real TFRecord (`file_version` event), with authoritative masking.
        assert_eq!(
            MaskedCrc::compute(b"\x1a\x11CRC test, one two"),
            MaskedCrc(0x5794d08a)
        );
    }

    #[test]
    fn test_debug() {
        let long_crc = MaskedCrc(0xf1234567);
        assert_eq!(format!("{}", long_crc), "0xf1234567");
        assert_eq!(format!("{:?}", long_crc), "MaskedCrc(0xf1234567)");

        let short_crc = MaskedCrc(0x00000123);
        assert_eq!(format!("{}", short_crc), "0x00000123");
        assert_eq!(format!("{:?}", short_crc), "MaskedCrc(0x00000123)");
    }
}
