//! Resumable parsing for TFRecord streams.

use byteorder::{ByteOrder, LittleEndian};
use std::io::{self, Read};

use crate::masked_crc::MaskedCrc;

const LENGTH_CRC_OFFSET: usize = 8;
const DATA_OFFSET: usize = LENGTH_CRC_OFFSET + 4;
const HEADER_LENGTH: usize = DATA_OFFSET;
const FOOTER_LENGTH: usize = 4;

/// State for reading one `TfRecord`, potentially over multiple attempts to handle growing,
/// partially flushed files.
//
// From TensorFlow `record_writer.cc` comments:
// Format of a single record:
//  uint64    length
//  uint32    masked crc of length
//  byte      data[length]
//  uint32    masked crc of data
pub struct TfRecordState {
    /// TFRecord header: little-endian u64 length, u32 length-CRC. This vector always has capacity
    /// `HEADER_LENGTH`.
    //
    // Could be replaced by an inline `[u8; HEADER_LENGTH]` buffer plus `usize` length field to
    // avoid a level of memory indirection. Unlikely to matter a lot.
    header: Vec<u8>,
    /// Everything past the header in the TFRecord: the data buffer, plus a little-endian u32 CRC
    /// of the data buffer. Once `header.len() == HEADER_LENGTH`, this will have capacity equal to
    /// the data length plus `FOOTER_LENGTH`; before then, it will have no capacity.
    data_plus_footer: Vec<u8>,
}

impl TfRecordState {
    /// Create an empty `TfRecordState`, ready to read a record from its beginning. This allocates
    /// a vector with 12 bytes of capacity, which will be reused for all records read with this
    /// state value.
    pub fn new() -> Self {
        TfRecordState {
            header: Vec::with_capacity(HEADER_LENGTH),
            data_plus_footer: Vec::new(),
        }
    }
}

impl Default for TfRecordState {
    fn default() -> Self {
        Self::new()
    }
}

/// A TFRecord with a data buffer and expected checksum. The checksum may or may not match the
/// actual contents.
#[derive(Debug)]
pub struct TfRecord {
    /// The payload of the TFRecord.
    pub data: Vec<u8>,
    data_crc: MaskedCrc,
}

/// A buffer's checksum was computed, but it did not match the expected value.
#[derive(Debug, PartialEq, Eq, thiserror::Error)]
#[error("checksum mismatch: got {got}, want {want}")]
pub struct ChecksumError {
    /// The actual checksum of the buffer.
    got: MaskedCrc,
    /// The expected checksum.
    want: MaskedCrc,
}

impl TfRecord {
    /// Validates the integrity of the record by computing its CRC-32-C and checking it against the
    /// expected value.
    pub fn checksum(&self) -> Result<(), ChecksumError> {
        let got = MaskedCrc::compute(&self.data);
        let want = self.data_crc;
        if got == want {
            Ok(())
        } else {
            Err(ChecksumError { got, want })
        }
    }
}

/// Error returned by [`TfRecordState::read_record`].
#[derive(Debug, thiserror::Error)]
pub enum ReadRecordError {
    /// Length field failed checksum. The file is corrupt, and reading must abort.
    #[error("length checksum mismatch: got {}, want {}", .0.got, .0.want)]
    BadLengthCrc(ChecksumError),
    /// No fatal errors so far, but the record is not complete. Call `read_record` again with the
    /// same state buffer once new data may be available.
    ///
    /// This includes the "trivial truncation" case where there are no bytes in a new record, so
    /// repeatedly reading records from a file with zero or more well-formed records will always
    /// finish with a `Truncated` error.
    #[error("record truncated")]
    Truncated,
    /// Record is too large to be represented in memory on this system.
    ///
    /// In principle, it would be possible to recover from this error, but in practice this should
    /// rarely occur since serialized protocol buffers do not exceed 2 GiB in size. Thus, no
    /// recovery codepath has been implemented, so reading must abort.
    #[error("record too large to fit in memory ({0} bytes)")]
    TooLarge(u64),
    /// Underlying I/O error. May be retryable if the underlying error is.
    #[error(transparent)]
    Io(#[from] io::Error),
}

impl TfRecordState {
    /// Attempt to read a TFRecord, pausing gracefully in the face of truncations. If the record is
    /// truncated, the result is a `Truncated` error, and the state buffer will be updated to
    /// contain the prefix of the raw record that was read. The same state buffer should be passed
    /// to a subsequent call to `read_record` that it may continue where it left off. If the record
    /// is read successfully, this `TfRecordState` is left at its default value (equivalent to
    /// `TfRecordState::new`, but without re-allocating) and may be reused by the caller to read a
    /// fresh record.
    ///
    /// The record's length field is always validated against its checksum, but the full data is
    /// only validated if you call `checksum()` on the resulting record.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use rustboard_core::tf_record::{ReadRecordError, TfRecordState};
    /// use std::io::Cursor;
    ///
    /// let mut buf: Vec<u8> = Vec::new();
    /// buf.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00"); // length: 24 bytes
    /// buf.extend(b"\xa3\x7f\x4b\x22"); // length checksum (0x224b7fa3)
    /// let contents = b"\x09\x00\x00\x80\x38\x99\xd6\xd7\x41\x1a\x0dbrain.Event:2";
    /// buf.extend(&contents[..5]); // file truncated mid-write
    ///
    /// let mut state = TfRecordState::new();
    ///
    /// // First attempt: read what we can, then encounter truncation.
    /// assert!(matches!(
    ///     state.read_record(&mut Cursor::new(buf)),
    ///     Err(ReadRecordError::Truncated)
    /// ));
    ///
    /// let mut buf: Vec<u8> = Vec::new();
    /// buf.extend(&contents[5..]); // rest of the payload
    /// buf.extend(b"\x12\x4b\x36\xab"); // data checksum (0xab364b12)
    ///
    /// // Second read: read the rest of the record.
    /// let record = state.read_record(&mut Cursor::new(buf)).unwrap();
    /// assert_eq!(record.data, contents);
    /// assert_eq!(record.checksum(), Ok(()));
    /// ```
    pub fn read_record<R: Read>(&mut self, reader: &mut R) -> Result<TfRecord, ReadRecordError> {
        if self.header.len() < self.header.capacity() {
            read_remaining(reader, &mut self.header)?;

            let (length_buf, length_crc_buf) = self.header.split_at(LENGTH_CRC_OFFSET);
            let length_crc = MaskedCrc(LittleEndian::read_u32(length_crc_buf));
            let actual_crc = MaskedCrc::compute(length_buf);
            if length_crc != actual_crc {
                return Err(ReadRecordError::BadLengthCrc(ChecksumError {
                    got: actual_crc,
                    want: length_crc,
                }));
            }

            let length = LittleEndian::read_u64(length_buf);
            let data_plus_footer_length_u64 = length + (FOOTER_LENGTH as u64);
            let data_plus_footer_length = data_plus_footer_length_u64 as usize;
            if data_plus_footer_length as u64 != data_plus_footer_length_u64 {
                return Err(ReadRecordError::TooLarge(length));
            }
            self.data_plus_footer.reserve_exact(data_plus_footer_length);
        }

        if self.data_plus_footer.len() < self.data_plus_footer.capacity() {
            read_remaining(reader, &mut self.data_plus_footer)?;
        }

        let data_length = self.data_plus_footer.len() - FOOTER_LENGTH;
        let data_crc_buf = self.data_plus_footer.split_off(data_length);
        let data = std::mem::take(&mut self.data_plus_footer);
        let data_crc = MaskedCrc(LittleEndian::read_u32(&data_crc_buf));
        self.header.clear(); // reset; caller may use this again
        Ok(TfRecord { data, data_crc })
    }
}

/// Fill `buf`'s remaining capacity from `reader`, or fail with `Truncated` if the reader is dry.
fn read_remaining<R: Read>(reader: &mut R, buf: &mut Vec<u8>) -> Result<(), ReadRecordError> {
    let want = buf.capacity() - buf.len();
    reader.take(want as u64).read_to_end(buf)?;
    if buf.len() < buf.capacity() {
        return Err(ReadRecordError::Truncated);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::io::Cursor;

    /// A reader that delegates to a sequence of cursors, reading from each in turn and simulating
    /// EOF after each one.
    struct ScriptedReader(VecDeque<Cursor<Vec<u8>>>);

    impl ScriptedReader {
        fn new<I: IntoIterator<Item = Vec<u8>>>(vecs: I) -> Self {
            ScriptedReader(vecs.into_iter().map(Cursor::new).collect())
        }
    }

    impl Read for ScriptedReader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let sub = match self.0.front_mut() {
                None => return Ok(0),
                Some(sub) => sub,
            };
            let read = sub.read(buf)?;
            if read == 0 {
                self.0.pop_front();
            }
            Ok(read)
        }
    }

    mod scripted_reader_tests {
        #[test]
        fn test() {
            let mut sr = super::ScriptedReader::new(vec![
                vec![0, 1, 2, 3],
                vec![],
                vec![4, 5, 6, 7, 8, 9],
                vec![10],
            ]);
            let expected: Vec<Vec<u8>> = vec![
                // Read first buffer, with underread.
                vec![0u8, 1, 2],
                vec![3],
                vec![],
                // Read second buffer, which is empty.
                vec![],
                // Read third buffer, exactly.
                vec![4, 5, 6],
                vec![7, 8, 9],
                vec![],
                // Read fourth buffer, with underread.
                vec![10],
                vec![],
                // Read past end of buffer list.
                vec![],
                vec![],
            ];
            for expected_data in expected {
                use std::io::Read;
                let mut buf = vec![0u8; 3];
                let n = sr.read(&mut buf).unwrap();
                assert_eq!(n, expected_data.len());
                let mut expected_buf = expected_data.clone();
                expected_buf.resize(3, 0u8); // pad
                assert_eq!(buf, expected_buf);
            }
        }
    }

    #[test]
    /// Tests a happy path with multiple records, one of which is truncated.
    fn test_success() {
        // Event file with `tf.summary.scalar("accuracy", 0.99, step=77)`
        // dumped via `xxd logs/*`.
        let record_1a = &b"\x09\x00\x00\x80\x38\x99"[..];
        let record_1b = &b"\xd6\xd7\x41\x1a\x0dbrain.Event:2"[..];
        let record_2 = &b"\
            \x09\xc4\x05\xb7\x3d\x99\xd6\xd7\x41\
            \x10\x4d\x2a\x25\
            \x0a\x23\x0a\x08accuracy\
            \x42\x0a\x08\x01\x12\x00\x22\x04\xa4\x70\x7d\x3f\x4a\
            \x0b\x0a\x09\x0a\x07scalars\
        "[..];

        let mut reads = Vec::new();
        // Record 1: first 5 bytes of header
        reads.push(b"\x18\x00\x00\x00\x00".to_vec());
        // Record 1: next 6 bytes of header
        reads.push(b"\x00\x00\x00\xa3\x7f\x4b".to_vec());
        // Record 1: last byte of header and Part A of contents
        reads.push([&b"\x22"[..], record_1a].concat().to_vec());
        // Record 1: Part B of contents, 4 bytes of footer; Record 2: first 2 bytes of header
        reads.push({
            let mut v = record_1b.to_vec();
            v.extend(b"\x12\x4b\x36\xab");
            v.extend(b"\x32\x00");
            v
        });
        // Record 2: last 10 bytes of header, all contents, all of footer
        reads.push({
            let mut v = Vec::new();
            v.extend(b"\x00\x00\x00\x00\x00\x00\x24\x19\x56\xec");
            v.extend(record_2);
            v.extend(b"\xa5\x5b\x64\x33");
            v
        });

        let mut sr = ScriptedReader::new(reads);
        let mut st = TfRecordState::new();

        #[derive(Debug)]
        enum TestCase {
            Truncated,
            Record(Vec<u8>),
        }
        use TestCase::*;

        let steps: Vec<TestCase> = vec![
            Truncated,
            Truncated,
            Truncated,
            Record([record_1a, record_1b].concat().to_vec()),
            Truncated,
            Record(record_2.to_vec()),
        ];
        for (i, step) in steps.into_iter().enumerate() {
            let result = st.read_record(&mut sr);
            match (step, result) {
                (Truncated, Err(ReadRecordError::Truncated)) => (),
                (Record(v), Ok(r)) if v == r.data => {
                    r.checksum()
                        .unwrap_or_else(|e| panic!("step {}: checksum failure: {:?}", i + 1, e));
                }
                (step, result) => {
                    panic!("step {}: got {:?}, want {:?}", i + 1, result, step);
                }
            }
        }
    }

    #[test]
    fn test_length_crc_mismatch() {
        let mut file = Vec::new();
        file.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00");
        file.extend(b"\x99\x7f\x4b\x55");
        file.extend(b"123456789abcdef012345678");
        file.extend(b"\x00\x00\x00\x00");

        let mut st = TfRecordState::new();
        match st.read_record(&mut Cursor::new(file)) {
            Err(ReadRecordError::BadLengthCrc(ChecksumError {
                got: MaskedCrc(0x224b7fa3),
                want: MaskedCrc(0x554b7f99),
            })) => (),
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn test_data_crc_mismatch() {
        let mut file = Vec::new();
        file.extend(b"\x18\x00\x00\x00\x00\x00\x00\x00");
        file.extend(b"\xa3\x7f\x4b\x22");
        file.extend(b"123456789abcdef012345678");
        file.extend(b"\xdf\x9b\x57\x13"); // 0x13579bdf

        let mut st = TfRecordState::new();
        let record = st.read_record(&mut Cursor::new(file)).expect("read_record");
        assert_eq!(record.data, b"123456789abcdef012345678".to_vec());
        match record.checksum() {
            Err(ChecksumError {
                want: MaskedCrc(0x13579bdf),
                got: _,
            }) => (),
            other => panic!("{:?}", other),
        }
    }

    #[test]
    fn test_error_display() {
        let e = ReadRecordError::BadLengthCrc(ChecksumError {
            got: MaskedCrc(0x01234567),
            want: MaskedCrc(0xfedcba98),
        });
        assert_eq!(
            e.to_string(),
            "length checksum mismatch: got 0x01234567, want 0xfedcba98"
        );

        let e = ReadRecordError::Truncated;
        assert_eq!(e.to_string(), "record truncated");

        let e = ReadRecordError::TooLarge(999);
        assert_eq!(
            e.to_string(),
            "record too large to fit in memory (999 bytes)"
        );

        let io_error = io::Error::new(io::ErrorKind::BrokenPipe, "pipe machine broke");
        let expected_message = io_error.to_string();
        let e = ReadRecordError::Io(io_error);
        assert_eq!(e.to_string(), expected_message);
    }
}
