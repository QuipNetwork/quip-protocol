#[derive(Debug, PartialEq)]
pub enum WireError { BadLength, BadSpinByte(u8) }

impl std::fmt::Display for WireError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WireError::BadLength => write!(f, "byte length is not a multiple of 4"),
            WireError::BadSpinByte(b) => write!(f, "invalid spin byte: 0x{b:02X}"),
        }
    }
}

impl std::error::Error for WireError {}

pub fn encode_i32_le(values: &[i32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values { out.extend_from_slice(&v.to_le_bytes()); }
    out
}

pub fn decode_i32_le(bytes: &[u8]) -> Result<Vec<i32>, WireError> {
    if !bytes.len().is_multiple_of(4) { return Err(WireError::BadLength); }
    Ok(bytes.chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

pub fn encode_spins(spins: &[i8]) -> Vec<u8> {
    spins.iter().map(|&s| if s >= 0 { 0x01u8 } else { 0xFFu8 }).collect()
}

pub fn decode_spins(bytes: &[u8]) -> Result<Vec<i8>, WireError> {
    bytes.iter().map(|&b| match b {
        0x01 => Ok(1i8),
        0xFF => Ok(-1i8),
        other => Err(WireError::BadSpinByte(other)),
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i32_le_roundtrip_and_bytes() {
        // -1000 milli little-endian = 0x18 0xFC 0xFF 0xFF
        assert_eq!(encode_i32_le(&[-1000, 0, 1000]),
            vec![0x18, 0xFC, 0xFF, 0xFF, 0, 0, 0, 0, 0xE8, 0x03, 0, 0]);
        assert_eq!(decode_i32_le(&encode_i32_le(&[-1000, 0, 1000])).unwrap(),
            vec![-1000, 0, 1000]);
        assert!(matches!(decode_i32_le(&[1, 2, 3]), Err(WireError::BadLength)));
    }

    #[test]
    fn spins_bytes() {
        assert_eq!(encode_spins(&[1, -1, 1]), vec![0x01, 0xFF, 0x01]);
        assert_eq!(decode_spins(&[0x01, 0xFF, 0x01]).unwrap(), vec![1, -1, 1]);
        assert!(matches!(decode_spins(&[0x00]), Err(WireError::BadSpinByte(0))));
    }
}
