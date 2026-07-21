//! ChaCha8Rng + PoW draw-order reference, ported from `shared/chacha8.py`.
//!
//! Produces byte-identical output to the Python reference for cross-language
//! deterministic Ising model generation. Not intended for cryptographic use.

const CONSTANTS: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574]; // "expand 32-byte k"

pub struct ChaCha8Rng {
    key: [u32; 8],
    counter: u64,
    block: [u32; 16],
    idx: usize, // 16 => exhausted, regenerate
}

fn quarter_round(s: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize) {
    s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(16);
    s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(12);
    s[a] = s[a].wrapping_add(s[b]); s[d] ^= s[a]; s[d] = s[d].rotate_left(8);
    s[c] = s[c].wrapping_add(s[d]); s[b] ^= s[c]; s[b] = s[b].rotate_left(7);
}

impl ChaCha8Rng {
    pub fn from_seed(key_bytes: [u8; 32]) -> Self {
        let mut key = [0u32; 8];
        for i in 0..8 {
            key[i] = u32::from_le_bytes(key_bytes[i * 4..i * 4 + 4].try_into().unwrap());
        }
        ChaCha8Rng { key, counter: 0, block: [0; 16], idx: 16 }
    }

    fn regen(&mut self) {
        let mut s = [0u32; 16];
        s[0..4].copy_from_slice(&CONSTANTS);
        s[4..12].copy_from_slice(&self.key);
        s[12] = (self.counter & 0xFFFF_FFFF) as u32;
        s[13] = (self.counter >> 32) as u32;
        s[14] = 0; // stream lo
        s[15] = 0; // stream hi
        let start = s;
        for _ in 0..4 {
            quarter_round(&mut s, 0, 4, 8, 12);
            quarter_round(&mut s, 1, 5, 9, 13);
            quarter_round(&mut s, 2, 6, 10, 14);
            quarter_round(&mut s, 3, 7, 11, 15);
            quarter_round(&mut s, 0, 5, 10, 15);
            quarter_round(&mut s, 1, 6, 11, 12);
            quarter_round(&mut s, 2, 7, 8, 13);
            quarter_round(&mut s, 3, 4, 9, 14);
        }
        for i in 0..16 { self.block[i] = s[i].wrapping_add(start[i]); }
        self.counter += 1;
        self.idx = 0;
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.idx >= 16 { self.regen(); }
        let w = self.block[self.idx];
        self.idx += 1;
        w
    }
}

pub fn draw_ising_milli(
    nonce: [u8; 32], n_nodes: usize, n_edges: usize, allowed_h: &[i32], allowed_j: &[i32],
) -> (Vec<i32>, Vec<i32>) {
    let mut rng = ChaCha8Rng::from_seed(nonce);
    let h = (0..n_nodes).map(|_| allowed_h[(rng.next_u32() as usize) % allowed_h.len()]).collect();
    let j = (0..n_edges).map(|_| allowed_j[(rng.next_u32() as usize) % allowed_j.len()]).collect();
    (h, j)
}
