#![no_std]

#[macro_use]
extern crate std;

extern crate alloc;

use core::convert::TryFrom;
use alloc::vec::Vec;
use num_enum::TryFromPrimitive;

const PNG_MAGIC_BYTES: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10];

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum BitDepth {
    One = 1,
    Two = 2,
    Four = 4,
    Eight = 8,
    Sixteen = 16,
}

// #[repr(u8)]
// #[derive(Debug)]
// pub enum ColorTypeCode {
//     PaletteUsed = 1,
//     ColorUsed = 2,
//     AlphaChannelUsed = 4,
// }

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum ColorType {
    Grayscale = 0,
    Rgb = 2,
    Palette = 3,
    GrayscaleAlpha = 4,
    RgbAlpha = 6,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum CompressionMethod {
    Deflate = 0
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum FilterMethod {
    Adaptive = 0,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum InterlaceMethod {
    None = 0,
    Adam7 = 1,
}

#[derive(Debug)]
pub struct PngHeader {
    width: u32,
    height: u32,
    bit_depth: BitDepth,
    color_type: ColorType,
    compression_method: CompressionMethod,
    filter_method: FilterMethod,
    interlace_method: InterlaceMethod,
}

impl PngHeader {
    fn from_chunk(chunk: &Chunk) -> Result<Self, DecodeError> {
        if chunk.chunk_type != ChunkType::ImageHeader {
            return Err(DecodeError::Blah);
        }

        let width = read_u32(chunk.data, 0);
        let height = read_u32(chunk.data, 4);
        let bit_depth = chunk.data[8];
        let color_type = chunk.data[9];
        let compression_method = chunk.data[10];
        let filter_method = chunk.data[11];
        let interlace_method = chunk.data[12];

        // Err(DecodeError::Blah)
        Ok(PngHeader {
            width,
            height,
            bit_depth: TryFrom::try_from(bit_depth).map_err(|_| DecodeError::InvalidBitDepth)?,
            color_type: TryFrom::try_from(color_type).map_err(|_| DecodeError::InvalidColorType)?,
            compression_method: TryFrom::try_from(compression_method).map_err(|_| DecodeError::InvalidCompressionMethod)?,
            filter_method: TryFrom::try_from(filter_method).map_err(|_| DecodeError::InvalidFilterMethod)?,
            interlace_method: TryFrom::try_from(interlace_method).map_err(|_| DecodeError::InvalidInterlaceMethod)?,
        })
    }
}

#[derive(Debug)]
pub enum DecodeError {
    InvalidMagicBytes,
    MissingBytes,
    HeaderChunkNotFirst,
    EndChunkNotLast,
    InvalidChunkType,
    InvalidChunk,

    InvalidBitDepth,
    InvalidColorType,
    InvalidCompressionMethod,
    InvalidFilterMethod,
    InvalidInterlaceMethod,

    Blah,
}

#[derive(Debug)]
pub struct Color {
    r: u8,
    g: u8,
    b: u8,
    a: u8,
}

#[derive(Debug, PartialEq)]
pub enum ChunkType {
    ImageHeader,
    Palette,
    Transparency,
    Srgb,
    ImageData,
    ImageEnd,
    Gamma,
}

impl ChunkType {
    fn from_bytes(bytes: &[u8; 4]) -> Result<Self, DecodeError> {
        match bytes {
            b"IHDR" => Ok(ChunkType::ImageHeader),
            b"PLTE" => Ok(ChunkType::Palette),
            b"tRNS" => Ok(ChunkType::Transparency),
            b"sRGB" => Ok(ChunkType::Srgb),
            b"IDAT" => Ok(ChunkType::ImageData),
            b"IEND" => Ok(ChunkType::ImageEnd),
            b"gAMA" => Ok(ChunkType::Gamma),
            chunk_type => {
                println!("chunk_type: {:?}", alloc::string::String::from_utf8(chunk_type.to_vec()));
                Err(DecodeError::InvalidChunkType)
            }
        }
    }
}

#[derive(Debug)]
struct Chunk<'a> {
    length: u32,
    chunk_type: ChunkType,
    data: &'a [u8],
    crc: u32,
}

impl<'a> Chunk<'a> {
    fn byte_size(&self) -> usize {
        // length bytes + chunk type bytes + data bytes + crc bytes
        4 + 4 + self.length as usize + 4
    }
}

fn read_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([bytes[offset], bytes[offset + 1], bytes[offset + 2], bytes[offset + 3]])
}

fn read_chunk(bytes: &[u8]) -> Result<Chunk, DecodeError> {
    if bytes.len() < 4 {
        return Err(DecodeError::MissingBytes);
    }

    let length = read_u32(bytes, 0);
    let bytes = &bytes[4..];

    if bytes.len() < (4 + length as usize + 4) {
        return Err(DecodeError::MissingBytes);
    }

    let chunk_type = ChunkType::from_bytes(&[bytes[0], bytes[1], bytes[2], bytes[3]])?;

    let crc_offset = 4 + length as usize;
    let crc = read_u32(bytes, crc_offset);

    Ok(Chunk {
        length,
        chunk_type,
        data: &bytes[4..(4 + length as usize)],
        crc,
    })
}

pub fn decode(bytes: &[u8]) -> Result<(PngHeader, Vec<Color>), DecodeError> {
    if bytes.len() < PNG_MAGIC_BYTES.len() {
        return Err(DecodeError::MissingBytes);
    }

    if &bytes[0..PNG_MAGIC_BYTES.len()] != PNG_MAGIC_BYTES {
        return Err(DecodeError::MissingBytes);
    }

    let bytes = &bytes[PNG_MAGIC_BYTES.len()..];

    let header_chunk = read_chunk(bytes)?;
    println!("header chunk: {:?}", header_chunk);
    let header = PngHeader::from_chunk(&header_chunk)?;
    println!("Png Header: {:#?}", header);

    let mut bytes = &bytes[header_chunk.byte_size()..];

    while bytes.len() > 0 {
        let chunk = read_chunk(bytes)?;
        bytes = &bytes[chunk.byte_size()..];

        println!("chunk: {:?}", chunk);
    }

    Err(DecodeError::Blah)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test_png_bytes = include_bytes!("../test_pngs/png_suite/basi0g01.png");
        let result = decode(test_png_bytes);

        println!("Result: {:?}", result);
    }
}
