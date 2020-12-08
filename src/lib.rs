#![no_std]

extern crate alloc;

#[macro_use]
extern crate std;

use alloc::vec::Vec;
use core::convert::TryFrom;
use miniz_oxide::inflate::{
    core::{inflate_flags, DecompressorOxide},
    TINFLStatus,
};
use num_enum::TryFromPrimitive;

const PNG_MAGIC_BYTES: &[u8] = &[137, 80, 78, 71, 13, 10, 26, 10];

#[repr(u8)]
#[derive(Debug, Copy, Clone, TryFromPrimitive)]
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

impl ColorType {
    pub fn sample_multiplier(&self) -> u32 {
        match self {
            ColorType::Grayscale => 1,
            ColorType::Rgb => 3,
            ColorType::Palette => 1,
            ColorType::GrayscaleAlpha => 2,
            ColorType::RgbAlpha => 4,
        }
    }
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum CompressionMethod {
    Deflate = 0,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum FilterMethod {
    Adaptive = 0,
}

#[repr(u8)]
#[derive(Debug, TryFromPrimitive)]
pub enum FilterType {
    None = 0,
    Sub = 1,
    Up = 2,
    Average = 3,
    Paeth = 4,
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
            compression_method: TryFrom::try_from(compression_method)
                .map_err(|_| DecodeError::InvalidCompressionMethod)?,
            filter_method: TryFrom::try_from(filter_method)
                .map_err(|_| DecodeError::InvalidFilterMethod)?,
            interlace_method: TryFrom::try_from(interlace_method)
                .map_err(|_| DecodeError::InvalidInterlaceMethod)?,
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
    Decompress(TINFLStatus),

    InvalidBitDepth,
    InvalidColorType,
    InvalidCompressionMethod,
    InvalidFilterMethod,
    InvalidFilterType,
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
    Unknown([u8; 4]),
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
            unknown_chunk_type => {
                // println!("chunk_type: {:?}", alloc::string::String::from_utf8(chunk_type.to_vec()));
                Ok(ChunkType::Unknown(*unknown_chunk_type))
            },
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

    Ok(Chunk { length, chunk_type, data: &bytes[4..(4 + length as usize)], crc })
}

fn defilter(
    header: &PngHeader,
    scanline_data: &mut [u8],
    output_rgba: &mut [u8],
) -> Result<(), DecodeError> {
    let mut cursor = 0;
    let bytes_per_pixel =
        ((header.bit_depth as u32 * header.color_type.sample_multiplier()) + 7) / 8;
    println!("bytes_per_pixel - {}", bytes_per_pixel);

    match header.interlace_method {
        InterlaceMethod::None => {},
        InterlaceMethod::Adam7 => {
            // let max_bytes_per_scanline = ((header.width * header.bit_depth as u32 * header.color_type.sample_multiplier()) + 7) / 8;
            let max_bytes_per_scanline = header.width * bytes_per_pixel;
            let mut last_scanline = vec![0u8; max_bytes_per_scanline as usize];

            for pass in 1..=7 {
                let (pass_width, pass_height) = match pass {
                    1 => {
                        let pass_width = (header.width + 7) / 8;
                        let pass_height = (header.height + 7) / 8;
                        (pass_width, pass_height)
                    },
                    2 => {
                        let pass_width = (header.width / 8) + ((header.width % 8) / 4);
                        let pass_height = (header.height + 7) / 8;
                        (pass_width, pass_height)
                    },
                    3 => {
                        let pass_width = (header.width + 3) / 4;
                        let pass_height = (header.height / 8) + ((header.height % 8) / 4);
                        (pass_width, pass_height)
                    },
                    4 => {
                        let pass_width = (header.width / 4) + (header.width % 4) / 2;
                        let pass_height = (header.width + 3) / 4;
                        (pass_width, pass_height)
                    },
                    5 => {
                        let pass_width = (header.width / 2) + (header.width % 2);
                        let pass_height = (header.height / 4) + (header.height % 4) / 2;
                        (pass_width, pass_height)
                    },
                    6 => {
                        let pass_width = header.width / 2;
                        let pass_height = (header.height / 2) + (header.height % 2);
                        (pass_width, pass_height)
                    },
                    7 => {
                        let pass_width = header.width;
                        let pass_height = header.height / 2;
                        (pass_width, pass_height)
                    },
                    _ => (0, 0),
                };

                // Skip empty passes.
                if pass_width == 0 || pass_height == 0 {
                    println!("Pass {} - Empty", pass);
                    continue;
                }

                let bytes_per_scanline = ((pass_width
                    * header.bit_depth as u32
                    * header.color_type.sample_multiplier())
                    + 7)
                    / 8;
                println!(
                    "Pass {}, {}x{}, bytes_per_scanline={}",
                    pass, pass_width, pass_height, bytes_per_scanline
                );

                let last_scanline = &mut last_scanline[..(bytes_per_scanline as usize)];
                for byte in last_scanline.iter_mut() {
                    *byte = 0;
                }

                for _y in 0..pass_height {
                    let filter_type = FilterType::try_from(scanline_data[cursor])
                        .map_err(|_| DecodeError::InvalidFilterType)?;
                    cursor += 1;

                    println!("Filter type: {:?}", filter_type);
                    let current_scanline =
                        &mut scanline_data[cursor..(cursor + bytes_per_scanline as usize)];

                    for x in 0..(bytes_per_scanline as usize) {
                        let unfiltered_byte = match filter_type {
                            FilterType::None => current_scanline[x],
                            FilterType::Sub => {
                                if let Some(idx) = x.checked_sub(bytes_per_pixel as usize) {
                                    current_scanline[x] + current_scanline[idx]
                                } else {
                                    current_scanline[x]
                                }
                            },
                            FilterType::Up => current_scanline[x] + last_scanline[x],
                            FilterType::Average => {
                                if let Some(idx) = x.checked_sub(bytes_per_pixel as usize) {
                                    (current_scanline[x] + current_scanline[idx]) / 2
                                } else {
                                    last_scanline[x] / 2
                                }
                            },
                            FilterType::Paeth => {
                                // TODO - Implement
                                current_scanline[x]
                            },
                        };

                        println!("unfiltered_byte: {}", unfiltered_byte);
                        current_scanline[x] = unfiltered_byte;
                    }

                    last_scanline.copy_from_slice(current_scanline);

                    cursor += bytes_per_scanline as usize;
                }
            }
        },
    }

    Ok(())
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

    let mut compressed_data: Vec<u8> =
        Vec::with_capacity(header.width as usize * header.height as usize * 3);

    while bytes.len() > 0 {
        let chunk = read_chunk(bytes)?;

        if chunk.chunk_type == ChunkType::ImageData {
            compressed_data.extend_from_slice(chunk.data);
        }

        bytes = &bytes[chunk.byte_size()..];
    }

    let mut scanline_data = miniz_oxide::inflate::decompress_to_vec_zlib(&compressed_data)
        .map_err(|e| DecodeError::Decompress(e))?;

    let mut output_rgba = vec![0u8; header.width as usize * header.height as usize * 4];

    // Defilter bytes
    defilter(&header, &mut scanline_data, &mut output_rgba)?;

    // Deinterlace if needed

    println!("scanline_data: {:?}", scanline_data);

    println!("Decompression success!");
    println!("Final image size: {}", scanline_data.len());

    Err(DecodeError::Blah)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let test_png_bytes = include_bytes!("../test_pngs/png_suite/basi0g01.png");
        let result = decode(test_png_bytes);

        // let test_png_bytes = include_bytes!("../test_pngs/png_suite/oi4n2c16.png");
        // let result = decode(test_png_bytes);

        // let test_png_bytes = include_bytes!("../test_pngs/1x1.png");
        // let result = decode(test_png_bytes);

        let test_png_bytes = include_bytes!("../test_pngs/3x3.png");
        let result = decode(test_png_bytes);

        println!("Result: {:?}", result);
    }
}
