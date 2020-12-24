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
#[derive(Debug, Copy, Clone, PartialEq, TryFromPrimitive)]
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
#[derive(Debug, Copy, Clone, TryFromPrimitive)]
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

#[derive(Debug, Copy, Clone)]
enum PixelType {
    Grayscale1,
    Grayscale2,
    Grayscale4,
    Grayscale8,

    Rgb8,
    Rgb16,

    Palette1,
    Palette2,
    Palette4,
    Palette8,

    GrayscaleAlpha8,
    GrayscaleAlpha16,

    RgbAlpha8,
    RgbAlpha16,
}

impl PixelType {
    fn new(color_type: ColorType, bit_depth: BitDepth) -> Result<Self, DecodeError> {
        let result = match color_type {
            ColorType::Grayscale => match bit_depth {
                BitDepth::One => PixelType::Grayscale1,
                BitDepth::Two => PixelType::Grayscale2,
                BitDepth::Four => PixelType::Grayscale4,
                BitDepth::Eight => PixelType::Grayscale8,
                _ => return Err(DecodeError::InvalidColorTypeBitDepthCombination),
            },
            ColorType::Rgb => match bit_depth {
                BitDepth::Eight => PixelType::Rgb8,
                BitDepth::Sixteen => PixelType::Rgb16,
                _ => return Err(DecodeError::InvalidColorTypeBitDepthCombination),
            },
            ColorType::Palette => match bit_depth {
                BitDepth::One => PixelType::Palette1,
                BitDepth::Two => PixelType::Palette2,
                BitDepth::Four => PixelType::Palette4,
                BitDepth::Eight => PixelType::Palette8,
                _ => return Err(DecodeError::InvalidColorTypeBitDepthCombination),
            },
            ColorType::GrayscaleAlpha => match bit_depth {
                BitDepth::Eight => PixelType::GrayscaleAlpha8,
                BitDepth::Sixteen => PixelType::GrayscaleAlpha16,
                _ => return Err(DecodeError::InvalidColorTypeBitDepthCombination),
            },
            ColorType::RgbAlpha => match bit_depth {
                BitDepth::Eight => PixelType::RgbAlpha8,
                BitDepth::Sixteen => PixelType::RgbAlpha16,
                _ => return Err(DecodeError::InvalidColorTypeBitDepthCombination),
            },
        };

        Ok(result)
    }
}

struct ScanlineIterator<'a> {
    image_width: usize, // Width in pixels
    pixel_cursor: usize,
    pixel_type: PixelType,
    scanline: &'a [u8],
}

impl<'a> ScanlineIterator<'a> {
    fn new(image_width: u32, pixel_type: PixelType, scanline: &'a [u8]) -> Self {
        // TODO - Assert scanline.len() == bytes_per_pixel * image_width
        Self { image_width: image_width as usize, pixel_cursor: 0, pixel_type, scanline }
    }
}

impl<'a> Iterator for ScanlineIterator<'a> {
    type Item = (u8, u8, u8, u8);

    fn next(&mut self) -> Option<Self::Item> {
        if self.pixel_cursor >= self.image_width {
            return None;
        }

        let pixel = match self.pixel_type {
            PixelType::Grayscale1 => {
                let byte = self.scanline[self.pixel_cursor / 8];
                let bit_offset = 7 - self.pixel_cursor % 8;
                let grayscale_val = (byte >> bit_offset) & 1;
                let pixel_val = grayscale_val * 255;
                Some((pixel_val, pixel_val, pixel_val, 255))
            },
            PixelType::Grayscale2 => {
                let byte = self.scanline[self.pixel_cursor / 4];
                let bit_offset = 6 - ((self.pixel_cursor % 4) * 2);
                let grayscale_val = (byte >> bit_offset) & 0b11;

                // TODO - use a lookup table
                let pixel_val = ((grayscale_val as f32 / 4.0) * 255.0) as u8;
                Some((pixel_val, pixel_val, pixel_val, 255))
            },
            PixelType::Grayscale4 => {
                let byte = self.scanline[self.pixel_cursor / 2];
                let bit_offset = 4 - ((self.pixel_cursor % 2) * 4);
                let grayscale_val = (byte >> bit_offset) & 0b1111;

                // TODO - use a lookup table
                let pixel_val = ((grayscale_val as f32 / 16.0) * 255.0) as u8;
                Some((pixel_val, pixel_val, pixel_val, 255))
            },
            PixelType::Grayscale8 => {
                let byte = self.scanline[self.pixel_cursor];
                Some((byte, byte, byte, 255))
            },
            PixelType::Rgb8 => {
                let offset = self.pixel_cursor * 3;
                let r = self.scanline[offset];
                let g = self.scanline[offset + 1];
                let b = self.scanline[offset + 2];

                Some((r, g, b, 255))
            },
            PixelType::Rgb16 => {
                let offset = self.pixel_cursor * 6;
                let r = u16::from_be_bytes([self.scanline[offset], self.scanline[offset + 1]]);
                let g = u16::from_be_bytes([self.scanline[offset + 2], self.scanline[offset + 3]]);
                let b = u16::from_be_bytes([self.scanline[offset + 4], self.scanline[offset + 5]]);

                let r = ((r as f32 / u16::MAX as f32) * 255.0) as u8;
                let g = ((g as f32 / u16::MAX as f32) * 255.0) as u8;
                let b = ((b as f32 / u16::MAX as f32) * 255.0) as u8;

                Some((r, g, b, 255))
            },
            PixelType::Palette1 => Some((0, 0, 0, 0)),
            PixelType::Palette2 => Some((0, 0, 0, 0)),
            PixelType::Palette4 => Some((0, 0, 0, 0)),
            PixelType::Palette8 => Some((0, 0, 0, 0)),
            PixelType::GrayscaleAlpha8 => {
                let offset = self.pixel_cursor * 2;
                let grayscale_val = self.scanline[offset];
                let alpha = self.scanline[offset + 1];

                Some((grayscale_val, grayscale_val, grayscale_val, alpha))
            },
            PixelType::GrayscaleAlpha16 => {
                let offset = self.pixel_cursor * 4;
                let grayscale_val =
                    u16::from_be_bytes([self.scanline[offset], self.scanline[offset + 1]]);
                let alpha =
                    u16::from_be_bytes([self.scanline[offset + 2], self.scanline[offset + 3]]);

                let grayscale_val = ((grayscale_val as f32 / u16::MAX as f32) * 255.0) as u8;
                let alpha = ((alpha as f32 / u16::MAX as f32) * 255.0) as u8;

                Some((grayscale_val, grayscale_val, grayscale_val, alpha))
            },
            PixelType::RgbAlpha8 => {
                let offset = self.pixel_cursor * 4;
                let r = self.scanline[offset];
                let g = self.scanline[offset + 1];
                let b = self.scanline[offset + 2];
                let a = self.scanline[offset + 3];

                Some((r, g, b, a))
            },
            PixelType::RgbAlpha16 => {
                let offset = self.pixel_cursor * 8;
                let r = u16::from_be_bytes([self.scanline[offset], self.scanline[offset + 1]]);
                let g = u16::from_be_bytes([self.scanline[offset + 2], self.scanline[offset + 3]]);
                let b = u16::from_be_bytes([self.scanline[offset + 4], self.scanline[offset + 5]]);
                let a = u16::from_be_bytes([self.scanline[offset + 6], self.scanline[offset + 7]]);

                let r = ((r as f32 / u16::MAX as f32) * 255.0) as u8;
                let g = ((g as f32 / u16::MAX as f32) * 255.0) as u8;
                let b = ((b as f32 / u16::MAX as f32) * 255.0) as u8;
                let a = ((a as f32 / u16::MAX as f32) * 255.0) as u8;

                Some((r, g, b, a))
            },
        };

        self.pixel_cursor += 1;
        pixel
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
    InvalidColorTypeBitDepthCombination,
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

struct TwoByteProcessor {
    last_byte: u8,
    is_lsb: bool,
}

struct ByteProcessor<'a> {
    png_header: &'a PngHeader,
    pixel_type: PixelType,
    byte_counter: u32,
    bytes_per_scanline: u32,
    output_rgba: &'a mut [u8],
    output_cursor: usize,
    two_byte_processor: Option<TwoByteProcessor>,
}

impl<'a> ByteProcessor<'a> {
    fn new(png_header: &'a PngHeader, output_rgba: &'a mut [u8]) -> Result<Self, DecodeError> {
        let bytes_per_scanline = ((png_header.width
            * png_header.bit_depth as u32
            * png_header.color_type.sample_multiplier())
            + 7)
            / 8;

        let pixel_type = PixelType::new(png_header.color_type, png_header.bit_depth)?;

        let two_byte_processor = if png_header.bit_depth == BitDepth::Sixteen {
            Some(TwoByteProcessor { last_byte: 0, is_lsb: true })
        } else {
            None
        };

        Ok(Self {
            png_header,
            pixel_type,
            byte_counter: 0,
            bytes_per_scanline,
            output_rgba,
            output_cursor: 0,
            two_byte_processor,
        })
    }

    fn handle_byte(&mut self, byte: u8) {
        self.byte_counter += 1;
        let last_byte_on_scanline = self.byte_counter == self.bytes_per_scanline;
        self.byte_counter %= self.bytes_per_scanline; // TODO - this is sus.

        if let Some(two_byte_processor) = &mut self.two_byte_processor {
            two_byte_processor.is_lsb = !two_byte_processor.is_lsb;
            if two_byte_processor.is_lsb {
                // MSB = two_byte_processor.last_byte
                // LSB = byte
                let value = u16::from_be_bytes([two_byte_processor.last_byte, byte]);
                let remapped = ((value as f32 / u16::MAX as f32) * 255.0) as u8;

                // TODO - move cursor forward variable amount depending on color type.
                self.output_rgba[self.output_cursor] = remapped;
                self.output_cursor += 1;
            } else {
                two_byte_processor.last_byte = byte;
            }
        } else {
            match self.png_header.color_type {
                ColorType::Grayscale => match self.png_header.bit_depth {
                    BitDepth::One => {},
                    BitDepth::Two => {},
                    BitDepth::Four => {},
                    BitDepth::Eight => {},
                    _ => unreachable!(),
                },
                ColorType::Rgb => match self.png_header.bit_depth {
                    BitDepth::Eight => {},
                    _ => unreachable!(),
                },
                ColorType::Palette => match self.png_header.bit_depth {
                    BitDepth::One => {},
                    BitDepth::Two => {},
                    BitDepth::Four => {},
                    BitDepth::Eight => {},
                    _ => unreachable!(),
                },
                ColorType::GrayscaleAlpha => match self.png_header.bit_depth {
                    BitDepth::Eight => {},
                    _ => unreachable!(),
                },
                ColorType::RgbAlpha => match self.png_header.bit_depth {
                    BitDepth::Eight => {},
                    _ => unreachable!(),
                },
            }
        }
    }

    fn handle_interlaced_byte(&mut self, byte: u8, pass_num: u8) {}
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

    let pixel_type = PixelType::new(header.color_type, header.bit_depth)?;

    match header.interlace_method {
        InterlaceMethod::None => {},
        InterlaceMethod::Adam7 => {
            // let max_bytes_per_scanline = ((header.width * header.bit_depth as u32 * header.color_type.sample_multiplier()) + 7) / 8;
            let max_bytes_per_scanline = header.width * bytes_per_pixel;
            let mut last_scanline = vec![0u8; max_bytes_per_scanline as usize];

            // Adam7 Interlacing Pattern
            // 1 6 4 6 2 6 4 6
            // 7 7 7 7 7 7 7 7
            // 5 6 5 6 5 6 5 6
            // 7 7 7 7 7 7 7 7
            // 3 6 4 6 3 6 4 6
            // 7 7 7 7 7 7 7 7
            // 5 6 5 6 5 6 5 6
            // 7 7 7 7 7 7 7 7

            for pass in 1..=7 {
                let (pass_width, pass_height) = match pass {
                    1 => {
                        let pass_width = (header.width + 7) / 8;
                        let pass_height = (header.height + 7) / 8;
                        (pass_width, pass_height)
                    },
                    2 => {
                        let pass_width = (header.width / 8) + ((header.width % 8) / 5);
                        let pass_height = (header.height + 7) / 8;
                        (pass_width, pass_height)
                    },
                    3 => {
                        let pass_width = ((header.width / 8) * 2) + (header.width % 8 + 3) / 4;
                        let pass_height = (header.height / 8) + ((header.height % 8) / 5);
                        (pass_width, pass_height)
                    },
                    4 => {
                        let pass_width = ((header.width / 8) * 2) + (header.width % 8) / 6;
                        let pass_height = (header.height + 3) / 4;
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

                for y in 0..pass_height {
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
                                    current_scanline[x].wrapping_add(current_scanline[idx])
                                } else {
                                    current_scanline[x]
                                }
                            },
                            FilterType::Up => current_scanline[x] + last_scanline[x],
                            FilterType::Average => {
                                if let Some(idx) = x.checked_sub(bytes_per_pixel as usize) {
                                    let avg = (current_scanline[x] as u16
                                        + current_scanline[idx] as u16)
                                        / 2;
                                    avg as u8
                                } else {
                                    last_scanline[x] / 2
                                }
                            },
                            FilterType::Paeth => {
                                if let Some(idx) = x.checked_sub(bytes_per_pixel as usize) {
                                    let left = current_scanline[idx];
                                    let above = last_scanline[x];
                                    let upper_left = last_scanline[idx];

                                    let predictor = paeth_predictor(left, above, upper_left);

                                    current_scanline[x] + predictor
                                } else {
                                    let left = 0;
                                    let above = last_scanline[x];
                                    let upper_left = 0;

                                    let predictor = paeth_predictor(left, above, upper_left);

                                    current_scanline[x] + predictor
                                }
                            },
                        };

                        current_scanline[x] = unfiltered_byte;
                    }

                    let scanline_iter =
                        ScanlineIterator::new(pass_width, pixel_type, current_scanline);

                    for (idx, (r, g, b, a)) in scanline_iter.enumerate() {
                        // Put rgba in output_rgba
                        let (output_x, output_y) = match pass {
                            1 => (idx * 8, y * 8),
                            2 => (idx * 8 + 4, y * 8),
                            3 => (idx * 4, y * 8 + 4),
                            4 => (idx * 4 + 2, y * 4),
                            5 => (idx * 2, y * 4 + 2),
                            6 => (idx * 2 + 1, y * 2),
                            7 => (idx, y * 2 + 1),
                            _ => (0, 0),
                        };

                        let output_idx =
                            (output_y as usize * header.width as usize * 4) + (output_x * 4);
                        output_rgba[output_idx] = r;
                        output_rgba[output_idx + 1] = g;
                        output_rgba[output_idx + 2] = b;
                        output_rgba[output_idx + 3] = a;
                    }

                    last_scanline.copy_from_slice(current_scanline);

                    cursor += bytes_per_scanline as usize;
                }
            }
        },
    }

    Ok(())
}

fn paeth_predictor(a: u8, b: u8, c: u8) -> u8 {
    // TODO(bschwind) - Accept i16 or convert once and store in a temp.
    // a = left pixel
    // b = above pixel
    // c = upper left
    let p = a as i16 + b as i16 - c as i16;
    let pa = (p as i16 - a as i16).abs();
    let pb = (p as i16 - b as i16).abs();
    let pc = (p as i16 - c as i16).abs();

    if pa <= pb && pa <= pc {
        a
    } else if pb <= pc {
        b
    } else {
        c
    }
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

    // For now, output data is always RGBA, 1 byte per channel.
    let mut output_rgba = vec![0u8; header.width as usize * header.height as usize * 4];

    // Defilter bytes
    defilter(&header, &mut scanline_data, &mut output_rgba)?;

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

        let test_png_bytes = include_bytes!("../test_pngs/png_suite/oi4n2c16.png");
        let result = decode(test_png_bytes);

        let test_png_bytes = include_bytes!("../test_pngs/1x1.png");
        let result = decode(test_png_bytes);

        let test_png_bytes = include_bytes!("../test_pngs/3x3.png");
        let result = decode(test_png_bytes);

        let test_png_bytes = include_bytes!("../test_pngs/paeth.png");
        let result = decode(test_png_bytes);

        println!("Result: {:?}", result);
    }
}
