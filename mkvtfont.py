#!/usr/bin/env python3
"""
VFNT Font Generator for FreeBSD VT Console
Features:
1. Integrates ASCII VFNT font and CJK TrueType/OpenType font
2. Restricts CJK character set to GB2312 standard (6763 Chinese characters + 682 symbols)
3. Splits wide CJK glyphs into two half-width glyphs compatible with VT console
4. Implements memory-efficient strict glyph deduplication with collision protection
"""
import struct
import os
import sys
import argparse
import freetype
from typing import List, Dict, Tuple, Optional, BinaryIO
from dataclasses import dataclass

# FreeBSD VT font constants (VFNT format specification)
VFNT_MAPS = 4
VFNT_MAP_NORMAL = 0      # ASCII characters + left half of CJK glyphs
VFNT_MAP_NORMAL_RIGHT = 1# Right half of CJK glyphs
VFNT_MAP_BOLD = 2        # Bold ASCII (unused in this implementation)
VFNT_MAP_BOLD_RIGHT = 3  # Bold right half of CJK glyphs (unused)

VFNT_MAXGLYPHS_PER_TABLE = 131072  # Max glyphs per mapping table
FONT_HEADER_MAGIC = b'VFNT0002'    # VFNT format magic number
FONT_HEADER_SIZE = 32              # VFNT header size in bytes
MAP_NAMES = ['NORMAL', 'NORMAL_RIGHT', 'BOLD', 'BOLD_RIGHT']  # Mapping table names

@dataclass
class GlyphEntry:
    """Glyph data storage with reference count for deduplication"""
    data: bytes          # Glyph bitmap data
    ref_count: int = 0   # Reference count for deduplication statistics

@dataclass
class MappingEntry:
    """Character to glyph mapping entry"""
    char_code: int       # Unicode codepoint
    glyph_idx: int       # Local glyph index in the mapping table
    map_idx: int         # Mapping table index (0-3)
    length: int = 1      # Continuous character count for table folding

# VFNT Font Parser (parses existing VFNT font files)
class VFNTFontParser:
    """Parser for FreeBSD VT VFNT format font files"""
    def __init__(self, filename: str):
        """Initialize parser with VFNT font file path"""
        self.filename = filename
        self.width = 0                # Font width in pixels
        self.height = 0               # Font height in pixels
        self.total_glyphs = 0         # Total glyphs in the font
        self.map_counts = [0]*4       # Mapping table entry counts
        self.glyph_data = []          # Glyph bitmap data list
        self.normal_maps = []         # Normal mapping table entries
        self.load()                   # Load font data on initialization

    def load(self) -> None:
        """Load and parse VFNT font file"""
        with open(self.filename, 'rb') as f:
            # Read and validate magic number
            magic = f.read(8)
            if magic != FONT_HEADER_MAGIC:
                raise ValueError(f"Invalid VFNT file format: {self.filename} (magic: {magic.hex()})")
            
            # Read font header fields
            self.width = struct.unpack('B', f.read(1))[0]
            self.height = struct.unpack('B', f.read(1))[0]
            f.read(2)  # Reserved bytes
            self.total_glyphs = struct.unpack('>I', f.read(4))[0]
            self.map_counts = [struct.unpack('>I', f.read(4))[0] for _ in range(4)]
            
            # Calculate glyph size and read glyph data
            glyph_size = (self.width +7)//8 * self.height
            self.glyph_data = [f.read(glyph_size) for _ in range(self.total_glyphs)]
            
            # Read normal mapping table
            self.normal_maps = self._read_mapping_table(f, self.map_counts[0])

    def _read_mapping_table(self, f: BinaryIO, count: int) -> List[Dict]:
        """Read mapping table entries from file"""
        mappings = []
        for _ in range(count):
            src = struct.unpack('>I', f.read(4))[0]    # Start codepoint
            dst = struct.unpack('>H', f.read(2))[0]    # Start glyph index
            length = struct.unpack('>H', f.read(2))[0] + 1  # Continuous length
            mappings.append({'src': src, 'dst': dst, 'len': length})
        return mappings

    def get_glyph_for_codepoint(self, codepoint: int) -> bytes:
        """Get glyph data for specific Unicode codepoint"""
        for mapping in self.normal_maps:
            if mapping['src'] <= codepoint < mapping['src'] + mapping['len']:
                glyph_idx = mapping['dst'] + (codepoint - mapping['src'])
                if 0 <= glyph_idx < len(self.glyph_data):
                    return self.glyph_data[glyph_idx]
        return None

    def get_glyph_size(self) -> Tuple[int, int, int]:
        """Return font dimensions and single glyph size in bytes"""
        glyph_size = (self.width +7)//8 * self.height
        return self.width, self.height, glyph_size

class VFNTGenerator:
    """Generates VFNT font files with integrated ASCII and GB2312 CJK characters"""
    def __init__(self):
        self.args = None                  # Command line arguments
        self.face = None                  # Freetype CJK font face
        self.ascii_vfnt = None            # Parsed ASCII VFNT font
        self.verbose = False              # Verbose output flag
        
        # Font parameters (derived from ASCII VFNT font)
        self.FONT_WIDTH = 0               # Half-width font width (VT console standard)
        self.FONT_HEIGHT = 0              # Font height (same for half/full width)
        self.HALF_GLYPH_SIZE = 0          # Half-width glyph size in bytes
        self.WIDE_GLYPH_WIDTH = 32        # Full-width CJK glyph width (32 pixels)
        self.WIDE_BYTES_PER_ROW = 4       # Full-width row bytes (32/8)
        
        # Glyph storage (4 independent tables for VFNT format)
        self.glyph_tables: List[List[GlyphEntry]] = [[] for _ in range(VFNT_MAPS)]
        # Hash map for deduplication: key=hash_value, value=glyph_idx (memory-efficient)
        self.glyph_hash: List[Dict[int, int]] = [{} for _ in range(VFNT_MAPS)]
        
        # Mapping tables (raw and folded)
        self.raw_mappings: List[List[MappingEntry]] = [[] for _ in range(VFNT_MAPS)]
        self.final_mappings: List[List[MappingEntry]] = [[] for _ in range(VFNT_MAPS)]
        
        # Statistics for verbose output
        self.stats = {
            'raw_mappings': [0]*VFNT_MAPS,    # Raw mapping entries count
            'folded_mappings': [0]*VFNT_MAPS, # Folded mapping entries count
            'glyphs_before_dedup': [0]*VFNT_MAPS,  # Glyph count before deduplication
            'glyphs_after_dedup': [0]*VFNT_MAPS,    # Glyph count after deduplication
            'hash_collisions': [0]*VFNT_MAPS        # Hash collision count for each table
        }
        self.glyph_offsets = [0]*VFNT_MAPS  # Global glyph index offsets for each table

    def parse_args(self) -> argparse.Namespace:
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(
            description='VFNT Font Generator for FreeBSD VT Console (GB2312 CJK Support)',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
========================================================================
mkvtfont.py - FreeBSD VT Console VFNT Font Generator with GB2312 Support
========================================================================
Function:
  This tool generates VFNT format font files for FreeBSD VT console,
  integrating existing ASCII VFNT fonts with GB2312 CJK characters from
  TrueType/OpenType fonts. It splits wide CJK glyphs into half-width
  compatible with VT console and applies deduplication to reduce file size.

Key Features:
  1. Supports GB2312 character set (6763 Chinese characters + 682 symbols)
  2. Splits 32x32 CJK glyphs into two 16x32 half-width glyphs
  3. Memory-efficient strict deduplication with 64-bit FNV-1a and collision protection
  4. Mapping table folding for compact font files
  5. Verbose mode for detailed statistics and debugging

Usage Examples:
  # Basic usage (generate hybrid font with strict deduplication)
  python3 mkvtfont.py -a terminus-b32.fnt -c wqy-microhei.ttc -o vt_hybrid.fnt

  # Verbose mode (show detailed statistics)
  python3 mkvtfont.py -a terminus-b32.fnt -c simsun.ttf -o vt_chinese.fnt --verbose

  # Disable deduplication (for debugging, larger file size)
  python3 mkvtfont.py -a consola-b32.fnt -c msyh.ttc -o vt_no_dedup.fnt --no-dedup

Note:
  - Input ASCII font must be in FreeBSD VFNT format (VFNT0002)
  - CJK font can be TTF/OTF format (e.g., WenQuanYi, SimSun, Microsoft YaHei)
  - Generated font is compatible with FreeBSD vidcontrol command
            """
        )
        parser.add_argument('-a', '--ascii-font', required=True, 
                            help='Input ASCII font in VFNT format (e.g., terminus-b32.fnt)')
        parser.add_argument('-c', '--chinese-font', required=True, 
                            help='Input CJK font in TTF/OTF format (e.g., wqy-microhei.ttc, simsun.ttf)')
        parser.add_argument('-o', '--output', required=True, 
                            help='Output VFNT font file path (e.g., vt_hybrid.fnt)')
        parser.add_argument('--ascii-start', type=lambda x: int(x,0), default=0x20, 
                            help='Start Unicode codepoint for ASCII (hex, default: 0x20)')
        parser.add_argument('--ascii-end', type=lambda x: int(x,0), default=0x7E, 
                            help='End Unicode codepoint for ASCII (hex, default: 0x7E)')
        parser.add_argument('--no-dedup', action='store_true', 
                            help='Disable glyph deduplication (increases file size)')
        parser.add_argument('--verbose', action='store_true', 
                            help='Enable verbose output with detailed statistics')
        return parser.parse_args()

    def validate(self) -> bool:
        """Validate input files and parameters"""
        # Check if font files exist
        if not os.path.exists(self.args.ascii_font):
            print(f"Error: ASCII font file not found - {self.args.ascii_font}", file=sys.stderr)
            return False
        if not os.path.exists(self.args.chinese_font):
            print(f"Error: CJK font file not found - {self.args.chinese_font}", file=sys.stderr)
            return False
        
        # Check ASCII range validity
        if self.args.ascii_start >= self.args.ascii_end:
            print(f"Error: Invalid ASCII range - start (0x{self.args.ascii_start:04X}) >= end (0x{self.args.ascii_end:04X})", file=sys.stderr)
            return False
        
        return True

    def load_fonts(self) -> bool:
        """Load and initialize ASCII VFNT font and CJK TrueType/OpenType font"""
        try:
            # Load and parse ASCII VFNT font
            self.ascii_vfnt = VFNTFontParser(self.args.ascii_font)
            self.FONT_WIDTH = self.ascii_vfnt.width
            self.FONT_HEIGHT = self.ascii_vfnt.height
            self.HALF_GLYPH_SIZE = (self.FONT_WIDTH + 7) // 8 * self.FONT_HEIGHT
            
            # Load CJK font with Freetype
            self.face = freetype.Face(self.args.chinese_font)
            self.face.set_pixel_sizes(self.WIDE_GLYPH_WIDTH, self.FONT_HEIGHT)
            
            # Verbose output for font parameters
            if self.verbose:
                print("\n=== Font Parameters ===")
                print(f"ASCII VFNT font: {self.args.ascii_font}")
                print(f"VT console half-width: {self.FONT_WIDTH}x{self.FONT_HEIGHT} pixels")
                print(f"Half-width glyph size: {self.HALF_GLYPH_SIZE} bytes")
                print(f"CJK font: {self.args.chinese_font}")
                print(f"CJK full-width render size: {self.WIDE_GLYPH_WIDTH}x{self.FONT_HEIGHT} pixels")
                print(f"Full-width row bytes: {self.WIDE_BYTES_PER_ROW}")
            
            return True
        except Exception as e:
            print(f"Error loading fonts: {str(e)}", file=sys.stderr)
            return False

    def fnv1a_64_hash(self, data: bytes) -> int:
        """
        64-bit FNV-1a hash function for glyph deduplication
        Significantly reduces hash collision probability compared to 12-bit version
        FNV parameters from: https://en.wikipedia.org/wiki/Fowler–Noll–Vo_hash_function
        """
        fnv_prime = 1099511628211
        fnv_offset_basis = 14695981039346656037
        hash_val = fnv_offset_basis
        
        for b in data:
            hash_val ^= b
            hash_val *= fnv_prime
            # Keep hash as 64-bit unsigned integer
            hash_val &= 0xFFFFFFFFFFFFFFFF
        
        return hash_val

    def generate_gb2312_unicode_list(self) -> List[int]:
        """
        Generate sorted Unicode codepoint list for GB2312 character set
        GB2312 specification:
        - 1-9 zones: Symbols and punctuation (682 characters)
        - 16-87 zones: Chinese characters (6763 characters)
        - Each zone contains 94 positions
        """
        gb2312_unicode = []
        
        # Process symbol zones (1-9)
        for zone in range(1, 10):
            for pos in range(1, 95):
                try:
                    # Convert GB2312 zone/position to bytes
                    gb_bytes = bytes([0xA0 + zone, 0xA0 + pos])
                    # Convert GB2312 to Unicode
                    unicode_char = gb_bytes.decode('gb2312')
                    codepoint = ord(unicode_char)
                    gb2312_unicode.append(codepoint)
                except UnicodeDecodeError:
                    continue
        
        # Process Chinese character zones (16-87)
        for zone in range(16, 88):
            for pos in range(1, 95):
                try:
                    gb_bytes = bytes([0xA0 + zone, 0xA0 + pos])
                    unicode_char = gb_bytes.decode('gb2312')
                    codepoint = ord(unicode_char)
                    gb2312_unicode.append(codepoint)
                except UnicodeDecodeError:
                    continue
        
        # Remove duplicates and sort codepoints
        unique_codepoints = sorted(list(set(gb2312_unicode)))
        
        # Verbose output for GB2312 statistics
        if self.verbose:
            print("\n=== GB2312 Character Set Statistics ===")
            print(f"Total unique Unicode codepoints: {len(unique_codepoints)}")
            print(f"First codepoint: U+{unique_codepoints[0]:04X} ({chr(unique_codepoints[0])})")
            print(f"Last codepoint: U+{unique_codepoints[-1]:04X} ({chr(unique_codepoints[-1])})")
        
        return unique_codepoints

    def add_glyph(self, data: bytes, map_idx: int) -> int:
        """
        Add glyph data to specified mapping table with memory-efficient strict deduplication
        Implements 64-bit hash + index-based data comparison to prevent incorrect merging
        Returns local glyph index in the table
        """
        self.stats['glyphs_before_dedup'][map_idx] += 1
        
        # Deduplication logic disabled
        if self.args.no_dedup:
            glyph_idx = len(self.glyph_tables[map_idx])
            if glyph_idx >= VFNT_MAXGLYPHS_PER_TABLE:
                raise ValueError(f"Mapping table {MAP_NAMES[map_idx]} exceeded maximum glyph count ({VFNT_MAXGLYPHS_PER_TABLE})")
            self.glyph_tables[map_idx].append(GlyphEntry(data=data))
            self.stats['glyphs_after_dedup'][map_idx] = len(self.glyph_tables[map_idx])
            return glyph_idx
        
        # Calculate 64-bit hash
        glyph_hash = self.fnv1a_64_hash(data)
        
        # Strict deduplication (hash + index-based data comparison)
        if glyph_hash in self.glyph_hash[map_idx]:
            stored_idx = self.glyph_hash[map_idx][glyph_hash]
            # Get stored glyph data from table via index (memory-efficient)
            stored_data = self.glyph_tables[map_idx][stored_idx].data
            # Check if glyph data is exactly the same (prevent collision)
            if data == stored_data:
                self.glyph_tables[map_idx][stored_idx].ref_count += 1
                return stored_idx
            else:
                # Hash collision detected - add as new glyph
                self.stats['hash_collisions'][map_idx] += 1
                glyph_idx = len(self.glyph_tables[map_idx])
                if glyph_idx >= VFNT_MAXGLYPHS_PER_TABLE:
                    raise ValueError(f"Mapping table {MAP_NAMES[map_idx]} exceeded maximum glyph count ({VFNT_MAXGLYPHS_PER_TABLE})")
                self.glyph_tables[map_idx].append(GlyphEntry(data=data))
                # Update hash map with new glyph index (overwrite collision entry)
                self.glyph_hash[map_idx][glyph_hash] = glyph_idx
                self.stats['glyphs_after_dedup'][map_idx] = len(self.glyph_tables[map_idx])
                return glyph_idx
        else:
            # No hash match - add new glyph
            glyph_idx = len(self.glyph_tables[map_idx])
            if glyph_idx >= VFNT_MAXGLYPHS_PER_TABLE:
                raise ValueError(f"Mapping table {MAP_NAMES[map_idx]} exceeded maximum glyph count ({VFNT_MAXGLYPHS_PER_TABLE})")
            self.glyph_tables[map_idx].append(GlyphEntry(data=data))
            self.glyph_hash[map_idx][glyph_hash] = glyph_idx
            self.stats['glyphs_after_dedup'][map_idx] = len(self.glyph_tables[map_idx])
            return glyph_idx

    def add_mapping(self, char_code: int, glyph_idx: int, map_idx: int) -> None:
        """Add character to glyph mapping entry"""
        self.raw_mappings[map_idx].append(MappingEntry(char_code, glyph_idx, map_idx))
        self.stats['raw_mappings'][map_idx] += 1

    def process_ascii(self) -> bool:
        """Process ASCII characters from input VFNT font"""
        print("\n=== Processing ASCII Characters ===")
        ascii_start = self.args.ascii_start
        ascii_end = self.args.ascii_end
        ascii_count = ascii_end - ascii_start + 1
        
        print(f"Processing ASCII range: U+{ascii_start:04X} to U+{ascii_end:04X} ({ascii_count} characters)")
        
        processed = 0
        for codepoint in range(ascii_start, ascii_end + 1):
            # Get glyph data from ASCII VFNT font
            glyph_data = self.ascii_vfnt.get_glyph_for_codepoint(codepoint) or bytes(self.HALF_GLYPH_SIZE)
            # Add glyph to normal table
            glyph_idx = self.add_glyph(glyph_data, VFNT_MAP_NORMAL)
            # Add mapping entry
            self.add_mapping(codepoint, glyph_idx, VFNT_MAP_NORMAL)
            processed += 1
        
        # Output statistics
        print(f"Successfully processed {processed} ASCII characters")
        
        if self.verbose:
            print(f"Raw mappings for NORMAL table: {self.stats['raw_mappings'][VFNT_MAP_NORMAL]}")
            print(f"Glyphs in NORMAL table (after dedup): {self.stats['glyphs_after_dedup'][VFNT_MAP_NORMAL]}")
            print(f"Hash collisions in NORMAL table: {self.stats['hash_collisions'][VFNT_MAP_NORMAL]}")
        
        return True

    def render_wide_glyph(self, codepoint: int) -> bytes:
        """
        Render full-width CJK glyph as 32x32 monochrome bitmap
        Uses Freetype with MONO target for VT console compatibility
        Returns empty bytes if glyph is not available
        """
        # Initialize empty glyph buffer
        bytes_per_row = self.WIDE_BYTES_PER_ROW
        glyph_size = bytes_per_row * self.FONT_HEIGHT
        buffer = bytearray(glyph_size)

        try:
            # Get glyph index from CJK font
            glyph_index = self.face.get_char_index(codepoint)
            if glyph_index == 0:
                return bytes(buffer)  # Glyph not found in font

            # Load glyph with monochrome rendering flags
            load_flags = (
                freetype.FT_LOAD_RENDER |
                freetype.FT_LOAD_TARGET_MONO |
                freetype.FT_LOAD_MONOCHROME |
                freetype.FT_LOAD_NO_HINTING
            )
            self.face.load_char(codepoint, load_flags)
            
            # Get glyph bitmap
            bitmap = self.face.glyph.bitmap
            if not bitmap.buffer or bitmap.width == 0 or bitmap.rows == 0:
                return bytes(buffer)

            # Calculate centering offsets
            glyph_width = bitmap.width
            glyph_height = bitmap.rows
            x_offset = (self.WIDE_GLYPH_WIDTH - glyph_width) // 2
            y_offset = (self.FONT_HEIGHT - glyph_height) // 2
            
            # Ensure offsets are within bounds
            x_offset = max(0, x_offset)
            y_offset = max(0, y_offset)

            # Copy bitmap data to buffer (MSB first for VT console)
            for y in range(glyph_height):
                if y + y_offset >= self.FONT_HEIGHT:
                    continue

                src_row = y * bitmap.pitch
                dst_row = (y + y_offset) * bytes_per_row

                for x in range(glyph_width):
                    if x + x_offset >= self.WIDE_GLYPH_WIDTH:
                        continue

                    # Get source pixel (monochrome format)
                    src_byte_idx = src_row + x // 8
                    src_bit_offset = 7 - (x % 8)

                    if src_byte_idx < len(bitmap.buffer):
                        src_byte = bitmap.buffer[src_byte_idx]
                        src_bit = (src_byte >> src_bit_offset) & 1

                        if src_bit:
                            # Set destination pixel
                            dst_x = x + x_offset
                            dst_byte_idx = dst_row + dst_x // 8
                            dst_bit_offset = 7 - (dst_x % 8)
                            buffer[dst_byte_idx] |= (1 << dst_bit_offset)

            return bytes(buffer)
        
        except Exception as e:
            if self.verbose:
                print(f"Warning: Failed to render glyph U+{codepoint:04X}: {str(e)}")
            return bytes(buffer)

    def split_wide_glyph(self, wide_glyph: bytes) -> Tuple[bytes, bytes]:
        """
        Split full-width 32x32 CJK glyph into two half-width 16x32 glyphs
        Returns (left_half, right_half) as byte arrays
        """
        half_width = self.FONT_WIDTH
        half_bytes_per_row = (half_width + 7) // 8
        left_half = bytearray()
        right_half = bytearray()

        # Split each row into left and right halves
        for row in range(self.FONT_HEIGHT):
            # Get full-width row data
            wide_row_start = row * self.WIDE_BYTES_PER_ROW
            wide_row_data = wide_glyph[wide_row_start:wide_row_start + self.WIDE_BYTES_PER_ROW]
            
            # Split into left and right halves
            left_row = wide_row_data[:half_bytes_per_row]
            right_row = wide_row_data[half_bytes_per_row:]
            
            left_half.extend(left_row)
            right_half.extend(right_row)

        return bytes(left_half), bytes(right_half)

    def process_cjk(self) -> bool:
        """Process GB2312 CJK characters and split into half-width glyphs"""
        print("\n=== Processing CJK Characters (GB2312) ===")
        
        # Generate sorted GB2312 Unicode list
        gb2312_codepoints = self.generate_gb2312_unicode_list()
        total_cjk = len(gb2312_codepoints)
        print(f"Total GB2312 characters to process: {total_cjk}")
        
        processed = 0
        valid_glyphs = 0
        
        # Process each codepoint in sorted list
        for codepoint in gb2312_codepoints:
            # Render full-width glyph
            wide_glyph = self.render_wide_glyph(codepoint)
            
            # Split into left and right half-width glyphs
            left_half, right_half = self.split_wide_glyph(wide_glyph)
            
            # Check if glyph is valid (non-empty)
            if any(left_half) or any(right_half):
                valid_glyphs += 1
            
            # Add left half to NORMAL table
            left_idx = self.add_glyph(left_half, VFNT_MAP_NORMAL)
            self.add_mapping(codepoint, left_idx, VFNT_MAP_NORMAL)
            
            # Add right half to NORMAL_RIGHT table
            right_idx = self.add_glyph(right_half, VFNT_MAP_NORMAL_RIGHT)
            self.add_mapping(codepoint, right_idx, VFNT_MAP_NORMAL_RIGHT)
            
            processed += 1
        
        # Output statistics
        print(f"Successfully processed {processed} GB2312 characters")
        print(f"Valid CJK glyphs rendered: {valid_glyphs}")
        
        if self.verbose:
            print("\n=== CJK Glyph Statistics ===")
            print(f"Raw mappings for NORMAL table (CJK left): {self.stats['raw_mappings'][VFNT_MAP_NORMAL] - (self.args.ascii_end - self.args.ascii_start + 1)}")
            print(f"Raw mappings for NORMAL_RIGHT table (CJK right): {self.stats['raw_mappings'][VFNT_MAP_NORMAL_RIGHT]}")
            print(f"Glyphs in NORMAL_RIGHT table (after dedup): {self.stats['glyphs_after_dedup'][VFNT_MAP_NORMAL_RIGHT]}")
            print(f"Glyph deduplication ratio (RIGHT table): {self.stats['glyphs_after_dedup'][VFNT_MAP_NORMAL_RIGHT]/self.stats['glyphs_before_dedup'][VFNT_MAP_NORMAL_RIGHT]:.2%}")
            print(f"Hash collisions in RIGHT table: {self.stats['hash_collisions'][VFNT_MAP_NORMAL_RIGHT]}")
        
        return True

    def fold_mappings(self) -> None:
        """
        Fold mapping tables by combining continuous character ranges
        Reduces mapping table size by merging consecutive entries with sequential glyph indices
        """
        print("\n=== Folding Mapping Tables ===")
        
        for map_idx in range(VFNT_MAPS):
            raw_entries = self.raw_mappings[map_idx]
            if not raw_entries:
                self.final_mappings[map_idx] = []
                self.stats['folded_mappings'][map_idx] = 0
                continue
            
            # Sort entries by Unicode codepoint
            sorted_entries = sorted(raw_entries, key=lambda e: e.char_code)
            folded_entries = []
            current_entry = sorted_entries[0]
            
            # Fold consecutive entries
            for entry in sorted_entries[1:]:
                # Check if current entry is continuous
                if (entry.char_code == current_entry.char_code + current_entry.length) and \
                   (entry.glyph_idx == current_entry.glyph_idx + current_entry.length):
                    current_entry.length += 1
                else:
                    folded_entries.append(current_entry)
                    current_entry = entry
            folded_entries.append(current_entry)
            
            # Update final mappings and statistics
            self.final_mappings[map_idx] = folded_entries
            self.stats['folded_mappings'][map_idx] = len(folded_entries)
            
            # Output folding statistics
            print(f"{MAP_NAMES[map_idx]} table: {len(raw_entries)} raw entries → {len(folded_entries)} folded entries")
            
            if self.verbose:
                fold_ratio = len(folded_entries)/len(raw_entries) * 100
                print(f"  Folding ratio: {fold_ratio:.2f}%")
                print(f"  Longest continuous range: {max([e.length for e in folded_entries]) if folded_entries else 0} characters")

    def calc_offsets(self) -> None:
        """
        Calculate global glyph index offsets for each mapping table
        Global index = local index + table offset
        """
        # Calculate cumulative offsets
        self.glyph_offsets[0] = 0
        self.glyph_offsets[1] = len(self.glyph_tables[0])
        self.glyph_offsets[2] = self.glyph_offsets[1] + len(self.glyph_tables[1])
        self.glyph_offsets[3] = self.glyph_offsets[2] + len(self.glyph_tables[2])
        
        # Verbose output for offsets
        if self.verbose:
            print("\n=== Glyph Table Offsets ===")
            total_glyphs = 0
            for i in range(VFNT_MAPS):
                table_glyphs = len(self.glyph_tables[i])
                total_glyphs += table_glyphs
                print(f"{MAP_NAMES[i]} table: offset = {self.glyph_offsets[i]}, glyphs = {table_glyphs}")
            print(f"Total glyphs across all tables: {total_glyphs}")
            print(f"Total hash collisions across all tables: {sum(self.stats['hash_collisions'])}")

    def write_vfnt(self) -> bool:
        """
        Write VFNT font file with integrated ASCII and CJK glyphs
        Follows FreeBSD VT VFNT format specification (VFNT0002)
        """
        print(f"\n=== Writing VFNT File ===")
        output_path = self.args.output
        
        # Calculate glyph offsets
        self.calc_offsets()
        
        try:
            with open(output_path, 'wb') as f:
                # 1. Write VFNT header
                header = bytearray(FONT_HEADER_SIZE)
                header[:8] = FONT_HEADER_MAGIC  # Magic number
                header[8] = self.FONT_WIDTH     # Font width
                header[9] = self.FONT_HEIGHT    # Font height
                header[10:12] = b'\x00\x00'     # Reserved bytes
                total_glyphs = sum(len(t) for t in self.glyph_tables)
                struct.pack_into('>I', header, 12, total_glyphs)  # Total glyphs
                # Write mapping table counts
                for i in range(VFNT_MAPS):
                    struct.pack_into('>I', header, 16 + 4*i, self.stats['folded_mappings'][i])
                
                # Write header to file
                f.write(header)
                
                # Verbose header information
                if self.verbose:
                    print("\n=== VFNT File Header ===")
                    print(f"Magic number: {header[:8].decode('ascii')}")
                    print(f"Font dimensions: {header[8]}x{header[9]} pixels")
                    print(f"Total glyphs: {struct.unpack('>I', header[12:16])[0]}")
                    print(f"Mapping table counts: {[struct.unpack('>I', header[16+4*i:20+4*i])[0] for i in range(VFNT_MAPS)]}")
                    print(f"Header size: {len(header)} bytes (expected: {FONT_HEADER_SIZE} bytes)")
                
                # 2. Write glyph data
                print(f"Writing glyph data ({total_glyphs} glyphs)")
                for map_idx in range(VFNT_MAPS):
                    glyph_table = self.glyph_tables[map_idx]
                    if not glyph_table:
                        continue
                    # Write all glyphs in the table
                    for glyph in glyph_table:
                        f.write(glyph.data)
                    if self.verbose:
                        print(f"  {MAP_NAMES[map_idx]} table: {len(glyph_table)} glyphs written")
                
                # 3. Write mapping tables
                print(f"Writing mapping tables (total {sum(self.stats['folded_mappings'])} entries)")
                for map_idx in range(VFNT_MAPS):
                    folded_entries = self.final_mappings[map_idx]
                    if not folded_entries:
                        continue
                    # Write each folded entry
                    for entry in folded_entries:
                        # Calculate global glyph index
                        global_glyph_idx = entry.glyph_idx + self.glyph_offsets[map_idx]
                        # Pack entry: source codepoint (4B), glyph index (2B), length-1 (2B)
                        packed_entry = struct.pack('>I H H', entry.char_code, global_glyph_idx, entry.length - 1)
                        f.write(packed_entry)
                    if self.verbose:
                        print(f"  {MAP_NAMES[map_idx]} table: {len(folded_entries)} folded entries written")
            
            # Output final statistics
            file_size = os.path.getsize(output_path)
            print(f"\nSuccessfully created VFNT file: {output_path}")
            print(f"File size: {file_size} bytes ({file_size/1024:.2f} KB)")
            
            if self.verbose:
                print("\n=== Final Statistics ===")
                print(f"Glyph deduplication enabled: {not self.args.no_dedup}")
                print(f"Total unique glyphs: {total_glyphs}")
                print(f"Total mapping entries (folded): {sum(self.stats['folded_mappings'])}")
                print(f"Average glyph size: {file_size/total_glyphs:.2f} bytes per glyph (including headers)")
                print(f"Total hash collisions detected: {sum(self.stats['hash_collisions'])}")
            
            return True
        
        except Exception as e:
            print(f"Error writing VFNT file: {str(e)}", file=sys.stderr)
            return False

    def run(self) -> int:
        """Main execution flow"""
        # Print program header
        print("========================================")
        print("mkvtfont.py - FreeBSD VT Font Generator")
        print("GB2312 CJK Character Set Support")
        print("========================================")
        
        # Parse command line arguments
        self.args = self.parse_args()
        self.verbose = self.args.verbose
        
        # Validate inputs
        if not self.validate():
            return 1
        
        # Load fonts
        if not self.load_fonts():
            return 1
        
        # Process ASCII characters
        if not self.process_ascii():
            return 1
        
        # Process CJK characters
        if not self.process_cjk():
            return 1
        
        # Fold mapping tables
        self.fold_mappings()
        
        # Write output file
        if not self.write_vfnt():
            return 1
        
        # Success
        print("\n=== Generation Complete ===")
        print("测试中文：VFNT 字体文件生成成功")  # 仅保留这一处中文测试信息
        return 0

if __name__ == '__main__':
    sys.exit(VFNTGenerator().run())
