# mkvtfont-chinese
FreeBSD VT4 console font generator for chinese users
# VFNT Font Generator for FreeBSD VT Console
A tool to generate VFNT font files for FreeBSD VT console, integrating ASCII VFNT fonts and CJK TrueType/OpenType fonts with GB2312 charset support.

## Why did so
- The default font cannot display Chinese.
- Pre-made Chinese font by other users is too small for high-resolution screen.
- vtfontcvt is really nightmare.

## Features
- Integrating ASCII VFNT fonts and CJK TrueType/OpenType - ASCII chars in Chinese font is urgly and maybe not mono.
- Supports GB2312 Chinese character set (6763 characters + 682 symbols) - GBK set is too big to load by kernel.
- Splits wide CJK glyphs into two half-width glyphs compatible with VT console.
- Memory-efficient strict glyph deduplication with 64-bit FNV-1a hash to avoid incorrect merging.
- Mapping table folding to reduce font file size.

## Dependencies
- Python 3.x
- freetype-py (`sudo pkg install freetype-py`)

## Usage
```bash
# Basic usage
python3 mkvtfont.py -a terminus-b32.fnt -c wqy-microhei.ttc -o vt_hybrid.fnt

# Verbose mode
python3 mkvtfont.py -a terminus-b32.fnt -c wqy-microhei.ttc -o vt_hybrid.fnt --verbose

# Disable deduplication
python3 mkvtfont.py -a terminus-b32.fnt -c wqy-microhei.ttc -o vt_no_dedup.fnt --no-dedup
