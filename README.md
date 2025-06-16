# Indian Frequency Allocation Chart Generator

This project parses Indian National Frequency Allocation Plan (NFAP) documents and generates structured data outputs and visualizations that closely match the US spectrum chart style.

## Features

- **CSV Export**: Generate CSV files from NFAP data for analysis
- **JSON Export**: Generate structured JSON output
- **Professional Visual Charts**: Create frequency allocation charts with 7 standard frequency rows
- **US Spectrum Chart Style**: Service names inside rectangles with smart rotation and abbreviations
- **Multiple Output Formats**: Support for various resolutions and dimensions
- **Smart Text Positioning**: Automatic font sizing and color selection for optimal readability
- **Command Line Interface**: Easy-to-use CLI for all operations

## Chart Features

✅ **Service names inside rectangles** (horizontal/vertical as needed)  
✅ **Smart text abbreviations** (like US spectrum chart)  
✅ **Dynamic font sizing** based on rectangle dimensions  
✅ **Intelligent text color** (black/white) based on background  
✅ **7 frequency bands** with independent scales  
✅ **Rectangle stacking/nesting** for overlapping allocations  
✅ **Professional color scheme** and legend  
✅ **Multiple output formats** and resolutions

## Installation

```bash
# Clone or navigate to the project directory
cd freq-band-v3

# Install dependencies using uv (recommended)
uv add pandas matplotlib pymupdf

# Or using pip
pip install pandas matplotlib pymupdf
```

## Usage

### 1. Parse NFAP Document and Export to CSV

```bash
# Parse document and export to CSV
python nfap_parser.py document.pdf --format csv -o frequencies.csv

# Parse and export both JSON and CSV
python nfap_parser.py document.pdf --format both -o output

# Parse text file and export to CSV
python nfap_parser.py frequencies.txt --format csv -o frequencies.csv
```

### 2. Generate Visual Frequency Chart (Enhanced)

```bash
# Standard chart (300 DPI) with automatic high-quality version
uv run python generate_chart_clean.py

# Ultra high-quality chart for presentations (600 DPI)
uv run python generate_chart_clean.py --dpi 600 -o presentation_chart.png

# Large format chart (36x18 inches) for posters  
uv run python generate_chart_clean.py --width 36 --height 18 -o large_format.png

# Compact chart for reports (16x10 inches)
uv run python generate_chart_clean.py --width 16 --height 10 -o compact_chart.png

# Show all available options
uv run python generate_chart_clean.py --help
```

### 3. Demo All Chart Capabilities

```bash
# Run comprehensive demo showing all features
uv run python demo_charts.py
```

### 3. CLI Options for NFAP Parser

```bash
python nfap_parser.py [input_file] [options]

Options:
  -o, --output PATH         Output file path
  -f, --format FORMAT       Output format: json, csv, or both
  --extract-only           Extract text from PDF only
  --pdf-backend BACKEND    PDF extraction backend (auto, pdfplumber, pymupdf)
  --no-metadata           Exclude metadata from CSV output
  --verbose, -v           Enable verbose output
  --install-deps          Install required dependencies
```

### Examples

1. **Basic CSV Export**:
   ```bash
   python nfap_parser.py frequencies.txt -f csv -o india_frequencies.csv
   ```

2. **Generate Visual Chart**:
   ```bash
   uv run python generate_chart_clean.py india_frequencies.csv -o spectrum_chart.png
   ```

3. **Extract Text from PDF**:
   ```bash
   python nfap_parser.py document.pdf --extract-only -o extracted_text.txt
   ```

4. **Parse and Export Both Formats**:
   ```bash
   python nfap_parser.py document.pdf --format both -o india_spectrum
   # Creates: india_spectrum.json and india_spectrum.csv
   ```

## Chart Features

The generated frequency chart includes:

- **7 Standard Frequency Rows**: 
  - 3-30 kHz
  - 30-300 kHz  
  - 300 kHz-3 MHz
  - 3-30 MHz
  - 30-300 MHz
  - 300 MHz-3 GHz
  - 3-30 GHz

- **Color-Coded Services**:
  - FIXED (Blue)
  - MOBILE (Orange)
  - BROADCASTING (Pink)
  - AMATEUR (Green)
  - RADIOLOCATION (Red)
  - RADIONAVIGATION (Purple)
  - SATELLITE (Teal)
  - And more...

- **Logarithmic Frequency Scale**: Proper representation of the spectrum
- **Service Labels**: Clear identification of allocated services
- **Professional Layout**: Similar to official spectrum charts

## Files in Project

- `nfap_parser.py`: Main parser with CSV/JSON export capabilities
- `generate_chart_clean.py`: Visual chart generator (7-row layout)
- `frequencies.csv`: Sample Indian frequency allocation data
- `frequencies.json`: JSON format of frequency data
- `frequency_chart.html`: Interactive web-based chart viewer

## Output Formats

### CSV Format
Contains columns:
- Band Name
- Band Start/End Frequency (Hz)
- Service Type
- Priority (Primary/Secondary)
- Usage Description
- Amateur Band (if applicable)

### JSON Format
Structured hierarchical data with:
- Metadata (source, country, date)
- Frequency bands with nested allocations
- Complete service information

### Visual Chart
High-resolution PNG image showing:
- 7 frequency rows covering 3 kHz to 30 GHz
- Color-coded service allocations
- Professional spectrum chart layout
- Legend and frequency labels

## Dependencies

- Python 3.12+
- pandas: Data processing
- matplotlib: Chart generation
- pymupdf: PDF processing
- pathlib: File handling

## Notes

- The chart generator creates a layout similar to the US frequency spectrum chart
- Frequency allocations are automatically organized into the 7 standard rows
- Overlapping frequency ranges are handled appropriately
- Colors are assigned based on service type for consistency

## Source Data

Based on the Indian National Frequency Allocation Plan (NFAP) 2022 from the Wireless Planning and Coordination Wing (WPC), Department of Telecommunications, India.
