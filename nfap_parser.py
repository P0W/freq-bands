#!/usr/bin/env python3
"""
NFAP Frequency Allocation Parser
Parses Indian National Frequency Allocation Plan (NFAP) documents and generates structured JSON output.
Supports both PDF and text file inputs with CSV export capability.
"""

import re
import json
import csv
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import argparse
import sys

# PDF processing imports
try:
    import PyPDF2

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import pdfplumber

    PDFPLUMBER_AVAILABLE = True
except ImportError:
    PDFPLUMBER_AVAILABLE = False

try:
    import fitz  # PyMuPDF

    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False


@dataclass
class FrequencyAllocation:
    """Represents a single frequency allocation entry"""

    start_freq: int
    end_freq: int
    service: str
    priority: str
    usage: str
    band: Optional[str] = None
    region_specific: Optional[str] = None


@dataclass
class FrequencyLevel:
    """Represents a frequency allocation level (e.g., AMATEUR RADIO, BROADCASTING)"""

    label: str
    allocations: List[FrequencyAllocation]


@dataclass
class FrequencyBand:
    """Represents a complete frequency band with multiple levels"""

    name: str
    start_freq: int
    end_freq: int
    levels: List[FrequencyLevel]


@dataclass
class FrequencyChart:
    """Complete frequency allocation chart"""

    metadata: Dict[str, Any]
    frequency_bands: List[FrequencyBand]


class PDFExtractor:
    """PDF text extraction utility with multiple backend support"""

    def __init__(self):
        self.available_backends = []
        if PDFPLUMBER_AVAILABLE:
            self.available_backends.append("pdfplumber")
        if PYMUPDF_AVAILABLE:
            self.available_backends.append("pymupdf")
        if PDF_AVAILABLE:
            self.available_backends.append("pypdf2")

    def extract_text_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber (best for tables)"""
        import pdfplumber

        text_content = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try to extract tables first
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        for row in table:
                            if row:
                                text_content.append(
                                    " | ".join([cell or "" for cell in row])
                                )

                # Extract regular text
                text = page.extract_text()
                if text:
                    text_content.append(text)

        return "\n".join(text_content)

    def extract_text_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        import fitz

        text_content = []
        doc = fitz.open(pdf_path)

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text()
            text_content.append(text)

        doc.close()
        return "\n".join(text_content)

    def extract_text_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        import PyPDF2

        text_content = []
        with open(pdf_path, "rb") as file:
            reader = PyPDF2.PdfReader(file)

            for page in reader.pages:
                text = page.extract_text()
                text_content.append(text)

        return "\n".join(text_content)

    def extract_text(self, pdf_path: str, backend: str = "auto") -> str:
        """Extract text from PDF using specified backend"""
        if not self.available_backends:
            raise ImportError(
                "No PDF processing libraries available. Please install one of:\n"
                "uv add pdfplumber  # Recommended for tables\n"
                "uv add PyMuPDF     # Fast and reliable\n"
                "uv add PyPDF2      # Basic functionality"
            )

        if backend == "auto":
            backend = self.available_backends[0]

        if backend not in self.available_backends:
            raise ValueError(
                f"Backend '{backend}' not available. Available: {self.available_backends}"
            )

        try:
            if backend == "pdfplumber":
                return self.extract_text_pdfplumber(pdf_path)
            elif backend == "pymupdf":
                return self.extract_text_pymupdf(pdf_path)
            elif backend == "pypdf2":
                return self.extract_text_pypdf2(pdf_path)
        except Exception as e:
            print(f"Error with {backend}: {e}")
            # Try fallback backends
            for fallback in self.available_backends:
                if fallback != backend:
                    try:
                        print(f"Trying fallback: {fallback}")
                        return self.extract_text(pdf_path, fallback)
                    except Exception:
                        continue
            raise e


class NFAPParser:
    """Parser for NFAP frequency allocation documents"""

    def __init__(self):
        self.pdf_extractor = PDFExtractor()
        self.frequency_patterns = {
            # Pattern for frequency ranges like "3-30 kHz", "30-300 MHz", etc.
            "freq_range": re.compile(
                r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(k?Hz|MHz|GHz)",
                re.IGNORECASE,
            ),
            # Pattern for single frequencies
            "single_freq": re.compile(
                r"(\d+(?:\.\d+)?)\s*(k?Hz|MHz|GHz)", re.IGNORECASE
            ),
            # Pattern for allocation entries
            "allocation": re.compile(
                r"^([A-Z\s\-()]+)(?:\s+(\d+(?:\.\d+)?)\s*[-–]\s*(\d+(?:\.\d+)?))?\s*$"
            ),
            # Pattern for service types
            "service_types": re.compile(
                r"(FIXED|MOBILE|BROADCASTING|AMATEUR|RADIOLOCATION|RADIONAVIGATION|SATELLITE|AERONAUTICAL|MARITIME|ISM|RADIO ASTRONOMY|SPACE|EARTH EXPLORATION|METEOROLOGICAL)",
                re.IGNORECASE,
            ),
            # Pattern for frequency allocation table headers
            "table_header": re.compile(
                r"(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+(?:\.\d+)?)\s*(k?Hz|MHz|GHz)",
                re.IGNORECASE,
            ),
            # Pattern for India column indicators
            "india_column": re.compile(r"India|IND\s+\d+", re.IGNORECASE),
        }

        # Frequency unit conversion to Hz
        self.unit_multipliers = {
            "hz": 1,
            "khz": 1000,
            "mhz": 1000000,
            "ghz": 1000000000,
        }

        # Common amateur radio bands
        self.amateur_bands = {
            "160m": (1800000, 2000000),
            "80m": (3500000, 3900000),
            "40m": (7000000, 7300000),
            "30m": (10100000, 10150000),
            "20m": (14000000, 14350000),
            "17m": (18068000, 18168000),
            "15m": (21000000, 21450000),
            "12m": (24890000, 24990000),
            "10m": (28000000, 29700000),
            "6m": (50000000, 54000000),
            "2m": (144000000, 148000000),
            "70cm": (420000000, 450000000),
            "23cm": (1240000000, 1300000000),
            "13cm": (2300000000, 2450000000),
            "9cm": (3300000000, 3500000000),
            "5cm": (5650000000, 5925000000),
            "3cm": (10000000000, 10500000000),
            "1.2cm": (24000000000, 24250000000),
            "6mm": (47000000000, 47200000000),
            "4mm": (76000000000, 81000000000),
            "2mm": (142000000000, 149000000000),
            "1mm": (241000000000, 250000000000),
        }

    def convert_frequency_to_hz(self, freq_str: str, unit: str) -> int:
        """Convert frequency string and unit to Hz"""
        freq_val = float(freq_str.replace(",", ""))
        unit_lower = unit.lower()
        return int(freq_val * self.unit_multipliers.get(unit_lower, 1))

    def parse_frequency_range(self, text: str) -> Tuple[int, int, str]:
        """Parse frequency range from text like '3-30 MHz'"""
        match = self.frequency_patterns["freq_range"].search(text)
        if match:
            start_freq = match.group(1)
            end_freq = match.group(2)
            unit = match.group(3)

            start_hz = self.convert_frequency_to_hz(start_freq, unit)
            end_hz = self.convert_frequency_to_hz(end_freq, unit)

            return start_hz, end_hz, unit

        # Try single frequency
        match = self.frequency_patterns["single_freq"].search(text)
        if match:
            freq = match.group(1)
            unit = match.group(2)
            freq_hz = self.convert_frequency_to_hz(freq, unit)
            return freq_hz, freq_hz, unit

        return 0, 0, ""

    def identify_band_name(self, start_freq: int, end_freq: int) -> str:
        """Identify the band name based on frequency range"""
        # Convert to more readable format
        if end_freq < 30000:  # Below 30 kHz
            return f"VLF - Very Low Frequency ({start_freq//1000}-{end_freq//1000} kHz)"
        elif end_freq < 300000:  # 30-300 kHz
            return f"LF - Low Frequency ({start_freq//1000}-{end_freq//1000} kHz)"
        elif end_freq < 3000000:  # 300 kHz - 3 MHz
            return f"MF - Medium Frequency ({start_freq//1000} kHz - {end_freq//1000000} MHz)"
        elif end_freq < 30000000:  # 3-30 MHz
            return (
                f"HF - High Frequency ({start_freq//1000000}-{end_freq//1000000} MHz)"
            )
        elif end_freq < 300000000:  # 30-300 MHz
            return f"VHF - Very High Frequency ({start_freq//1000000}-{end_freq//1000000} MHz)"
        elif end_freq < 3000000000:  # 300 MHz - 3 GHz
            return f"UHF - Ultra High Frequency ({start_freq//1000000} MHz - {end_freq//1000000000} GHz)"
        elif end_freq < 30000000000:  # 3-30 GHz
            return f"SHF - Super High Frequency ({start_freq//1000000000}-{end_freq//1000000000} GHz)"
        elif end_freq < 300000000000:  # 30-300 GHz
            return f"EHF - Extremely High Frequency ({start_freq//1000000000}-{end_freq//1000000000} GHz)"
        else:
            return f"THF - Tremendously High Frequency ({start_freq//1000000000}+ GHz)"

    def extract_service_info(self, service_text: str) -> Tuple[str, str, str]:
        """Extract service type, priority, and usage from service text"""
        service_text = service_text.strip()

        # Determine priority
        priority = "primary"
        if service_text.islower() or "(" in service_text:
            priority = "secondary"

        # Clean up service name
        service = re.sub(r"\s*\([^)]*\)", "", service_text)  # Remove parenthetical info
        service = re.sub(r"\s+\d+[\.\d]*\w*\s*", " ", service)  # Remove frequency refs
        service = service.upper().strip()

        # Extract usage information
        usage = ""
        if "BROADCASTING" in service:
            if "SATELLITE" in service:
                usage = "Satellite broadcasting services"
            elif "AM" in service_text or "MW" in service_text:
                usage = "AM radio broadcast band"
            elif "FM" in service_text:
                usage = "FM radio broadcast band"
            elif "TV" in service_text:
                usage = "Television broadcasting"
            else:
                usage = "Broadcasting services"
        elif "AMATEUR" in service:
            # Try to match amateur band
            for band_name, (start, end) in self.amateur_bands.items():
                if band_name in service_text.lower():
                    usage = f"{band_name} amateur band"
                    break
            else:
                usage = "Amateur radio services"
        elif "MOBILE" in service:
            if "AERONAUTICAL" in service:
                usage = "Aircraft communication"
            elif "MARITIME" in service:
                usage = "Marine communication"
            elif "SATELLITE" in service:
                usage = "Mobile satellite services"
            else:
                usage = "Mobile communication services"
        elif "FIXED" in service:
            if "SATELLITE" in service:
                usage = "Fixed satellite services"
            else:
                usage = "Point-to-point communication"
        elif "RADIONAVIGATION" in service:
            if "AERONAUTICAL" in service:
                usage = "Aircraft navigation"
            elif "MARITIME" in service:
                usage = "Marine navigation"
            elif "SATELLITE" in service:
                usage = "Satellite navigation (GPS/GNSS)"
            else:
                usage = "Navigation services"
        elif "RADIOLOCATION" in service:
            usage = "Radar and location services"
        elif "RADIO ASTRONOMY" in service:
            usage = "Radio astronomy observations"
        elif "ISM" in service:
            usage = "Industrial, Scientific and Medical applications"
        else:
            usage = service.lower().replace("_", " ").title()

        return service, priority, usage

    def preprocess_pdf_text(self, text: str) -> str:
        """Preprocess extracted PDF text to make it more parseable"""
        # Split into lines and clean
        lines = text.split("\n")
        cleaned_lines = []

        current_table = []
        in_allocation_table = False

        for line in lines:
            line = line.strip()

            # Skip empty lines and page headers/footers
            if not line or len(line) < 3:
                continue

            # Skip common PDF artifacts
            if any(
                skip in line.lower() for skip in ["page", "chapter", "section", "- -"]
            ):
                continue

            # Detect allocation tables
            if self.frequency_patterns["table_header"].search(line):
                in_allocation_table = True
                current_table = [line]
                continue

            if in_allocation_table:
                # Look for service allocations
                if any(
                    service in line.upper()
                    for service in [
                        "FIXED",
                        "MOBILE",
                        "BROADCASTING",
                        "AMATEUR",
                        "RADIOLOCATION",
                    ]
                ):
                    current_table.append(line)
                elif (
                    line.startswith("---")
                    or "Allocation to Radiocommunication Services" in line
                ):
                    # End of table
                    if current_table:
                        cleaned_lines.extend(current_table)
                        current_table = []
                    in_allocation_table = False
                else:
                    current_table.append(line)
            else:
                cleaned_lines.append(line)

        # Add any remaining table
        if current_table:
            cleaned_lines.extend(current_table)

        return "\n".join(cleaned_lines)

    def extract_india_allocations(self, table_text: str) -> List[str]:
        """Extract allocations specific to India from multi-column tables"""
        lines = table_text.split("\n")
        india_allocations = []

        for line in lines:
            # Look for India-specific allocations
            if self.frequency_patterns["india_column"].search(line):
                india_allocations.append(line)
            # Also include lines that appear to be service allocations
            elif any(
                service in line.upper()
                for service in ["FIXED", "MOBILE", "BROADCASTING", "AMATEUR"]
            ):
                india_allocations.append(line)

        return india_allocations

    def parse_allocation_table(self, table_text: str) -> List[FrequencyBand]:
        """Parse a frequency allocation table from text"""
        bands = []
        lines = table_text.split("\n")

        current_band = None
        current_level = None
        current_freq_range = None

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for frequency range headers (like "3-30 kHz")
            freq_match = self.frequency_patterns["table_header"].search(line)
            if freq_match:
                start_freq, end_freq, unit = self.parse_frequency_range(line)
                if start_freq > 0 and end_freq > 0:
                    # Save previous band if exists
                    if current_band and current_band.levels:
                        bands.append(current_band)

                    # Create new band
                    band_name = self.identify_band_name(start_freq, end_freq)
                    current_band = FrequencyBand(
                        name=band_name,
                        start_freq=start_freq,
                        end_freq=end_freq,
                        levels=[],
                    )
                    current_level = None
                    current_freq_range = (start_freq, end_freq)
                    continue

            # Check for sub-frequency ranges within a band
            if current_band:
                sub_start, sub_end, _ = self.parse_frequency_range(line)
                if sub_start > 0 and sub_end > 0 and current_freq_range:
                    # Check if this is a sub-range within current band
                    if (
                        sub_start >= current_freq_range[0]
                        and sub_end <= current_freq_range[1]
                    ):
                        current_freq_range = (sub_start, sub_end)

            # Check for service level headers or direct service allocations
            if current_band:
                service_match = self.frequency_patterns["service_types"].search(line)
                if service_match or line.isupper():
                    # Determine if this is a level header or direct allocation
                    if (
                        line.isupper()
                        and len(line) > 10
                        and not any(
                            word in line for word in ["FIXED", "MOBILE", "BROADCASTING"]
                        )
                    ):
                        # This is a level header
                        if current_level and current_level.allocations:
                            current_band.levels.append(current_level)

                        current_level = FrequencyLevel(
                            label=line.replace("_", " ").title(), allocations=[]
                        )
                    else:
                        # This is a direct service allocation
                        if not current_level:
                            current_level = FrequencyLevel(
                                label="ALLOCATIONS", allocations=[]
                            )

                        service, priority, usage = self.extract_service_info(line)

                        # Use current frequency range or band range
                        alloc_start = (
                            current_freq_range[0]
                            if current_freq_range
                            else current_band.start_freq
                        )
                        alloc_end = (
                            current_freq_range[1]
                            if current_freq_range
                            else current_band.end_freq
                        )

                        # Try to extract band info for amateur allocations
                        band_info = None
                        if "AMATEUR" in service:
                            for band_name, (start, end) in self.amateur_bands.items():
                                if abs(alloc_start - start) < 100000:  # Close match
                                    band_info = band_name
                                    break

                        allocation = FrequencyAllocation(
                            start_freq=alloc_start,
                            end_freq=alloc_end,
                            service=service,
                            priority=priority,
                            usage=usage,
                            band=band_info,
                        )

                        current_level.allocations.append(allocation)

        # Add final band
        if current_band:
            if current_level and current_level.allocations:
                current_band.levels.append(current_level)
            if current_band.levels:  # Only add if it has allocations
                bands.append(current_band)

        return bands

    def create_metadata(self) -> Dict[str, Any]:
        """Create metadata for the frequency chart"""
        return {
            "source": "Indian Frequency Allocation Chart",
            "country": "India",
            "agency": "WPC (Wireless Planning and Coordination Wing), DoT, TRAI",
            "lastUpdated": "2025-01-01",
            "description": "Complete Indian frequency spectrum allocations based on ITU Region 3 allocations and Indian national frequency plans",
            "references": [
                "National Frequency Allocation Plan (NFAP) 2022",
                "ITU Radio Regulations - Region 3",
                "TRAI Recommendations on Spectrum Management",
                "WPC Wireless Advisories",
            ],
        }

    def parse_document(self, document_text: str) -> FrequencyChart:
        """Parse the complete NFAP document"""
        metadata = self.create_metadata()

        # Preprocess the text if it came from PDF
        processed_text = self.preprocess_pdf_text(document_text)

        # Extract frequency allocation tables
        frequency_bands = self.parse_allocation_table(processed_text)

        return FrequencyChart(metadata=metadata, frequency_bands=frequency_bands)

    def to_json(
        self, chart: FrequencyChart, indent: int = 2, camel_case: bool = True
    ) -> str:
        """Convert FrequencyChart to JSON string"""

        def snake_to_camel(snake_str: str) -> str:
            """Convert snake_case to camelCase"""
            components = snake_str.split("_")
            return components[0] + "".join(word.capitalize() for word in components[1:])

        def convert_keys_to_camel(obj):
            """Recursively convert dictionary keys from snake_case to camelCase"""
            if isinstance(obj, dict):
                return {
                    (snake_to_camel(k) if camel_case else k): convert_keys_to_camel(v)
                    for k, v in obj.items()
                }
            elif isinstance(obj, list):
                return [convert_keys_to_camel(item) for item in obj]
            else:
                return obj

        # Convert dataclasses to dictionaries
        chart_dict = {
            "metadata": chart.metadata,
            "frequency_bands": [asdict(band) for band in chart.frequency_bands],
        }

        # Convert to camelCase if requested
        if camel_case:
            chart_dict = convert_keys_to_camel(chart_dict)

        return json.dumps(chart_dict, indent=indent, ensure_ascii=False)

    def to_csv(self, chart: FrequencyChart, include_metadata: bool = True) -> str:
        """Convert FrequencyChart to CSV format"""
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write metadata as comments if requested
        if include_metadata:
            writer.writerow(["# NFAP Frequency Allocation Data"])
            writer.writerow([f'# Source: {chart.metadata.get("source", "Unknown")}'])
            writer.writerow([f'# Country: {chart.metadata.get("country", "Unknown")}'])
            writer.writerow(
                [f'# Last Updated: {chart.metadata.get("lastUpdated", "Unknown")}']
            )
            writer.writerow([""])  # Empty row separator

        # Write CSV header
        headers = [
            "Band Name",
            "Band Start Freq (Hz)",
            "Band End Freq (Hz)",
            "Allocation Start Freq (Hz)",
            "Allocation End Freq (Hz)",
            "Frequency Range (Readable)",
            "Service",
            "Priority",
            "Usage",
            "Amateur Band",
            "Level",
            "Region Specific",
        ]
        writer.writerow(headers)

        # Write data rows
        for band in chart.frequency_bands:
            for level in band.levels:
                for allocation in level.allocations:
                    # Create readable frequency range
                    readable_range = self._format_frequency_range(
                        allocation.start_freq, allocation.end_freq
                    )

                    row = [
                        band.name,
                        allocation.start_freq,
                        allocation.end_freq,
                        allocation.start_freq,
                        allocation.end_freq,
                        readable_range,
                        allocation.service,
                        allocation.priority,
                        allocation.usage,
                        allocation.band or "",
                        level.label,
                        allocation.region_specific or "",
                    ]
                    writer.writerow(row)

        return output.getvalue()

    def _format_frequency_range(self, start_freq: int, end_freq: int) -> str:
        """Format frequency range in human-readable format"""

        def format_freq(freq: int) -> str:
            if freq >= 1000000000:  # GHz
                return f"{freq / 1000000000:.3f} GHz".rstrip("0").rstrip(".")
            elif freq >= 1000000:  # MHz
                return f"{freq / 1000000:.3f} MHz".rstrip("0").rstrip(".")
            elif freq >= 1000:  # kHz
                return f"{freq / 1000:.3f} kHz".rstrip("0").rstrip(".")
            else:  # Hz
                return f"{freq} Hz"

        if start_freq == end_freq:
            return format_freq(start_freq)
        else:
            return f"{format_freq(start_freq)} - {format_freq(end_freq)}"

    def export_to_file(
        self,
        chart: FrequencyChart,
        output_path: str,
        format_type: str = "json",
        **kwargs,
    ) -> None:
        """Export FrequencyChart to file in specified format"""
        output_path = Path(output_path)

        if format_type.lower() == "json":
            content = self.to_json(chart, **kwargs)
        elif format_type.lower() == "csv":
            content = self.to_csv(chart, **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"Data exported to {output_path} ({format_type.upper()} format)")

    def parse_file(self, file_path: str, pdf_backend: str = "auto") -> FrequencyChart:
        """Parse NFAP document from file (supports PDF and text)"""
        file_path = Path(file_path)

        if file_path.suffix.lower() == ".pdf":
            # Extract text from PDF
            content = self.pdf_extractor.extract_text(str(file_path), pdf_backend)
        else:
            # Read as text file
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

        return self.parse_document(content)


def install_requirements(use_uv=True):
    """Helper function to install required packages"""
    required_packages = [
        "pdfplumber",  # Best for table extraction
        "PyMuPDF",  # Alternative PDF library
        "PyPDF2",  # Fallback option
    ]

    print("Installing required packages...")
    import subprocess
    import shutil

    # Check if uv is available
    uv_available = shutil.which("uv") is not None

    if use_uv and uv_available:
        print("Using uv for fast package installation...")
        install_cmd = ["uv", "add"]
    else:
        print("Using pip for package installation...")
        install_cmd = [sys.executable, "-m", "pip", "install"]

    for package in required_packages:
        try:
            subprocess.check_call(install_cmd + [package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"✗ Failed to install {package}")
            if use_uv and uv_available:
                print("Trying with pip as fallback...")
                try:
                    subprocess.check_call(
                        [sys.executable, "-m", "pip", "install", package]
                    )
                    print(f"✓ {package} installed successfully with pip")
                except subprocess.CalledProcessError:
                    print(f"✗ Failed to install {package} with pip as well")


def main():
    """Main function for command-line usage"""
    arg_parser = argparse.ArgumentParser(
        description="Parse NFAP frequency allocation documents (PDF or text) and export to JSON or CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Parse PDF and output JSON to stdout
  python nfap_parser.py document.pdf
  
  # Parse and save as JSON
  python nfap_parser.py document.pdf -o output.json
  
  # Parse and save as CSV
  python nfap_parser.py document.pdf -o output.csv --format csv
  
  # Parse and save both formats
  python nfap_parser.py document.pdf -o output --format both
  
  # Extract text only from PDF
  python nfap_parser.py document.pdf --extract-only -o extracted.txt
  
  # Install dependencies
  python nfap_parser.py --install-deps
        """,
    )
    arg_parser.add_argument(
        "input_file", nargs="?", help="Input NFAP document file (.pdf or .txt)"
    )
    arg_parser.add_argument(
        "-o",
        "--output",
        help="Output file path (extension determines format if --format not specified)",
    )
    arg_parser.add_argument(
        "-f",
        "--format",
        choices=["json", "csv", "both"],
        default="json",
        help="Output format: json, csv, or both (default: json, or auto-detect from file extension)",
    )
    arg_parser.add_argument(
        "--indent", type=int, default=2, help="JSON indentation (default: 2)"
    )
    arg_parser.add_argument(
        "--pdf-backend",
        choices=["auto", "pdfplumber", "pymupdf", "pypdf2"],
        default="auto",
        help="PDF extraction backend (default: auto)",
    )
    arg_parser.add_argument(
        "--extract-only",
        action="store_true",
        help="Only extract text from PDF without parsing",
    )
    arg_parser.add_argument(
        "--install-deps", action="store_true", help="Install required dependencies"
    )
    arg_parser.add_argument(
        "--camel-case",
        action="store_true",
        default=True,
        help="Use camelCase for JSON keys (default: True)",
    )
    arg_parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Exclude metadata from CSV output",
    )
    arg_parser.add_argument(
        "--use-pip",
        action="store_true",
        help="Use pip instead of uv for dependency installation",
    )
    arg_parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    args = arg_parser.parse_args()

    # Handle dependency installation
    if args.install_deps:
        install_requirements(use_uv=not args.use_pip)
        return

    # Check if input file is provided
    if not args.input_file:
        arg_parser.print_help()
        return

    # Check if input file exists
    if not Path(args.input_file).exists():
        print(f"Error: File {args.input_file} not found")
        sys.exit(1)

    # Initialize parser
    parser_instance = NFAPParser()

    try:
        if args.extract_only and args.input_file.lower().endswith(".pdf"):
            # Just extract text from PDF
            if args.verbose:
                print(
                    f"Extracting text from {args.input_file} using {args.pdf_backend} backend..."
                )

            extracted_text = parser_instance.pdf_extractor.extract_text(
                args.input_file, args.pdf_backend
            )
            if args.output:
                with open(args.output, "w", encoding="utf-8") as f:
                    f.write(extracted_text)
                print(f"Extracted text written to {args.output}")
            else:
                print(extracted_text)
        else:
            # Parse the document
            if args.verbose:
                print(f"Parsing {args.input_file}...")

            chart = parser_instance.parse_file(args.input_file, args.pdf_backend)

            if args.verbose:
                total_bands = len(chart.frequency_bands)
                total_allocations = sum(
                    len(level.allocations)
                    for band in chart.frequency_bands
                    for level in band.levels
                )
                print(
                    f"Parsed {total_bands} frequency bands with {total_allocations} allocations"
                )

            # Determine output format(s)
            output_formats = []
            if args.output:
                output_path = Path(args.output)
                if args.format == "both":
                    output_formats = ["json", "csv"]
                elif args.format == "json" and output_path.suffix.lower() == ".csv":
                    output_formats = ["csv"]
                elif args.format == "csv" and output_path.suffix.lower() == ".json":
                    output_formats = ["json"]
                elif args.format in ["json", "csv"]:
                    output_formats = [args.format]
                else:
                    # Auto-detect from extension
                    if output_path.suffix.lower() == ".csv":
                        output_formats = ["csv"]
                    else:
                        output_formats = ["json"]
            else:
                # Output to stdout - use specified format or default to JSON
                output_formats = [args.format] if args.format != "both" else ["json"]

            # Generate and output data
            for output_format in output_formats:
                if output_format == "json":
                    use_camel_case = args.camel_case
                    json_output = parser_instance.to_json(
                        chart, indent=args.indent, camel_case=use_camel_case
                    )

                    if args.output:
                        if args.format == "both":
                            json_path = output_path.with_suffix(".json")
                        else:
                            json_path = output_path

                        with open(json_path, "w", encoding="utf-8") as f:
                            f.write(json_output)
                        print(f"JSON output written to {json_path}")
                    else:
                        print(json_output)

                elif output_format == "csv":
                    include_metadata = not args.no_metadata
                    csv_output = parser_instance.to_csv(
                        chart, include_metadata=include_metadata
                    )

                    if args.output:
                        if args.format == "both":
                            csv_path = output_path.with_suffix(".csv")
                        else:
                            csv_path = output_path

                        with open(csv_path, "w", encoding="utf-8", newline="") as f:
                            f.write(csv_output)
                        print(f"CSV output written to {csv_path}")
                    else:
                        print(csv_output)

    except ImportError as e:
        print(f"Missing required libraries: {e}")
        print("\nTo install required PDF libraries with uv:")
        print("uv add pdfplumber PyMuPDF PyPDF2")
        print("\nOr with pip:")
        print("pip install pdfplumber PyMuPDF PyPDF2")
        print("\nOr run: python nfap_parser.py --install-deps")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing document: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
