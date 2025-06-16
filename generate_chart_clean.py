#!/usr/bin/env python3
"""
India Frequency Allocation Chart Generator
Creates a visual frequency allocation chart with 7 frequency rows 
similar to the US spectrum chart
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import defaultdict
import colorsys
import argparse
from io import StringIO
import random


def get_random_color():
    """Generate a random pastel color."""
    r = lambda: random.randint(100, 255)
    return f"#{r():02x}{r():02x}{r():02x}"


class FrequencyChartGenerator:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.data = None
        self.bands = []
        # Color scheme for different services (matching the US chart style)
        self.service_colors = {
            # Primary colors from US spectrum chart - more vibrant
            "FIXED": "#1E8449",  # Vibrant Green
            "MOBILE": "#FF9800",  # Vibrant Orange
            "MARITIME MOBILE": "#00ACC1",  # Bright Teal
            "AERONAUTICAL MOBILE": "#2196F3",  # Vibrant Blue
            "BROADCASTING": "#E91E63",  # Hot Pink
            "AMATEUR": "#8BC34A",  # Lime Green
            "AMATEUR SATELLITE": "#81C784",  # Light Green variant
            "RADIOLOCATION": "#FF5722",  # Deep Orange
            "RADIONAVIGATION": "#673AB7",  # Deep Purple
            "AERONAUTICAL RADIONAVIGATION": "#3F51B5",  # Indigo
            "MARITIME RADIONAVIGATION": "#009688",  # Teal
            "SATELLITE": "#00BCD4",  # Cyan
            "MOBILE SATELLITE": "#FFC107",  # Amber
            "MOBILE-SATELLITE": "#FFC107",  # Amber
            "AERONAUTICAL": "#42A5F5",  # Blue variant
            "MARITIME": "#00ACC1",  # Cyan variant
            "ISM": "#FFEB3B",  # Yellow
            "RADIO ASTRONOMY": "#795548",  # Brown
            "SPACE": "#5C6BC0",  # Bright Indigo
            "SPACE RESEARCH": "#9C27B0",  # Vibrant Purple
            "SPACE OPERATION": "#7B1FA2",  # Deep Purple variant
            "EARTH EXPLORATION": "#607D8B",  # Blue Grey
            "EARTH EXPLORATION SATELLITE": "#607D8B",  # Blue Grey
            "EARTH EXPLORATION-SATELLITE": "#607D8B",  # Blue Grey
            "METEOROLOGICAL": "#9575CD",  # Deep Purple variant
            "METEOROLOGICAL SATELLITE": "#7E57C2",  # Deep Purple lighter
            "METEOROLOGICAL-SATELLITE": "#7E57C2",  # Deep Purple lighter
            "STANDARD FREQUENCY": "#FF8A65",  # Deep Orange lighter
            "TIME SIGNAL": "#FFAB91",  # Deep Orange lightest
            "INTER-SATELLITE": "#CE93D8",  # Purple lighter
            "RADIO DETERMINATION": "#FB8C00",  # Orange darker
            "RADIO DETERMINATION SATELLITE": "#F57C00",  # Orange darkest
            "RADIODETERMINATION-SATELLITE": "#F57C00",  # Orange darkest
            "FIXED-SATELLITE": "#388E3C",  # Green darker
            "BROADCASTING-SATELLITE": "#D81B60",  # Pink darker
            "LAND MOBILE": "#F57F17",  # Dark amber
            "LAND MOBILE-SATELLITE": "#EF6C00",  # Dark orange
            "RADIO ASTRONOMY": "#4A148C",  # Dark purple
            "RADIODETERMINATION": "#D84315",  # Deep orange variant
            "NOT ALLOCATED": "#EEEEEE",  # Very Light Grey
            "UNKNOWN": "#BDBDBD",  # Light Grey
            "DEFAULT": "#9E9E9E",  # Medium Grey
        }

    def load_data(self):
        """Load and process the CSV data"""
        print("Loading frequency data...")

        # Read the file manually to handle malformed CSV
        with open(self.csv_file, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Filter out comment lines and empty lines
        data_lines = []
        header_found = False

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not header_found and "Band Name" in line:
                # This is the header line, but might be split
                header_line = line
                header_found = True
                continue
            elif header_found:
                data_lines.append(line)

        # Create a temporary CSV content
        csv_content = header_line + "\n" + "\n".join(data_lines)

        # Use StringIO to read the CSV
        self.data = pd.read_csv(StringIO(csv_content))

        # Clean up column names
        self.data.columns = self.data.columns.str.strip()

        # Convert frequency columns to numeric
        freq_cols = [
            "Band Start Freq (Hz)",
            "Band End Freq (Hz)",
            "Allocation Start Freq (Hz)",
            "Allocation End Freq (Hz)",
        ]
        for col in freq_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors="coerce")

        # Remove rows with invalid frequency data
        valid_cols = [col for col in freq_cols if col in self.data.columns]
        if valid_cols:
            self.data = self.data.dropna(subset=valid_cols)

        print(f"Loaded {len(self.data)} frequency allocations")
        print(f"Columns: {list(self.data.columns)}")

    def organize_into_frequency_rows(self):
        """Organize frequency allocations into 7 standard frequency rows"""
        print("Organizing frequencies into 7 standard rows...")

        # Define 7 frequency bands similar to US spectrum chart
        freq_rows = [
            {"name": "3-30 kHz", "start": 3000, "end": 30000, "allocations": []},
            {"name": "30-300 kHz", "start": 30000, "end": 300000, "allocations": []},
            {
                "name": "300 kHz-3 MHz",
                "start": 300000,
                "end": 3000000,
                "allocations": [],
            },
            {"name": "3-30 MHz", "start": 3000000, "end": 30000000, "allocations": []},
            {
                "name": "30-300 MHz",
                "start": 30000000,
                "end": 300000000,
                "allocations": [],
            },
            {
                "name": "300 MHz-3 GHz",
                "start": 300000000,
                "end": 3000000000,
                "allocations": [],
            },
            {
                "name": "3-30 GHz",
                "start": 3000000000,
                "end": 30000000000,
                "allocations": [],
            },
        ]

        # Process each allocation and assign to appropriate row
        for _, row in self.data.iterrows():
            start_freq = row["Band Start Freq (Hz)"]
            end_freq = row["Band End Freq (Hz)"]

            if pd.isna(start_freq) or pd.isna(end_freq):
                continue

            # Find which frequency row this allocation belongs to
            for freq_row in freq_rows:
                if (
                    (start_freq >= freq_row["start"] and start_freq < freq_row["end"])
                    or (end_freq > freq_row["start"] and end_freq <= freq_row["end"])
                    or (start_freq <= freq_row["start"] and end_freq >= freq_row["end"])
                ):

                    # Calculate the overlap with this frequency row
                    overlap_start = max(start_freq, freq_row["start"])
                    overlap_end = min(end_freq, freq_row["end"])

                    if overlap_end > overlap_start:
                        service = (
                            str(row["Service"]).strip()
                            if pd.notna(row["Service"])
                            else "UNKNOWN"
                        )
                        priority = (
                            str(row["Priority"]).strip()
                            if pd.notna(row["Priority"])
                            else "primary"
                        )
                        usage = (
                            str(row["Usage"]).strip() if pd.notna(row["Usage"]) else ""
                        )

                        freq_row["allocations"].append(
                            {
                                "start_freq": overlap_start,
                                "end_freq": overlap_end,
                                "service": service,
                                "priority": priority,
                                "usage": usage,
                                "color": self.get_service_color(service),
                            }
                        )

        # Replace the bands with organized rows
        self.bands = freq_rows

        # Sort allocations within each row by frequency
        for band in self.bands:
            band["allocations"].sort(key=lambda x: x["start_freq"])
            print(f"{band['name']}: {len(band['allocations'])} allocations")

    def process_bands(self):
        """Process and organize frequency bands into 7 standard rows"""
        print("Processing frequency bands...")

        # Use the new organization method
        self.organize_into_frequency_rows()

        print(f"Organized into {len(self.bands)} frequency rows")

    def get_service_color(self, service):
        """Get color for a service type"""
        service_upper = service.upper()

        # Check for exact matches first
        if service_upper in self.service_colors:
            color_to_return = self.service_colors[service_upper]
            # print(f"Exact match for service '{service_upper}': {color_to_return}")
            return color_to_return

        # Check for partial matches
        for key, color in self.service_colors.items():
            if key in service_upper:
                # print(f"Partial match for service '{service_upper}' with key '{key}': {color}")
                return color

        # Log unmatched service names for debugging
        print(
            f"Unmatched service: {service_upper}"
        )  # Assign a random pastel color for unmatched services
        random_color = get_random_color()
        print(
            f"Assigned random color for unmatched service '{service_upper}': {random_color}"
        )
        return random_color

    def create_chart(
        self, output_file="indian_frequency_chart.png", dpi=300, width=24, height=14
    ):
        """Create the frequency allocation chart with US chart style"""
        print("Creating frequency chart with US spectrum chart style...")

        # US chart has landscape orientation with specific dimensions
        fig = plt.figure(figsize=(width, height))
        fig.patch.set_facecolor("white")
        # Create main layout - US chart has specific ratio of sidebar to chart
        # Adjusted width ratios to allow space for vertical band labels
        gs = fig.add_gridspec(1, 2, width_ratios=[1, 4.5], hspace=0, wspace=0.04)

        # Left sidebar for title and legend
        ax_sidebar = fig.add_subplot(gs[0, 0])
        ax_sidebar.set_xlim(0, 1)
        ax_sidebar.set_ylim(0, 1)
        ax_sidebar.axis("off")

        # Add title - US chart style with all caps and bold
        ax_sidebar.text(
            0.05,
            0.95,
            "INDIA",
            fontsize=36,
            fontweight="bold",
            transform=ax_sidebar.transAxes,
            va="top",
            color="black",
            family="Arial",
            style="normal",
        )  # US chart uses a sans-serif font
        ax_sidebar.text(
            0.05,
            0.88,
            "FREQUENCY",
            fontsize=36,
            fontweight="bold",
            transform=ax_sidebar.transAxes,
            va="top",
            color="black",
            family="Arial",
            style="normal",
        )
        ax_sidebar.text(
            0.05,
            0.81,
            "ALLOCATIONS",
            fontsize=36,
            fontweight="bold",
            transform=ax_sidebar.transAxes,
            va="top",
            color="black",
            family="Arial",
            style="normal",
        )

        # Add subtitle - US chart style with red all caps
        ax_sidebar.text(
            0.05,
            0.74,
            "THE RADIO SPECTRUM",
            fontsize=18,
            color="red",
            fontweight="bold",
            transform=ax_sidebar.transAxes,
            va="top",
            family="Arial",
        )

        # Add legend - US chart style
        self.add_legend(ax_sidebar)

        # Add horizontal line under title - US chart style
        ax_sidebar.axhline(y=0.72, xmin=0.05, xmax=0.95, color="black", linewidth=1.5)

        # Right side for frequency chart
        ax_chart = fig.add_subplot(gs[0, 1])
        # Set up chart with no x-axis scale (each row is independent)
        ax_chart.set_xlim(-0.12, 1.08)  # Extra space for vertical frequency row labels

        # Set up 7 frequency rows with optimized spacing matching US chart
        num_rows = 7
        row_height = 0.95  # Increased row height for more detail
        row_spacing = 0.10  # Reduced spacing between rows for US-style density
        total_height = num_rows * (row_height + row_spacing) - row_spacing

        # Adjust the y-axis limits to minimize unused space - US chart style
        ax_chart.set_ylim(-0.01, total_height + 0.01)  # Ultra-minimal margins

        # Draw frequency bands - each row independent with tight spacing
        for i, band in enumerate(self.bands):
            y_pos = (num_rows - i - 1) * (row_height + row_spacing)
            self.draw_frequency_row(ax_chart, band, y_pos, row_height)

        # Add thin separator lines between bands - US chart style
        for i in range(1, num_rows):
            sep_y = i * (row_height + row_spacing) - row_spacing / 2
            ax_chart.axhline(
                y=sep_y, color="gray", linewidth=0.5, alpha=0.5, xmin=0, xmax=1
            )

        # Customize chart appearance - US chart style
        ax_chart.set_ylabel("")
        ax_chart.set_xlabel("")
        ax_chart.grid(False)

        # Remove both x and y axis ticks and labels
        ax_chart.set_xticks([])
        ax_chart.set_yticks([])

        # Add thick border - US chart style
        for spine in ax_chart.spines.values():
            spine.set_linewidth(2.5)
            spine.set_color("black")

        # Tight layout with minimal margins - US chart style
        plt.tight_layout(rect=[0, 0, 1, 1])

        # Minimal padding in the saved file
        plt.savefig(
            output_file,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.1,
            facecolor="white",
            edgecolor="black",
        )
        print(f"Chart saved as {output_file}")

    def is_rectangle_overlapping(self, rect1, rect2):
        """Check if two rectangles overlap"""
        return (
            rect1["x"] < rect2["x"] + rect2["width"]
            and rect1["x"] + rect1["width"] > rect2["x"]
            and rect1["y"] < rect2["y"] + rect2["height"]
            and rect1["y"] + rect1["height"] > rect2["y"]
        )

    def draw_frequency_row(self, ax, freq_row, y_pos, height):
        """Draw a single frequency row with allocations in US chart style"""
        row_start = 0.0
        row_end = 1.0

        # Draw row background - US chart uses clean white background
        bg_rect = patches.Rectangle(
            (row_start, y_pos),
            row_end - row_start,
            height,
            facecolor="white",
            edgecolor="black",
            linewidth=1.5,
            alpha=1.0,
        )
        ax.add_patch(bg_rect)
        # Add row label on the left edge - vertically oriented to prevent overlapping with headings
        ax.text(
            -0.08,
            y_pos + height / 2,
            freq_row["name"],
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="center",
            rotation=90,
            transform=ax.transData,
        )

        # Add start frequency at left edge - Blue like US chart
        ax.text(
            -0.02,
            y_pos + height * 0.2,
            self.format_frequency(freq_row["start"]),
            fontsize=9,
            va="center",
            ha="right",
            color="blue",
            fontweight="bold",
        )

        # Add end frequency at right edge - Blue like US chart
        ax.text(
            1.02,
            y_pos + height * 0.2,
            self.format_frequency(freq_row["end"]),
            fontsize=9,
            va="center",
            ha="left",
            color="blue",
            fontweight="bold",
        )

        # Add light gray vertical grid lines for frequency scale reference - US chart style
        num_grid_lines = 9
        for i in range(1, num_grid_lines):
            grid_x = i / num_grid_lines
            ax.axvline(
                x=grid_x,
                color="#DDDDDD",
                linewidth=0.5,
                alpha=0.7,
                ymin=y_pos / ax.get_ylim()[1],
                ymax=(y_pos + height) / ax.get_ylim()[1],
            )
            # Draw allocations if any exist
        if not freq_row["allocations"]:
            # If no allocations, show "NOT ALLOCATED" in US chart style
            ax.text(
                0.5,
                y_pos + height / 2,
                "NOT ALLOCATED",
                fontsize=12,
                ha="center",
                va="center",
                style="italic",
                color="#888888",
                fontweight="light",
            )
            return

        # Sort allocations by frequency
        allocations = sorted(freq_row["allocations"], key=lambda x: x["start_freq"])

        # DON'T consolidate overlapping services - keep all rectangles for visual complexity
        # Only consolidate text placement to avoid overlaps
        # Calculate total frequency span for this row
        band_start_freq = freq_row["start"]
        band_end_freq = freq_row["end"]
        total_freq_span = band_end_freq - band_start_freq

        # Show all individual allocations with proper spectrum layout
        layout_allocations = self.calculate_allocation_layout(
            allocations, band_start_freq, total_freq_span
        )

        # Before drawing allocations, analyze and mark potential gaps
        if layout_allocations:
            # Sort by start position to find gaps
            gap_analysis = sorted(layout_allocations, key=lambda x: x["start_norm"])
            current_end = 0.0
            gaps = []

            # Find significant gaps in the layout
            for alloc in gap_analysis:
                if (
                    alloc["start_norm"] > current_end + 0.01
                ):  # Gap larger than 1% of band
                    gaps.append((current_end, alloc["start_norm"]))
                current_end = max(current_end, alloc["end_norm"])

            # Mark significant gaps with visual indicators
            for gap_start, gap_end in gaps:
                gap_width = gap_end - gap_start
                if gap_width > 0.01:  # Only mark gaps > 1% of band width
                    # Draw a gap indicator - checkered pattern
                    gap_rect = patches.Rectangle(
                        (gap_start, y_pos),
                        gap_width,
                        height,
                        facecolor="#EEEEEE",
                        edgecolor="#CCCCCC",
                        linewidth=0.5,
                        hatch="///",
                        alpha=0.5,
                    )
                    ax.add_patch(gap_rect)

                    # Add gap label if gap is large enough
                    if gap_width > 0.03:
                        gap_start_freq = band_start_freq + (gap_start * total_freq_span)
                        gap_end_freq = band_start_freq + (gap_end * total_freq_span)
                        gap_label = f"Gap: {self.format_frequency(gap_start_freq)} - {self.format_frequency(gap_end_freq)}"

                        if gap_width > 0.1:  # Only add text for larger gaps
                            ax.text(
                                gap_start + gap_width / 2,
                                y_pos + height / 2,
                                "GAP",
                                fontsize=8,
                                ha="center",
                                va="center",
                                color="#666666",
                                rotation=0,
                                bbox=dict(facecolor="white", alpha=0.7, pad=2),
                            )

        # Track text positions to avoid overlaps
        text_positions = []

        # Sort allocations by importance for text placement priority
        # Larger allocations and primary services get priority
        sorted_for_text = sorted(
            layout_allocations,
            key=lambda x: (
                -x["width"],  # Larger first
                0 if x.get("priority", "").lower() == "primary" else 1,  # Primary first
                x["start_norm"],  # Then by position
            ),
        )

        draw_rectanges = []

        # Draw each allocation rectangle first (without text)
        for allocation in layout_allocations:
            if allocation["end_norm"] <= allocation["start_norm"]:
                continue
            # Calculate position and size
            alloc_start_pos = allocation["start_norm"]
            alloc_width = allocation["end_norm"] - allocation["start_norm"]
            # Calculate vertical position and height based on layout
            alloc_y_pos = y_pos + (allocation["y_offset"] * height)
            alloc_height = allocation["height_fraction"] * height

            ## Display the dimensions of each allocation
            print(
                f"Drawing {allocation['service']} {allocation['color']} at {alloc_start_pos:.3f}-{alloc_start_pos+alloc_width:.3f} with height {alloc_height:.3f} at y={alloc_y_pos:.3f}"
            )

            # Skip drawing extremely small allocations that would be invisible
            if alloc_width < 0.001 or alloc_height < 0.001:
                print(
                    f"Skipping tiny allocation for service '{allocation['service']}' - too small to display"
                )
                continue

            # Print for debugging, but less verbose
            if (
                "UNKNOWN" not in allocation["service"]
                and len(allocation["service"]) < 30
            ):
                print(
                    f"Drawing {allocation['service']} {allocation['color']} at {alloc_start_pos:.3f}-{alloc_start_pos+alloc_width:.3f}"
                )

            ## Skip overlapping allocations using is_rectangle_overlapping
            is_overlapping = False
            for rect in draw_rectanges:
                if self.is_rectangle_overlapping(
                    {
                        "x": alloc_start_pos,
                        "y": alloc_y_pos,
                        "width": alloc_width,
                        "height": alloc_height,
                    },
                    {
                        "x": rect.get_x(),
                        "y": rect.get_y(),
                        "width": rect.get_width(),
                        "height": rect.get_height(),
                    },
                ):
                    is_overlapping = True
                    break
            if is_overlapping:
                print(
                    f"Skipping overlapping allocation for service '{allocation['service']}'"
                )
                allocation["skip"] = True
                continue

            # Special handling for footnotes - draw with distinct pattern
            if allocation.get("is_footnote", False):
                rect = patches.Rectangle(
                    (alloc_start_pos, alloc_y_pos),
                    alloc_width,
                    alloc_height,
                    facecolor=allocation["color"],
                    edgecolor="black",
                    linewidth=0.3,
                    hatch="...",  # Dotted pattern for footnotes
                    alpha=0.8,
                )
            else:
                # Draw allocation rectangle with US chart appearance
                rect = patches.Rectangle(
                    (alloc_start_pos, alloc_y_pos),
                    alloc_width,
                    alloc_height,
                    facecolor=allocation["color"],
                    edgecolor="black",
                    linewidth=0.3,  # Thinner borders like US chart
                    alpha=1.0,  # Full opacity for vibrant colors
                )

            allocation["skip"] = False
            # Add the rectangle to the list of rectangles to draw
            draw_rectanges.append(rect)

            ax.add_patch(rect)

        # Now add text labels with priority-based placement
        text_count = 0
        max_texts_per_row = 1000  # Limit to prevent overcrowding

        for allocation in sorted_for_text:
            if text_count >= max_texts_per_row:
                print(
                    f"Skipping text for service '{allocation['service']}' - too many text labels {text_count}"
                )
                break

            if allocation["end_norm"] <= allocation["start_norm"]:
                print(
                    f"Skipping allocation for service '{allocation['service']}' - zero width"
                )
                continue

            if allocation.get("skip", False):
                print(
                    f"Skipping allocation for service '{allocation['service']}' - skip flag set"
                )
                continue

            # Calculate position and size
            alloc_start_pos = allocation["start_norm"]
            alloc_width = allocation["end_norm"] - allocation["start_norm"]
            alloc_y_pos = y_pos + (allocation["y_offset"] * height)
            alloc_height = allocation["height_fraction"] * height

            # Add service labels with US spectrum chart style - ultra-dense labeling
            text_info = self.calculate_us_style_text_placement(
                allocation,
                alloc_start_pos,
                alloc_width,
                alloc_y_pos,
                alloc_height,
                height,
                text_positions,
            )

            if text_info:
                # Use the weight from text_info (bold for larger allocations)
                ax.text(
                    text_info["x"],
                    text_info["y"],
                    text_info["text"],
                    fontsize=text_info["fontsize"],
                    ha="center",
                    va="center",
                    weight=text_info["weight"],
                    color=text_info["color"],
                    rotation=text_info["rotation"],
                    clip_on=True,
                    zorder=10,
                )  # Higher zorder to ensure text appears on top

                # Track this text position for future collision detection
                text_positions.append(
                    {
                        "x": text_info["x"],
                        "y": text_info["y"],
                        "width": text_info["text_width"],
                        "height": text_info["text_height"],
                        "rotation": text_info["rotation"],
                    }
                )
                text_count += 1

                # Track for debugging
                if text_count % 10 == 0:
                    print(
                        f"Added {text_count} text labels so far"
                    )  # Add frequency tick marks for each row
        self.add_frequency_ticks(ax, freq_row, y_pos, height)

    def consolidate_overlapping_services(self, allocations):
        """Consolidate services that share the exact same frequency range"""
        if not allocations:
            return []

        # Group allocations by exact frequency range
        freq_groups = {}
        for alloc in allocations:
            key = (alloc["start_freq"], alloc["end_freq"])
            if key not in freq_groups:
                freq_groups[key] = []
            freq_groups[key].append(alloc)

        consolidated = []
        for (start_freq, end_freq), group in freq_groups.items():
            if len(group) == 1:
                # Single service for this frequency range
                consolidated.append(group[0])
            else:
                # Multiple services - consolidate them
                # Prioritize services: primary > secondary, and by importance
                primary_services = [
                    a for a in group if a["priority"].lower() == "primary"
                ]
                secondary_services = [
                    a for a in group if a["priority"].lower() != "primary"
                ]

                # Create consolidated service name
                if primary_services:
                    # Use the most important primary service as the main one
                    main_service = self.select_primary_service(primary_services)
                    if len(primary_services) > 1:
                        # Add other primary services as a combined name
                        other_primaries = [
                            s["service"] for s in primary_services if s != main_service
                        ]
                        if other_primaries:
                            combined_name = (
                                main_service["service"]
                                + " / "
                                + " / ".join(other_primaries[:2])
                            )
                            if len(other_primaries) > 2:
                                combined_name += " +"
                        else:
                            combined_name = main_service["service"]
                    else:
                        combined_name = main_service["service"]
                else:
                    # Only secondary services
                    main_service = group[0]
                    combined_name = main_service["service"]
                    if len(group) > 1:
                        combined_name += " / " + " / ".join(
                            [s["service"] for s in group[1:2]]
                        )
                        if len(group) > 2:
                            combined_name += " +"

                # Create consolidated allocation
                consolidated_alloc = {
                    "start_freq": start_freq,
                    "end_freq": end_freq,
                    "service": combined_name,
                    "priority": main_service["priority"],
                    "usage": main_service["usage"],
                    "color": main_service["color"],
                }
                consolidated.append(consolidated_alloc)

        return consolidated

    def select_primary_service(self, services):
        """Select the most important service from a list of primary services"""
        # Priority order for service importance
        service_priority = {
            "BROADCASTING": 1,
            "AMATEUR": 2,
            "MOBILE": 3,
            "FIXED": 4,
            "RADIOLOCATION": 5,
            "RADIONAVIGATION": 6,
            "SATELLITE": 7,
            "AERONAUTICAL": 8,
            "MARITIME": 9,
            "SPACE": 10,
        }

        def get_priority(service):
            service_name = service["service"].split()[0]  # Get first word
            return service_priority.get(service_name, 99)

        return min(services, key=get_priority)

    def add_legend(self, ax):
        """Add legend to the sidebar - US chart style"""
        legend_y = 0.65

        # Service colors legend - US chart uses all caps, clear header
        ax.text(
            0.05,
            legend_y,
            "RADIO SERVICES COLOR LEGEND",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Add note about color coding - US chart style
        ax.text(
            0.05,
            legend_y - 0.02,
            "(Colors indicate service types)",
            fontsize=7,
            style="italic",
            transform=ax.transAxes,
            color="#555555",
        )

        # US chart has dense, multi-column legend - simulate with tight spacing
        y_offset = 0.022  # Tighter spacing for US chart look

        # US chart has more comprehensive legend with all major services
        legend_items = [
            ("FIXED", self.service_colors["FIXED"]),
            ("MOBILE", self.service_colors["MOBILE"]),
            ("BROADCASTING", self.service_colors["BROADCASTING"]),
            ("AMATEUR", self.service_colors["AMATEUR"]),
            ("RADIOLOCATION", self.service_colors["RADIOLOCATION"]),
            ("RADIONAVIGATION", self.service_colors["RADIONAVIGATION"]),
            ("SATELLITE", self.service_colors["SATELLITE"]),
            ("MARITIME", self.service_colors["MARITIME"]),
            ("AERONAUTICAL", self.service_colors["AERONAUTICAL"]),
            ("ISM", self.service_colors["ISM"]),
            ("RADIO ASTRONOMY", self.service_colors["RADIO ASTRONOMY"]),
            ("SPACE RESEARCH", self.service_colors["SPACE RESEARCH"]),
            ("EARTH EXPLORATION", self.service_colors["EARTH EXPLORATION"]),
            ("METEOROLOGICAL", self.service_colors["METEOROLOGICAL"]),
            ("MOBILE-SATELLITE", self.service_colors["MOBILE-SATELLITE"]),
            ("FIXED-SATELLITE", self.service_colors["FIXED-SATELLITE"]),
            ("STANDARD FREQUENCY", self.service_colors["STANDARD FREQUENCY"]),
            ("NOT ALLOCATED", self.service_colors["NOT ALLOCATED"]),
        ]

        # Split into columns for US chart style - 2 columns
        col_1_items = legend_items[:10]
        col_2_items = legend_items[10:]

        # First column
        for i, (service, color) in enumerate(col_1_items):
            y = legend_y - 0.04 - (i * y_offset)

            # Color box - US chart uses thin borders
            rect = patches.Rectangle(
                (0.05, y - 0.008),
                0.03,
                0.016,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            # Service label - US chart has smaller text
            ax.text(0.09, y, service, fontsize=7, va="center", transform=ax.transAxes)

        # Second column - offset to the right
        for i, (service, color) in enumerate(col_2_items):
            y = legend_y - 0.04 - (i * y_offset)

            # Color box in second column
            rect = patches.Rectangle(
                (0.5, y - 0.008),
                0.03,
                0.016,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                transform=ax.transAxes,
            )
            ax.add_patch(rect)

            # Service label in second column
            ax.text(0.54, y, service, fontsize=7, va="center", transform=ax.transAxes)

        # Add activity code legend - US chart style
        activity_y = 0.25
        ax.text(
            0.05,
            activity_y,
            "ACTIVITY CODE",
            fontsize=10,
            fontweight="bold",
            transform=ax.transAxes,
        )

        # Primary services - US chart style
        rect = patches.Rectangle(
            (0.05, activity_y - 0.04),
            0.03,
            0.016,
            facecolor="#3366cc",
            edgecolor="black",
            linewidth=0.5,
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            0.09,
            activity_y - 0.032,
            "PRIMARY SERVICES",
            fontsize=7,
            transform=ax.transAxes,
            va="center",
        )

        # Secondary services - US chart style uses hatching
        rect = patches.Rectangle(
            (0.05, activity_y - 0.07),
            0.03,
            0.016,
            facecolor="#99ccff",
            edgecolor="black",
            linewidth=0.5,
            hatch="///",
            transform=ax.transAxes,
        )
        ax.add_patch(rect)
        ax.text(
            0.09,
            activity_y - 0.062,
            "SECONDARY SERVICES",
            fontsize=7,
            transform=ax.transAxes,
            va="center",
        )

        # Add source info - US chart style footer
        ax.text(
            0.05,
            0.08,
            "Source: NFAP India 2022",
            fontsize=7,
            transform=ax.transAxes,
            va="bottom",
            fontweight="bold",
        )
        ax.text(
            0.05,
            0.05,
            "WPC, DoT, Government of India",
            fontsize=7,
            transform=ax.transAxes,
            va="bottom",
        )

    def format_frequency(self, freq):
        """Format frequency for display"""
        if freq >= 1e9:
            return f"{freq/1e9:.0f} GHz"
        elif freq >= 1e6:
            return f"{freq/1e6:.0f} MHz"
        elif freq >= 1e3:
            return f"{freq/1e3:.0f} kHz"
        else:
            return f"{freq:.0f} Hz"

    def calculate_us_style_text_placement(
        self,
        allocation,
        start_pos,
        width,
        y_pos,
        alloc_height,
        row_height,
        existing_texts,
    ):
        """Calculate text placement using US spectrum chart principles - ultra-dense labeling"""

        # US-style minimum thresholds - ultra-aggressive for US-style chart
        min_width = 0.0015  # Extremely small to match US density
        min_height = row_height * 0.005  # Ultra-small to match US density

        # Only skip text for truly microscopic allocations
        if width < min_width or alloc_height < min_height:
            return None

        service_name = allocation["service"].replace("_", " ").strip()

        # Handle numbers/footnotes that appear in service names
        service_parts = service_name.split()
        clean_parts = []
        for part in service_parts:
            # Keep only the main service name part, removing footnotes like 5.xxx
            if not (part.startswith("5.") or part.startswith("IND")):
                clean_parts.append(part)

        # Use the first main part if we've filtered too much
        if clean_parts:
            service_name = " ".join(clean_parts)

        # Ultra-aggressive abbreviations matching US chart style
        ultra_abbreviations = {
            "BROADCASTING": "BC",
            "BROADCASTING-SATELLITE": "BC-SAT",
            "RADIOLOCATION": "RL",
            "RADIONAVIGATION": "RN",
            "RADIONAVIGATION-SATELLITE": "RN-S",
            "AERONAUTICAL MOBILE": "AM",
            "AERONAUTICAL RADIONAVIGATION": "ARN",
            "AERONAUTICAL": "AER",
            "MARITIME MOBILE": "MM",
            "MARITIME RADIONAVIGATION": "MRN",
            "MARITIME": "MAR",
            "SATELLITE": "SAT",
            "MOBILE SATELLITE": "MS",
            "MOBILE-SATELLITE": "MS",
            "RADIO ASTRONOMY": "RA",
            "METEOROLOGICAL": "MET",
            "METEOROLOGICAL SATELLITE": "MTS",
            "METEOROLOGICAL-SATELLITE": "MTS",
            "EARTH EXPLORATION": "EES",
            "EARTH EXPLORATION SATELLITE": "EES",
            "EARTH EXPLORATION-SATELLITE": "EES",
            "SPACE RESEARCH": "SR",
            "SPACE OPERATION": "SO",
            "AMATEUR": "HAM",
            "AMATEUR SATELLITE": "HS",
            "AMATEUR-SATELLITE": "HS",
            "FIXED": "FIX",
            "FIXED-SATELLITE": "FS",
            "MOBILE": "MOB",
            "MOBILE EXCEPT AERONAUTICAL": "MOB",
            "MOBILE EXCEPT AERONAUTICAL MOBILE": "MOB",
            "STANDARD FREQUENCY": "SF",
            "TIME SIGNAL": "TS",
            "INTER-SATELLITE": "IS",
            "RADIO DETERMINATION": "RD",
            "RADIO DETERMINATION SATELLITE": "RDS",
            "RADIODETERMINATION-SATELLITE": "RDS",
            "LAND MOBILE": "LM",
            "LAND MOBILE-SATELLITE": "LMS",
            "RADIODETERMINATION": "RD",
            "UNKNOWN": "",
        }

        # Use ultra abbreviation if available
        if service_name in ultra_abbreviations:
            service_name = service_name  # ultra_abbreviations[service_name]
        # elif len(service_name.split()) > 1:
        #     # Create abbreviation for multi-word services not in our dictionary
        #     abbr = "".join([word[0] for word in service_name.split()[:3]])
        #     service_name = abbr
        #     print (f"Abbreviating service name {service_name}")

        # Match exactly the US spectrum chart text sizing - ultra-tiny abbreviations everywhere
        if width > 0.15:  # Very wide
            max_chars = int(width * 200)  # More characters for wide allocations
            fontsize = min(6.5, max(3.5, int(alloc_height / row_height * 110)))
            rotation = 0
            weight = "normal"
        elif width > 0.05:  # Medium wide
            max_chars = int(width * 150)
            fontsize = min(5.0, max(3.0, int(alloc_height / row_height * 90)))
            rotation = 0
            weight = "normal"
        elif width > 0.02:  # Narrow
            max_chars = int(alloc_height / row_height * 100)
            fontsize = min(4.0, max(2.5, int(width * 220)))
            rotation = 0  # Vertical text for narrow allocations
            weight = "normal"
        else:  # Ultra narrow - exactly like US spectrum
            max_chars = max(1, int(alloc_height / row_height * 80))
            fontsize = max(2.0, min(3.5, int(width * 300)))  # Even smaller fonts
            rotation = 90
            weight = "normal"
            # Ultra short abbreviations for very narrow spaces - critical for US-style
            if len(service_name) > 2:
                service_name = service_name[:2]

        # Final length constraint
        if len(service_name) > max_chars:
            service_name = service_name[:max_chars]

        # Text positioning - hyper-aggressive placement like US chart
        text_x = start_pos + width / 2
        text_y = y_pos + alloc_height / 2

        # Estimate text dimensions for collision detection
        if rotation == 90:
            text_width = fontsize * 0.7 / 72
            text_height = len(service_name) * fontsize * 0.5 / 72
        else:
            text_width = len(service_name) * fontsize * 0.5 / 72
            text_height = fontsize * 0.7 / 72

        # Ultra-permissive collision detection - US chart style has very dense labeling
        # Reduce buffer to almost nothing - US chart has overlapping/touching text
        collision_buffer = 0.3
        collision_detected = False

        for existing in existing_texts:
            if (
                abs(text_x - existing["x"])
                < (text_width + existing["width"]) * collision_buffer / 2
                and abs(text_y - existing["y"])
                < (text_height + existing["height"]) * collision_buffer / 2
            ):
                # Try multiple positions before giving up - US chart approach
                if rotation == 90:
                    # Try shifting vertically with more aggressive offsets
                    for offset_try in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
                        new_text_y = (
                            y_pos + alloc_height / 2 + (fontsize * offset_try / 72)
                        )
                        if (
                            new_text_y >= y_pos and new_text_y <= y_pos + alloc_height
                        ):  # More permissive bounds
                            text_y = new_text_y
                            collision_detected = False
                            break
                        collision_detected = True
                else:
                    # Try shifting horizontally with more aggressive offsets
                    for offset_try in [0.3, -0.3, 0.6, -0.6, 0.9, -0.9, 1.2, -1.2]:
                        new_text_x = (
                            start_pos + width / 2 + (fontsize * offset_try / 72)
                        )
                        if (
                            new_text_x >= start_pos and new_text_x <= start_pos + width
                        ):  # More permissive bounds
                            text_x = new_text_x
                            collision_detected = False
                            break
                        collision_detected = True

                # If we've handled this collision, we can stop checking
                if not collision_detected:
                    break  # Text color - US chart uses white text on dark colors, black on light
        color_luminance = self.get_color_luminance(allocation["color"])
        text_color = "white" if color_luminance < 0.55 else "black"

        # Final check - ultra permissive like US chart
        # US chart allows text to significantly exceed bounds
        padding_factor = 0.6  # Allow text to go outside allocation

        # US chart has extremely permissive bounds checking - text can slightly overflow
        if (
            text_x >= start_pos - width * 0.1 and text_x <= start_pos + width * 1.1
        ) and (
            text_y >= y_pos - alloc_height * 0.1
            and text_y <= y_pos + alloc_height * 1.1
        ):

            # Only skip text if it's both tiny and has a collision
            if collision_detected and width < 0.005 and alloc_height < 0.01:
                print(
                    f"Skipping tiny text for service '{service_name}' - collision detected. {rotation}"
                )
                return None

            # Display the text
            print(
                f"Adding text for service '{service_name}' at {text_x:.3f}, {text_y:.3f} with rotation {rotation}"
            )
            return {
                "x": text_x,
                "y": text_y,
                "text": service_name,
                "fontsize": fontsize,
                "color": text_color,
                "rotation": rotation,
                "text_width": text_width,
                "text_height": text_height,
                "weight": weight,  # Weight is now determined earlier based on allocation width
            }

        return None

    def get_color_luminance(self, hex_color):
        """Calculate the luminance of a hex color for text contrast decisions"""
        # Remove # if present
        hex_color = hex_color.lstrip("#")

        # Convert hex to RGB
        try:
            r = int(hex_color[0:2], 16) / 255.0
            g = int(hex_color[2:4], 16) / 255.0
            b = int(hex_color[4:6], 16) / 255.0
        except (ValueError, IndexError):
            # Default to medium luminance if color parsing fails
            return 0.5

        # Calculate relative luminance using sRGB formula
        def gamma_correct(component):
            if component <= 0.03928:
                return component / 12.92
            else:
                return pow((component + 0.055) / 1.055, 2.4)

        r_linear = gamma_correct(r)
        g_linear = gamma_correct(g)
        b_linear = gamma_correct(b)

        # Weighted sum for luminance
        luminance = 0.2126 * r_linear + 0.7152 * g_linear + 0.0722 * b_linear
        return luminance

    def format_service_name(self, service, max_length):
        """Format service names for display in spectrum chart"""
        # Common abbreviations used in spectrum charts
        abbreviations = {
            "BROADCASTING": "BCAST",
            "RADIOLOCATION": "RADLOC",
            "RADIONAVIGATION": "RADNAV",
            "AERONAUTICAL MOBILE": "AERO MOB",
            "AERONAUTICAL RADIONAVIGATION": "AERO RADNAV",
            "AERONAUTICAL": "AERO",
            "MARITIME MOBILE": "MAR MOB",
            "MARITIME RADIONAVIGATION": "MAR RADNAV",
            "MARITIME": "MAR",
            "SATELLITE": "SAT",
            "RADIO ASTRONOMY": "R.ASTRO",
            "METEOROLOGICAL": "METEO",
            "METEOROLOGICAL SATELLITE": "METEO SAT",
            "EARTH EXPLORATION": "EARTH",
            "EARTH EXPLORATION SATELLITE": "EARTH SAT",
            "SPACE RESEARCH": "SPACE",
            "SPACE OPERATION": "SPACE OP",
            "AMATEUR": "HAM",
            "AMATEUR SATELLITE": "HAM SAT",
            "AMATEUR-SATELLITE": "HAMS",
            "FIXED": "FX",
            "FIXED-SATELLITE": "FS",
            "MOBILE": "MOB",
            "MOBILE SATELLITE": "MOB SAT",
            "STANDARD FREQUENCY": "STD FREQ",
            "TIME SIGNAL": "TIME",
            "INTER-SATELLITE": "INTER-SAT",
            "RADIO DETERMINATION": "RADIODET",
            "RADIO DETERMINATION SATELLITE": "RADIODET SAT",
        }

        # Clean up the service name
        service = service.replace("_", " ").strip()

        # Use abbreviation if available
        if service in abbreviations:
            service = abbreviations[service]

        # Truncate if still too long
        if len(service) > max_length:
            service = service[: max_length - 1] + "."

        return service

    def generate_chart(
        self, output_file="indian_frequency_chart.png", dpi=300, width=24, height=12
    ):
        """Main method to generate the complete chart"""
        self.load_data()
        self.process_bands()
        self.create_chart(output_file, dpi, width, height)

    def calculate_allocation_layout(
        self, allocations, band_start_freq, total_freq_span
    ):
        """Calculate spectrum allocation layout with lane stacking for overlaps - US style."""
        if not allocations:
            return []

        print(
            f"Processing {len(allocations)} allocations for layout {band_start_freq} with span {total_freq_span}"
        )

        # 1. Normalize allocations (add start_norm, end_norm, width)
        normalized_allocs = []
        skipped_count = 0
        skipped_footnotes = 0

        # Track coverage ranges to identify potential gaps
        coverage_ranges = []

        for alloc_idx, allocation_data in enumerate(allocations):
            alloc = dict(allocation_data)

            start_norm = (alloc["start_freq"] - band_start_freq) / total_freq_span
            end_norm = (alloc["end_freq"] - band_start_freq) / total_freq_span

            alloc_start_norm = max(0.0, min(1.0, start_norm))
            alloc_end_norm = max(0.0, min(1.0, end_norm))

            width = alloc_end_norm - alloc_start_norm

            # Track all frequency ranges for gap analysis
            coverage_ranges.append((alloc_start_norm, alloc_end_norm))

            # Skip microscopic allocations to avoid wasting lanes on invisible elements
            # But still count them for gap analysis
            if width <= 1e-6:
                skipped_count += 1
                if skipped_count <= 5 or skipped_count % 20 == 0:
                    print(
                        f"Skipping microscopic allocation: {alloc['service']} (width: {width})"
                    )
                continue

            # Check for and handle footnotes - add them with minimal visual representation
            service = alloc["service"].strip()
            if (
                service.startswith("5.")
                or service.startswith("IND")
                or service.startswith("NOTIFICATION")
            ):
                skipped_footnotes += 1
                # Include footnotes with minimal height to ensure visual continuity
                if width > 0.001:  # Only add if minimally visible
                    alloc.update(
                        {
                            "id": alloc_idx,
                            "start_norm": alloc_start_norm,
                            "end_norm": alloc_end_norm,
                            "width": width,
                            "y_offset": 0.9,  # Position at bottom of band
                            "height_fraction": 0.1,  # Very small height
                            "is_footnote": True,
                        }
                    )
                    normalized_allocs.append(alloc)
                continue

            # Skip allocations that are just numbers or very short text
            if service.replace(".", "").isdigit() or len(service) <= 2:
                print(f"Skipping numeric/short allocation: {service}")
                continue

            alloc.update(
                {
                    "id": alloc_idx,
                    "start_norm": alloc_start_norm,
                    "end_norm": alloc_end_norm,
                    "width": width,
                    "y_offset": 0.0,
                    "height_fraction": 1.0,
                    "is_footnote": False,
                }
            )
            normalized_allocs.append(alloc)

        print(
            f"Kept {len(normalized_allocs)} allocations after filtering (skipped {skipped_count} tiny allocations, {skipped_footnotes} footnotes)"
        )

        # Analyze and report potential gaps
        if coverage_ranges:
            coverage_ranges.sort()
            gaps = []
            current_end = 0.0

            for start, end in coverage_ranges:
                if start > current_end + 0.01:  # Gap larger than 1% of band
                    gaps.append((current_end, start))
                current_end = max(current_end, end)

            if gaps:
                print(
                    f"POTENTIAL GAPS DETECTED: {len(gaps)} significant gaps in frequency coverage"
                )
                for gap_start, gap_end in gaps:
                    gap_pct = (gap_end - gap_start) * 100
                    if gap_pct > 0.5:  # Only report gaps > 0.5% of band width
                        print(
                            f"  Gap from {gap_start:.3f} to {gap_end:.3f} ({gap_pct:.1f}% of band width)"
                        )

        # Sort by start_norm primarily, then by end_norm
        normalized_allocs.sort(key=lambda x: (x["start_norm"], x["end_norm"]))

        # 2. Enhanced lane assignment algorithm like US chart
        # Sort by service priority to ensure important services get good lanes
        service_priority = {
            "BROADCASTING": 1,  # Highest priority
            "AMATEUR": 2,
            "MOBILE": 3,
            "FIXED": 4,
            "RADIOLOCATION": 5,
            "RADIONAVIGATION": 6,
            "SATELLITE": 7,
            "SPACE RESEARCH": 8,
            "EARTH EXPLORATION": 9,
            "METEOROLOGICAL": 10,
            "STANDARD FREQUENCY": 11,
            "RADIO ASTRONOMY": 12,  # Lowest priority
        }

        # Function to determine service priority
        def get_service_priority(alloc):
            service = alloc["service"].split()[0]  # First word
            prio = service_priority.get(service, 99)  # Default low priority

            # Prioritize primary over secondary services
            if alloc["priority"].lower() == "primary":
                prio -= 100

            # Lower priority for footnotes so they don't take main lanes
            if alloc.get("is_footnote", False):
                prio += 500

            return prio

        # Sort allocations by priority and width
        normalized_allocs.sort(
            key=lambda x: (
                get_service_priority(x),  # Primary sort by service priority
                -x["width"],  # Then by width (wider first)
                x["start_norm"],  # Then by position
            )
        )

        # Ultra-aggressive lane management for complex bands
        max_lanes = min(
            50, int(30 + len(normalized_allocs) / 20)
        )  # Scale with complexity

        # First pass - assign allocations to lanes and track overlaps
        lanes_allocations = []
        lanes_busy_until = []

        print(
            f"Starting lane assignment for {len(normalized_allocs)} allocations Max Lanes {max_lanes}"
        )

        # First pass - try to place in existing lanes
        for alloc in normalized_allocs:
            placed_in_lane = False

            # Check all existing lanes for a fit
            for lane_idx in range(len(lanes_allocations)):
                # Allow a small overlap (US chart style) - overlap tolerance varies by allocation width
                # Increased overlap tolerance to reduce gaps
                overlap_tolerance = min(
                    0.01, alloc["width"] * 0.05
                )  # More permissive for larger allocations

                if (
                    alloc["start_norm"]
                    >= lanes_busy_until[lane_idx] - overlap_tolerance
                ):
                    lanes_allocations[lane_idx].append(alloc)
                    lanes_busy_until[lane_idx] = alloc["end_norm"]
                    placed_in_lane = True
                    break

            # Create a new lane if needed and we haven't hit our max
            if not placed_in_lane and len(lanes_allocations) < max_lanes:
                lanes_allocations.append([alloc])
                lanes_busy_until.append(alloc["end_norm"])
            elif not placed_in_lane:
                # If we've hit max lanes, find the optimal lane to reuse (best fit algorithm)
                best_lane_idx = -1
                best_overlap = float("inf")

                for lane_idx, busy_until in enumerate(lanes_busy_until):
                    if busy_until <= alloc["start_norm"]:
                        # Perfect fit - no overlap
                        best_lane_idx = lane_idx
                        best_overlap = 0
                        break

                    # Calculate overlap
                    overlap = (
                        busy_until - alloc["start_norm"]
                        if busy_until > alloc["start_norm"]
                        else 0
                    )
                    if overlap < best_overlap:
                        best_overlap = overlap
                        best_lane_idx = lane_idx

                if best_lane_idx >= 0:
                    lanes_allocations[best_lane_idx].append(alloc)
                    lanes_busy_until[best_lane_idx] = alloc["end_norm"]
                else:
                    # Shouldn't happen, but just in case - put in first lane
                    lanes_allocations[0].append(alloc)
                    lanes_busy_until[0] = max(lanes_busy_until[0], alloc["end_norm"])

        # 3. Group allocations by frequency range to identify overlapping groups
        num_lanes = len(lanes_allocations)
        if num_lanes == 0:
            return []

        output_allocations = []

        # If there's only one lane in this band, all allocations use full height
        if num_lanes == 1:
            for alloc in lanes_allocations[0]:
                alloc["y_offset"] = 0.0
                alloc["height_fraction"] = 1.0  # Full height
                output_allocations.append(alloc)
            return output_allocations

        # ---- IMPROVED HEIGHT ALLOCATION ALGORITHM ----
        # For each discrete frequency segment, calculate optimal height distribution

        # Step 1: Find all unique x-positions (start and end points)
        x_positions = set()
        for lane in lanes_allocations:
            for alloc in lane:
                x_positions.add(alloc["start_norm"])
                x_positions.add(alloc["end_norm"])

        x_positions = sorted(list(x_positions))

        # Step 2: Split frequency range into segments
        segments = []
        for i in range(len(x_positions) - 1):
            start_x = x_positions[i]
            end_x = x_positions[i + 1]
            if abs(end_x - start_x) > 1e-10:  # Avoid microscopic segments
                segments.append((start_x, end_x))

        print(f"Divided band into {len(segments)} segments for height optimization")

        # Step 3: For each segment, identify which lanes have allocations in this segment
        segment_lane_maps = []

        for seg_start, seg_end in segments:
            active_lanes = []
            for lane_idx, lane in enumerate(lanes_allocations):
                for alloc in lane:
                    # Check if allocation overlaps this segment
                    if alloc["start_norm"] < seg_end and alloc["end_freq"] > seg_start:
                        active_lanes.append(lane_idx)
                        break
            segment_lane_maps.append((seg_start, seg_end, active_lanes))

        # Step 4: Calculate height fractions for each allocation
        # based on number of overlapping allocations in each segment

        # Initialize height allocation map for each allocation
        alloc_heights = {}  # {alloc_id: [(start_x, end_x, height, y_offset), ...]}

        # Process each segment
        for seg_start, seg_end, active_lanes in segment_lane_maps:
            seg_width = seg_end - seg_start

            if not active_lanes:
                continue

            num_active_lanes = len(active_lanes)
            height_per_lane = 1.0 / num_active_lanes

            # Calculate a small spacing between rectangles (smaller than before)
            spacing = (
                min(0.003, 1.0 / (num_active_lanes * 20)) if num_active_lanes > 1 else 0
            )
            # Adjust height to account for spacing
            if num_active_lanes > 1:
                total_spacing = spacing * (num_active_lanes - 1)
                adjusted_height = (1.0 - total_spacing) / num_active_lanes
            else:
                adjusted_height = 1.0

            # Allocate vertical positions for each lane in this segment
            current_y = 0.0

            for lane_idx in active_lanes:
                # Find allocations that overlap this segment in this lane
                for alloc in lanes_allocations[lane_idx]:
                    if alloc["start_norm"] < seg_end and alloc["end_norm"] > seg_start:

                        # Get the overlapping part of this allocation with this segment
                        overlap_start = max(seg_start, alloc["start_norm"])
                        overlap_end = min(seg_end, alloc["end_freq"])

                        # Store segment height info for this allocation
                        if alloc["id"] not in alloc_heights:
                            alloc_heights[alloc["id"]] = []

                        alloc_heights[alloc["id"]].append(
                            (overlap_start, overlap_end, adjusted_height, current_y)
                        )

                # Move to next vertical position
                current_y += adjusted_height + spacing

        # Step 5: Create output allocations with proper height fractions and y_offsets
        for lane in lanes_allocations:
            for alloc in lane:
                if alloc["id"] not in alloc_heights:
                    # This might happen for extremely small allocations
                    # Use default values
                    alloc["y_offset"] = 0
                    alloc["height_fraction"] = 1.0 / num_lanes
                    output_allocations.append(alloc)
                    continue

                segments = alloc_heights[alloc["id"]]

                # If allocation spans multiple segments with different heights,
                # we need to create separate visual blocks
                if len(segments) == 1:
                    # Simple case - allocation fits in one segment
                    _, _, height, y_offset = segments[0]
                    alloc["y_offset"] = y_offset
                    alloc["height_fraction"] = height
                    output_allocations.append(alloc)
                else:
                    # Allocation spans multiple segments with potentially different heights
                    # Use weighted average of heights and y_offsets
                    total_width = alloc["end_norm"] - alloc["start_norm"]
                    weighted_height = 0
                    weighted_y = 0

                    for seg_start, seg_end, height, y_offset in segments:
                        seg_width = seg_end - seg_start
                        weight = seg_width / total_width
                        weighted_height += height * weight
                        weighted_y += y_offset * weight

                    # Use weighted averages
                    alloc["y_offset"] = weighted_y
                    alloc["height_fraction"] = weighted_height
                    output_allocations.append(alloc)

        # Service-specific color variations
        for alloc in output_allocations:
            service = alloc["service"].split()[0]  # Get service type

            # US chart-style color variations
            orig_color = alloc["color"]
            if not orig_color.startswith("#"):
                orig_color = "#808080"  # Default if invalid

            # Apply color variations based on service subtypes and priority
            # This creates the visual texture in the US chart
            try:
                r_hex = orig_color[1:3]
                g_hex = orig_color[3:5]
                b_hex = orig_color[5:7]

                r = int(r_hex, 16)
                g = int(g_hex, 16)
                b = int(b_hex, 16)

                # Variation based on priority - primary services are more vibrant
                priority_boost = (
                    15 if alloc.get("priority", "").lower() == "primary" else -5
                )

                # Small random variation + priority effect
                r = max(0, min(255, r + random.randint(-12, 12) + priority_boost))
                g = max(0, min(255, g + random.randint(-12, 12) + priority_boost))
                b = max(0, min(255, b + random.randint(-12, 12) + priority_boost))

                # Slight color shift for visual interest based on position
                position_shift = int((alloc["start_norm"] * 20) % 30) - 15
                r = max(0, min(255, r + position_shift))

                # Create new color with variation
                alloc["color"] = f"#{r:02x}{g:02x}{b:02x}"
            except ValueError:
                pass  # Keep original color if parsing fails

        # Sort output allocations by y_offset for proper drawing
        output_allocations.sort(key=lambda x: x["y_offset"])

        print(f"Prepared {len(output_allocations)} allocations in {num_lanes} lanes")

        # Calculate and print lane density stats
        total_allocations = len(output_allocations)
        if total_allocations > 0 and num_lanes > 0:
            print(f"Average allocations per lane: {total_allocations/num_lanes:.1f}")

            # Count allocations with text
            textable_count = sum(1 for a in output_allocations if a["width"] >= 0.002)
            print(
                f"Allocations that can fit text (~): {textable_count} ({textable_count/total_allocations*100:.1f}%)"
            )
        return output_allocations

    def add_frequency_ticks(self, ax, freq_row, y_pos, height):
        """Add frequency tick marks to a frequency row - US chart style"""
        # Calculate appropriate tick positions
        start_freq = freq_row["start"]
        end_freq = freq_row["end"]
        freq_span = end_freq - start_freq

        # Determine number of ticks based on frequency range - US chart has many ticks
        if freq_span >= 1e9:  # GHz range
            num_ticks = 8  # More ticks for GHz range like US chart
        elif freq_span >= 100e6:  # 100+ MHz range
            num_ticks = 10  # More ticks for higher frequencies
        elif freq_span >= 10e6:  # 10+ MHz range
            num_ticks = 12  # Many ticks for medium range
        else:  # Lower frequencies
            num_ticks = 8  # Baseline ticks

        # US chart also adds some intermediate ticks at important boundaries
        # Generate major tick positions
        major_ticks = []
        for i in range(num_ticks + 1):
            tick_pos = i / num_ticks  # Normalized position (0 to 1)
            tick_freq = start_freq + (freq_span * tick_pos)
            major_ticks.append((tick_pos, tick_freq))

        # Add logarithmic intermediate ticks like US chart - at 2, 5, 20, 50, etc.
        if freq_span / start_freq > 5:  # Only for bands that span multiple factors
            intermediate_ticks = []

            # Find the order of magnitude of start frequency
            start_order = 10 ** int(np.log10(start_freq))

            # Find key frequency points (1, 2, 5) × 10^n
            current = start_order
            while current <= end_freq:
                for factor in [1, 2, 5]:
                    tick_freq = current * factor
                    if start_freq <= tick_freq <= end_freq:
                        tick_pos = (tick_freq - start_freq) / freq_span
                        intermediate_ticks.append((tick_pos, tick_freq))
                current *= 10

            # Merge and remove duplicates
            all_ticks = sorted(list(set(major_ticks + intermediate_ticks)))
        else:
            all_ticks = major_ticks

        # Filter to avoid overcrowding - US chart has very strategic tick placement
        if len(all_ticks) > 15:
            filtered_ticks = []
            for i, (pos, freq) in enumerate(all_ticks):
                # Keep endpoints and strategic ticks
                if i == 0 or i == len(all_ticks) - 1 or i % 2 == 0:
                    filtered_ticks.append((pos, freq))
            all_ticks = filtered_ticks

        # Draw ticks - at the top of the row and pointing outward
        for i, (tick_pos, tick_freq) in enumerate(all_ticks):
            is_major = i % 2 == 0 or i == 0 or i == len(all_ticks) - 1

            # Place ticks at the top and point outward
            tick_length = 0.02 if is_major else 0.01
            ax.plot(
                [tick_pos, tick_pos],
                [y_pos + height, y_pos + height + tick_length],
                color="black",
                linewidth=1.0 if is_major else 0.5,
                alpha=1.0,
            )  # Label major ticks at the top outside the band
            if is_major:
                # Place labels above the ticks, outside the band
                ax.text(
                    tick_pos,
                    y_pos + height + 0.03,
                    self.format_frequency(tick_freq),
                    fontsize=6.5,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    color="black",
                    alpha=1.0,
                    weight="normal",
                )
                ax.text(
                    tick_pos,
                    y_pos + height + 0.03,
                    self.format_frequency(tick_freq),
                    fontsize=6.5,
                    ha="center",
                    va="bottom",
                    rotation=0,
                    color="black",
                    alpha=1.0,
                    weight="normal",
                )


def main():
    parser = argparse.ArgumentParser(
        description="Generate Indian Frequency Allocation Chart with 7 frequency rows in US spectrum chart style"
    )
    parser.add_argument(
        "csv_file",
        nargs="?",
        default="frequencies.csv",
        help="Input CSV file with frequency data (default: frequencies.csv)",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="output.png",
        help="Output image file (default: output.png)",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=600,
        help="Output image resolution (default: 600 DPI)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=28,
        help="Chart width in inches (default: 28, US chart uses wide format)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=16,
        help="Chart height in inches (default: 16, optimized for 7 frequency rows)",
    )

    args = parser.parse_args()

    try:
        print(f"Generating US-style spectrum chart from {args.csv_file}...")
        generator = FrequencyChartGenerator(args.csv_file)
        generator.generate_chart(args.output, args.dpi, args.width, args.height)
        print(f"Successfully generated US-style chart: {args.output}")

    except FileNotFoundError:
        print(f"Error: Could not find file {args.csv_file}")
    except Exception as e:
        print(f"Error generating chart: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
