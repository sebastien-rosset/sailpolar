#!/usr/bin/env python3
"""
Analyze NMEA file contents to understand the data structure.
"""
import os
import sys

# Add the src directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from collections import Counter
from datetime import datetime, timedelta
from sailpolar.parser.nmea0183 import NMEA0183Parser

def analyze_file(filepath):
    """Analyze contents of NMEA file."""
    parser = NMEA0183Parser()
    print(f"\nAnalyzing {filepath}...")
    
    # Parse file
    sentences = parser.parse_file(filepath)
    print(f"Total sentences: {len(sentences)}")
    
    # Analyze sentence types
    sentence_types = Counter(s.sentence_type for s in sentences)
    print("\nSentence types found:")
    for type_name, count in sentence_types.most_common():
        print(f"{type_name}: {count} sentences")
    
    # Analyze time range
    timestamps = [s.timestamp for s in sentences if hasattr(s, 'timestamp') and s.timestamp]
    if timestamps:
        start_time = min(timestamps)
        end_time = max(timestamps)
        duration = end_time - start_time
        print(f"\nTime range: {duration}")
        print(f"From: {start_time}")
        print(f"To: {end_time}")
        
        # Analyze sampling rates
        time_diffs = [(timestamps[i+1] - timestamps[i]).total_seconds() 
                     for i in range(len(timestamps)-1)]
        if time_diffs:
            avg_rate = sum(time_diffs) / len(time_diffs)
            print(f"\nAverage sampling rate: {avg_rate:.2f} seconds")
            print(f"Min time between samples: {min(time_diffs):.2f} seconds")
            print(f"Max time between samples: {max(time_diffs):.2f} seconds")
    
    # Analyze position range if available
    lats = [s.latitude for s in sentences if hasattr(s, 'latitude') and s.latitude]
    lons = [s.longitude for s in sentences if hasattr(s, 'longitude') and s.longitude]
    if lats and lons:
        print(f"\nPosition range:")
        print(f"Latitude: {min(lats):.4f} to {max(lats):.4f}")
        print(f"Longitude: {min(lons):.4f} to {max(lons):.4f}")
    
    # Analyze speed range if available
    speeds = [s.sog for s in sentences if hasattr(s, 'sog') and s.sog]
    if speeds:
        print(f"\nSpeed over ground range:")
        print(f"Min: {min(speeds):.1f} knots")
        print(f"Max: {max(speeds):.1f} knots")
        print(f"Average: {sum(speeds)/len(speeds):.1f} knots")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: analyze_nmea.py <nmea_file>")
        sys.exit(1)
        
    filepath = sys.argv[1]
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
        
    analyze_file(filepath)