import os
import pandas as pd
from datetime import timedelta
from sailpolar.parser.nmea0183 import NMEA0183Parser
from sailpolar.analysis.rot_analyzer import ROTAnalyzer

def test_rot_analyzer():
    # Set up file paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    nmea_file_path = os.path.join(current_dir, "data", "Race-AIS-Sart-10m.txt")
    
    output_dir = os.path.join(current_dir, "output")
    os.makedirs(output_dir, exist_ok=True)
    
    pickle_file_path = os.path.join(output_dir, "nmea_data.pickle")
    
    # Initialize analyzer with 5-second window
    window_size = timedelta(seconds=5)
    rot_analyzer = ROTAnalyzer(window_size)
    
    # Load or parse NMEA data
    if os.path.isfile(pickle_file_path):
        parsed_segments = NMEA0183Parser.load_nmea_data(pickle_file_path)
    else:
        parser = NMEA0183Parser()
        parsed_segments, _ = parser.parse_file(nmea_file_path)
        NMEA0183Parser.save_nmea_data(parsed_segments, pickle_file_path)
    
    # Analyze ROT for each segment
    print("Analyzing ROT patterns...")
    all_rot_data = []
    
    for i, segment in enumerate(parsed_segments):
        print(f"Processing segment {i+1}/{len(parsed_segments)}")
        rot_data = rot_analyzer.analyze_segment(segment)
        
        if not rot_data.empty:
            # Add segment identifier
            rot_data['segment'] = i
            all_rot_data.append(rot_data)
    
    if all_rot_data:
        # Combine all ROT data
        combined_rot_data = pd.concat(all_rot_data, ignore_index=True)
        
        # Export the ROT analysis to CSV
        output_file_path = os.path.join(
            output_dir, f"rot_analysis_{window_size.total_seconds()}s.csv"
        )
        combined_rot_data.to_csv(output_file_path, index=False)
        
        # Basic assertions to validate the analysis
        assert not combined_rot_data.empty, "ROT analysis should produce data"
        assert 'pattern' in combined_rot_data.columns, "ROT analysis should classify patterns"
        assert all(pd.notnull(combined_rot_data['rot'])), "ROT values should not be null"
        assert os.path.isfile(output_file_path), "Output file should be created"
        
        # Print some summary statistics
        print("\nROT Pattern Distribution:")
        pattern_counts = combined_rot_data['pattern'].value_counts()
        for pattern, count in pattern_counts.items():
            percentage = (count / len(combined_rot_data)) * 100
            print(f"{pattern}: {count} samples ({percentage:.1f}%)")
        
        print("\nROT Statistics:")
        rot_stats = combined_rot_data['rot'].describe()
        print(rot_stats)
        
        # Additional analysis of stable periods
        stable_periods = combined_rot_data[combined_rot_data['pattern'] == 'STABLE']
        if not stable_periods.empty:
            print("\nStable Period Statistics:")
            print(f"Total stable periods: {len(stable_periods)}")
            print(f"Longest stable period: {stable_periods['duration'].max() if 'duration' in stable_periods else 'N/A'} seconds")
