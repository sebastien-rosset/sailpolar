from datetime import timedelta
import numpy as np
import pandas as pd
from sailpolar.parser.nmea0183 import HDMSentence, RMCSentence, Segment, VHWSentence


class ROTAnalyzer:
    def __init__(self, window_size=5):  # 5-second window by default
        """Initialize ROT analyzer.
        
        Args:
            window_size: Window size in seconds (int/float) or timedelta
        """
        if isinstance(window_size, timedelta):
            self.window_size = window_size.total_seconds()
        else:
            self.window_size = float(window_size)
        
    def calculate_heading_rate(self, timestamp1, heading1, timestamp2, heading2):
        """Calculate rate of heading change handling 0/360 wraparound."""
        time_diff = (timestamp2 - timestamp1).total_seconds()
        if time_diff <= 0:
            return 0.0
            
        # Handle 0/360 wraparound
        heading_diff = heading2 - heading1
        if heading_diff > 180:
            heading_diff -= 360
        elif heading_diff < -180:
            heading_diff += 360
            
        return heading_diff / time_diff
        
    def analyze_segment(self, segment: Segment):
        """Analyze ROT patterns in a segment of NMEA data."""
        if not segment.sentences:
            print("Warning: Empty segment")
            return pd.DataFrame()  # Return empty DataFrame for empty segments
            
        # Extract heading data from relevant sentences
        heading_data = []
        
        for sentence in segment.sentences:
            timestamp = sentence.timestamp
            if not timestamp:
                continue
                
            heading = None
            source = None
            
            if isinstance(sentence, HDMSentence):
                heading = sentence.heading
                source = 'HDM'
            elif isinstance(sentence, VHWSentence):
                heading = sentence.heading_true or sentence.heading_mag
                source = 'VHW'
            elif isinstance(sentence, RMCSentence):
                heading = sentence.cog
                source = 'RMC'
                
            if heading is not None:
                heading_data.append({
                    'timestamp': timestamp,
                    'heading': heading,
                    'source': source
                })
        
        if not heading_data:
            print("Warning: No valid heading data found in segment")
            return pd.DataFrame()
        
        # Sort by timestamp
        heading_data.sort(key=lambda x: x['timestamp'])
        
        # Calculate ROT for each point
        rot_data = []
        for i in range(1, len(heading_data)):
            rot = self.calculate_heading_rate(
                heading_data[i-1]['timestamp'],
                heading_data[i-1]['heading'],
                heading_data[i]['timestamp'],
                heading_data[i]['heading']
            )
            
            rot_data.append({
                'timestamp': heading_data[i]['timestamp'],
                'heading': heading_data[i]['heading'],
                'rot': rot,
                'source': f"{heading_data[i-1]['source']}->{heading_data[i]['source']}"
            })
            
        return self.cluster_rot_patterns(rot_data)
        
    def cluster_rot_patterns(self, rot_data):
        """
        Cluster ROT data into patterns:
        1. Stable (low ROT)
        2. Small adjustments (moderate ROT with alternating sign)
        3. Deliberate turns (sustained moderate to high ROT)
        4. Rapid turns (very high ROT)
        """
        if not rot_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(rot_data)
        df.set_index('timestamp', inplace=True)
        
        # Ensure timestamps are datetime type
        df.index = pd.to_datetime(df.index)
        
        # Calculate rolling statistics using number of samples instead of time window
        # Estimate number of samples in window based on data frequency
        if len(df) > 1:
            avg_interval = (df.index[-1] - df.index[0]).total_seconds() / (len(df) - 1)
            window_samples = max(2, int(self.window_size / avg_interval))
        else:
            window_samples = 2
            
        df['rot_abs'] = abs(df['rot'])
        df['rot_rolling_mean'] = df['rot'].rolling(window=window_samples, center=True).mean()
        df['rot_rolling_std'] = df['rot'].rolling(window=window_samples, center=True).std()
        
        # Calculate sign changes using numpy directly
        def count_sign_changes_numpy(series, window_size):
            values = series.values
            result = np.zeros_like(values, dtype=float)
            
            for i in range(len(values)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(values), i + window_size // 2 + 1)
                window_vals = values[start_idx:end_idx]
                window_vals = window_vals[~np.isnan(window_vals)]  # Remove NaNs
                if len(window_vals) > 1:
                    signs = np.sign(window_vals)
                    result[i] = np.sum(signs[1:] != signs[:-1])
                else:
                    result[i] = 0
                    
            return pd.Series(result, index=series.index)
            
        df['sign_changes'] = count_sign_changes_numpy(df['rot'], window_samples)
        
        # Classify patterns
        def classify_pattern(row):
            if pd.isna(row['rot_abs']):
                return 'UNKNOWN'
            elif row['rot_abs'] < 1:  # Less than 1 degree per second
                return 'STABLE'
            elif row['rot_abs'] < 3 and row['sign_changes'] > 2:
                return 'SMALL_ADJUSTMENTS'
            elif row['rot_abs'] < 10:
                return 'DELIBERATE_TURN'
            else:
                return 'RAPID_TURN'
                
        df['pattern'] = df.apply(classify_pattern, axis=1)
        
        # Calculate durations
        df['duration'] = df.index.to_series().diff().dt.total_seconds()
        
        # Reset index to make timestamp a regular column
        df.reset_index(inplace=True)
        
        return df