import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import timedelta


class ROTVisualizer:
    def __init__(self, figsize=(15, 10)):
        self.figsize = figsize
        # Define color scheme for patterns
        self.pattern_colors = {
            "STABLE": "#2ecc71",
            "SMALL_ADJUSTMENTS": "#3498db",
            "DELIBERATE_TURN": "#f1c40f",
            "RAPID_TURN": "#e74c3c",
            "UNKNOWN": "#95a5a6",
        }

    def plot_time_series(self, df, save_path=None):
        """Plot heading and ROT time series with pattern classification."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, sharex=True)

        # Plot heading
        ax1.plot(df["timestamp"], df["heading"], "b-", label="Heading")
        ax1.set_ylabel("Heading (degrees)")
        ax1.set_title("Heading over Time")
        ax1.grid(True)
        ax1.legend()

        # Plot ROT with pattern coloring
        scatter = ax2.scatter(
            df["timestamp"],
            df["rot"],
            c=[self.pattern_colors[p] for p in df["pattern"]],
            alpha=0.6,
            s=20,
        )
        ax2.set_ylabel("Rate of Turn (deg/s)")
        ax2.set_title("Rate of Turn with Pattern Classification")
        ax2.grid(True)

        # Add pattern legend
        legend_elements = [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor=color,
                label=pattern,
                markersize=10,
            )
            for pattern, color in self.pattern_colors.items()
        ]
        ax2.legend(handles=legend_elements)

        plt.xlabel("Time")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_pattern_distribution(self, df, save_path=None):
        """Plot pattern distribution statistics."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.figsize)

        # Pattern count distribution
        pattern_counts = df["pattern"].value_counts()
        colors = [self.pattern_colors[p] for p in pattern_counts.index]
        pattern_counts.plot(kind="bar", ax=ax1, color=colors)
        ax1.set_title("Pattern Distribution")
        ax1.set_ylabel("Count")
        ax1.tick_params(axis="x", rotation=45)

        # ROT distribution by pattern
        sns.boxplot(
            data=df,
            x="pattern",
            y="rot",
            hue="pattern",
            palette=self.pattern_colors,
            ax=ax2,
            legend=False,
        )
        ax2.set_title("ROT Distribution by Pattern")
        ax2.set_ylabel("Rate of Turn (deg/s)")
        ax2.tick_params(axis="x", rotation=45)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_pattern_transitions(self, df, save_path=None):
        """Analyze and plot pattern transitions."""
        # Create transition matrix
        transitions = pd.crosstab(df["pattern"].shift(), df["pattern"])

        # Plot transition heatmap
        plt.figure(figsize=self.figsize)
        sns.heatmap(transitions, annot=True, cmap="YlOrRd", fmt="d")
        plt.title("Pattern Transition Matrix")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_stability_analysis(
        self, df, window_size=timedelta(minutes=5), save_path=None
    ):
        """Analyze and plot heading stability over time."""
        # Convert window_size to seconds for calculations
        window_secs = window_size.total_seconds()

        # Create a copy of the dataframe for stability analysis
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df.set_index("timestamp", inplace=True)

        # Calculate rolling stability metrics
        # Create binary columns for each pattern
        pattern_binary = pd.get_dummies(df["pattern"])

        # Resample data to regular intervals
        resample_rule = f"{int(window_secs)}S"

        # Calculate stability metrics
        stability_data = pd.DataFrame()
        stability_data["heading_std"] = df["heading"].rolling(resample_rule).std()
        stability_data["rot_std"] = df["rot"].rolling(resample_rule).std()

        # Calculate proportion of stable patterns in each window
        stable_binary = (df["pattern"] == "STABLE").astype(int)
        stability_data["stable_ratio"] = stable_binary.rolling(resample_rule).mean()

        # Plot stability metrics
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=self.figsize, sharex=True)

        # Heading stability
        ax1.plot(stability_data.index, stability_data["heading_std"])
        ax1.set_ylabel("Heading Std Dev")
        ax1.set_title(f"Heading Stability ({window_size})")
        ax1.grid(True)

        # ROT stability
        ax2.plot(stability_data.index, stability_data["rot_std"])
        ax2.set_ylabel("ROT Std Dev")
        ax2.set_title("ROT Stability")
        ax2.grid(True)

        # Stability percentage
        ax3.plot(stability_data.index, stability_data["stable_ratio"])
        ax3.set_ylabel("Stable Ratio")
        ax3.set_title("Proportion of Stable Patterns")
        ax3.grid(True)
        ax3.set_ylim(0, 1)

        plt.xlabel("Time")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

        return stability_data

    def plot_rose_diagram(self, df, save_path=None):
        """Create a rose diagram showing heading distribution with ROT patterns."""
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection="polar")

        # Convert headings to radians
        headings_rad = np.deg2rad(df["heading"])

        # Create bins for the rose diagram
        bins = np.linspace(0, 2 * np.pi, 37)  # 36 bins of 10 degrees each

        # Plot rose diagram for each pattern
        width = (2 * np.pi) / 36  # Width of each bin
        for pattern in self.pattern_colors.keys():
            mask = df["pattern"] == pattern
            if mask.any():
                heights, _ = np.histogram(headings_rad[mask], bins=bins)
                heights = heights / heights.max()  # Normalize
                ax.bar(
                    bins[:-1],
                    heights,
                    width=width,
                    bottom=0.0,
                    alpha=0.6,
                    label=pattern,
                    color=self.pattern_colors[pattern],
                )

        ax.set_theta_direction(-1)  # Clockwise
        ax.set_theta_zero_location("N")  # 0 degrees at North
        ax.legend(bbox_to_anchor=(1.2, 0.5))

        plt.title("Heading Distribution by Pattern")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
