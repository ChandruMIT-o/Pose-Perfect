import numpy as np
from collections import deque
from scipy.signal import savgol_filter, find_peaks
import matplotlib.pyplot as plt

class RepCounter:
    def __init__(self, window=11, poly_order=2, prominence=0.1):
        self.window = window  # Rolling window size
        self.poly_order = poly_order
        self.prominence = prominence
        self.vertical_displacements = [] # Rolling window for displacements
        self.smoothed_displacements = []
        self.rep_count = 0

    def process_frame(self, frame):
        """
        Process a new frame to update the vertical displacement series and count peaks.

        Args:
            frame (np.array): A NumPy array with 36 normalized x and y values.
        """
        if not isinstance(frame, np.ndarray) or frame.shape[0] != 36:
            raise ValueError("Each frame must be a NumPy array with 36 values.")

        # Extract the y-coordinates (last 18 values)
        y_points = frame[18:]
        displacement_sum = np.sum(np.abs(y_points)) / 18

        # Add the new displacement to the rolling window
        self.vertical_displacements.append(displacement_sum)

        # Check if we have enough data for smoothing
        if len(self.vertical_displacements) >= self.window:
            smoothed = savgol_filter(self.vertical_displacements, self.window, self.poly_order)
            self.smoothed_displacements = smoothed

            # Find peaks in the smoothed data
            peaks, _ = find_peaks(smoothed, prominence=self.prominence)

            # Update rep count
            self.rep_count = len(peaks)
    
    def get_rep_count(self):
        """Get the current repetition count."""
        return self.rep_count

    def plot_displacements(self):
        """Plot the current displacements and smoothed data."""
        plt.figure(figsize=(10, 5))
        plt.plot(self.vertical_displacements, alpha=0.6, label="Current Vertical Displacements")
        if len(self.smoothed_displacements) > 0:  # Check if smoothed_displacements is not empty
            plt.plot(self.smoothed_displacements, label="Smoothed Displacements", color='orange')
        plt.xlabel("Frame Number")
        plt.ylabel("Normalized Vertical Displacement")
        plt.title(f"Real-Time Vertical Displacement (Reps: {self.rep_count})")
        plt.legend()
        plt.grid()
        plt.show()
