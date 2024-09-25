import matplotlib.pyplot as plt
import numpy as np
import pickle
from icecream import ic

# Load the observations
with open('/tmp/training_dist.pkl', 'rb') as f:
    training_obs = np.array(pickle.load(f))
    ic(training_obs.shape)
feature_ranges = {
    'COM <-> AvgFoot': range(0, 1),
    'ZMP <-> AvgFoot': range(1, 2),
}


def plot_feature_distribution(test_observations, feature_name, feature_range):
    # test_observations = test_observations.clip(min=0, max=0.5)
    num_features = len(feature_range)
    cols = min(num_features, 4)  # At most 4 plots per row
    rows = (num_features + 3) // 4  # Calculate the number of rows needed, rounding up
    plt.figure(figsize=(5 * cols, 4 * rows))  # Adjust figure size based on the number of subplots
    
    for i, feature_idx in enumerate(feature_range):
        ax1 = plt.subplot(rows, cols, i + 1)
        # Plot training observations on the first axis
        # ax1.hist(training_observations[:, feature_idx], bins=50, alpha=0.5, label='Training')
        # ax1.set_xlabel('Value')
        # ax1.set_ylabel('Frequency (Training)', color='tab:blue')
        # ax1.tick_params(axis='y', labelcolor='tab:blue')
        
        # Create a second y-axis for the test observations
        ax2 = ax1.twinx()
        ax2.hist(test_observations[:, feature_idx], bins=50, alpha=0.5, color='orange', label='Test')
        ax2.set_ylabel('Frequency (Test)', color='tab:orange')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        
        # Highlight min/max values of the testing data
        min_test_val = test_observations[:, feature_idx].min()
        max_test_val = test_observations[:, feature_idx].max()
        ax2.axvline(min_test_val, color='green', linestyle='dashed', linewidth=2)
        ax2.axvline(max_test_val, color='red', linestyle='dashed', linewidth=2)
        ax2.text(min_test_val, 0.95 * ax2.get_ylim()[1], f'Min: {min_test_val:.2f}', color='green', ha='right')
        ax2.text(max_test_val, 0.95 * ax2.get_ylim()[1], f'Max: {max_test_val:.2f}', color='red', ha='left')
        
        plt.title(f'{feature_name} - Dim {feature_idx - min(feature_range) + 1}')
        # Manually adjust the position of the legend to not overlap the histograms
        # ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
    
    plt.tight_layout(pad=3.0)
    plt.show()


# Iterate over the predefined features and plot their distributions for both phases
for feature_name, feature_range in feature_ranges.items():
    plot_feature_distribution(training_obs, feature_name, feature_range)
# for feature_name, feature_range in high_level_feature_ranges.items():
#     plot_feature_distribution(training_obs, testing_obs, feature_name, feature_range)
