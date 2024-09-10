import matplotlib.pyplot as plt
import numpy as np
import pickle
from icecream import ic

# Load the observations
with open('/tmp/training_dist.pkl', 'rb') as f:
    training_obs = np.array(pickle.load(f))
    ic(training_obs.shape)
# with open('/tmp/testing_dist.pkl', 'rb') as f:
#     testing_obs = np.array(pickle.load(f))
#     ic(testing_obs.shape)
# with open('/tmp/training_high_level_dist.pkl', 'rb') as f:
#     training_obs= np.array(pickle.load(f))
#     testing_obs= training_obs
#     ic(training_obs.shape)


# Define the features and their corresponding index ranges
# Note: Adjust the start and end indices based on the cumulative dimensionality of each component
feature_ranges = {
    'Base lin vel': range(0, 3),
    'Base ang bel': range(3, 6),
    'Projected gravity': range(6, 9),
    'Object pose wrt left hand': range(9, 15),
    'Object pose wrt right hand': range(15, 21),
    'Fingertip contact': range(21, 27),
    'Joint states': range(27, 64),
    'Joint velocity': range(64, 101),
    'Action History': range(101, 138),
}

# high_level_feature_ranges = {
#     'Body Orientation': range(0, 2),
#     'DOF Position': range(2, 22),
#     'DOF Velocity': range(22, 42),
#     'Base Lin, Ang Vel': range(42, 48),
#     'EE Base yaw Pos, Orn': range(48, 54),
#     'Action History': range(54, 64),
#     'Object pose in Base yaw': range(64, 70),
#     'Object EE delta pose': range(70, 76),
#     'Object params': range(76, 78),
# }

def plot_feature_distribution(test_observations, feature_name, feature_range):
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
