import torch
import numpy as np
import matplotlib.pyplot as plt

joint_names = [
    "hip_pitch_l", "hip_pitch_r", "waist_yaw", "hip_roll_l", "hip_roll_r", 
    "waist_roll", "hip_yaw_l", "hip_yaw_r", "waist_pitch", "knee_l", "knee_r",
    "shoulder_pitch_l", "shoulder_pitch_r", "ankle_pitch_l", "ankle_pitch_r", 
    "shoulder_roll_l", "shoulder_roll_r", "ankle_roll_l", "ankle_roll_r",
    "shoulder_yaw_l", "shoulder_yaw_r", "elbow_l", "elbow_r", "wrist_roll_l",
    "wrist_roll_r", "wrist_pitch_l", "wrist_pitch_r", "wrist_yaw_l", "wrist_yaw_r"
]
sections = {
    # 1: {'grid': (2, 3), 'title': 'base ang vel & proj grav', 'vars': [
    #     ("base_ang_vel_x", 0),
    #     ("base_ang_vel_y", 1),
    #     ("base_ang_vel_z", 2),
    #     ("proj_grav_x", 3),
    #     ("proj_grav_y", 4),
    #     ("proj_grav_z", 5)
    # ]},
    # 2: {'grid': (4, 3), 'title': 'foot pose', 'vars': [
    #     ("foot_l_pos_x", 6),  ("foot_l_pos_y", 7),  ("foot_l_pos_z", 8),
    #     ("foot_r_pos_x", 9),  ("foot_r_pos_y", 10), ("foot_r_pos_z", 11),
    #     ("foot_l_axa_x", 12), ("foot_l_axa_y", 13), ("foot_l_axa_z", 14),
    #     ("foot_r_axa_x", 15), ("foot_r_axa_y", 16), ("foot_r_axa_z", 17)
    # ]},
    # 3: {'grid': (4, 3), 'title': 'hand pose', 'vars': [
    #     ("hand_l_pos_x", 18), ("hand_l_pos_y", 19), ("hand_l_pos_z", 20),
    #     ("hand_r_pos_x", 21), ("hand_r_pos_y", 22), ("hand_r_pos_z", 23),
    #     ("hand_l_axa_x", 24), ("hand_l_axa_y", 25), ("hand_l_axa_z", 26),
    #     ("hand_r_axa_x", 27), ("hand_r_axa_y", 28), ("hand_r_axa_z", 29)
    # ]},
    # 4: {'grid': (5, 6), 'title': 'joint position', 'vars': [(name, 30+i) for i, name in enumerate(joint_names)]},
    5: {'grid': (5, 6), 'title': 'joint velocity', 'vars': [(name, 59+i) for i, name in enumerate(joint_names)]},
    # 6: {'grid': (2, 3), 'title': 'hands command', 'vars': [
    #     ("hands_cmd_pos_dx", 88), ("hands_cmd_pos_dy", 89), ("hands_cmd_pos_dz", 90),
    #     ("hands_cmd_axa_dx", 91), ("hands_cmd_axa_dy", 92), ("hands_cmd_axa_dz", 93)
    # ]},
    # 7: {'grid': (1, 1), 'title': 'pelvis height', 'vars': [("pelvis_height", 94)]}
}

real_dict = np.load(
    # '/tmp/e2e/log_0926_l3_30k_1759023822.npy',
    # "/tmp/e2e/log_0926_l3_30k_1759024130.npy",
    # "/tmp/e2e/log_0926_l3_30k_1759028934.npy",
    # "/tmp/e2e/log_0926_l3_30k_1759029068.npy",
    # "/tmp/e2e/log_0926_l3_30k_1759029189.npy",
    "/tmp/e2e/log_0926_l3_30k_1759029264.npy",
    allow_pickle=True
).item()
real_data = real_dict['trajectories']['sit_observations']  # [timestep, obs_dim]
START_TIME = 0
END_TIME = 100
START_STEPS = int(START_TIME / 0.02)
END_STEPS = int(END_TIME / 0.02)
END_STEPS = min(END_STEPS, real_data.shape[0])
time_steps = np.arange(START_STEPS, END_STEPS)

for fig_id, sec in sections.items():
    rows, cols = sec['grid']
    title = sec['title']
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3), dpi=100)
    axes_flat = np.ravel(axes)

    for idx, (label, obs_idx) in enumerate(sec['vars']):
        real_window = real_data[START_STEPS:END_STEPS, obs_idx]

        ax = axes_flat[idx]
        ax.plot(time_steps, real_window, '--', color='r', label='real', marker='o', markersize=1, markerfacecolor='g', markeredgecolor='g')

        ax.set_title(label, fontsize=10)
        ax.set_xlim([START_STEPS, END_STEPS])
        ax.tick_params(axis='both', which='major', labelsize=8)

        if idx == 0:
            ax.legend(loc='upper right', fontsize='x-small')

    for j in range(len(sec['vars']), len(axes_flat)):
        axes_flat[j].axis('off')

    fig.suptitle(
        f'Figure {fig_id}: {title}',
        fontsize=16
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    # fig.savefig(f'figures/figure_{fig_id}.png', dpi=100)
    plt.show()
