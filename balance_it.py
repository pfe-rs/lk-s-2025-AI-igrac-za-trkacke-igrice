import os
import pickle
import random

def load_and_balance_recordings(
    recordings_folder="clean-codes/recordings/",
    gas_keep_ratio=0.1,
    target_count=1000
):
    all_states = []
    all_actions = []
    numbers = [0, 0, 0, 0]

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)

                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])
                    numbers = [x + y for x, y in zip(numbers, action)]

    print(f"Loaded {len(all_states)} total samples.")
    print(f"Total action counts: {numbers}")

    # Separate frames by action type
    gas_frames = []
    brake_frames = []
    left_frames = []
    right_frames = []

    for state, action in zip(all_states, all_actions):
        if action[1]:  # brake
            brake_frames.append((state, action))
        elif action[2]:  # left
            left_frames.append((state, action))
        elif action[3]:  # right
            right_frames.append((state, action))
        elif action[0] and not any(action[1:]):  # pure gas-only
            if random.random() < gas_keep_ratio:
                gas_frames.append((state, action))

    print(f"After gas filtering:")
    print(f"Gas-only: {len(gas_frames)} Brake: {len(brake_frames)} Left: {len(left_frames)} Right: {len(right_frames)}")

    # Upsample rare actions to approx target_count each
    def upsample(data_list, target):
        if len(data_list) >= target:
            return data_list
        return data_list + random.choices(data_list, k=target - len(data_list))

    brake_frames = upsample(brake_frames, target_count)
    left_frames = upsample(left_frames, target_count)
    right_frames = upsample(right_frames, target_count)

    # Combine all after upsampling
    final_states = [s for s, _ in gas_frames]
    final_actions = [a for _, a in gas_frames]

    final_states.extend([s for s, _ in brake_frames])
    final_actions.extend([a for _, a in brake_frames])

    final_states.extend([s for s, _ in left_frames])
    final_actions.extend([a for _, a in left_frames])

    final_states.extend([s for s, _ in right_frames])
    final_actions.extend([a for _, a in right_frames])

    # Final counts
    numbers = [0, 0, 0, 0]
    for action in final_actions:
        numbers = [x + y for x, y in zip(numbers, action)]

    print(f"\nFinal balanced dataset size: {len(final_states)}")
    print(f"Final action counts: {numbers}")

    return final_states, final_actions


def load_and_balance_recordings_new(
    recordings_folder="clean-codes/recordings/",
    gas_keep_ratio=0.1,
    target_count=1000
):
    import random
    import os
    import pickle

    all_states = []
    all_actions = []

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)
                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])

    # Separate frames by action type
    gas_frames = []
    brake_frames = []
    left_frames = []
    right_frames = []

    for state, action in zip(all_states, all_actions):
        if action[1]:  # brake
            brake_frames.append((state, action))
        elif action[2]:  # left
            left_frames.append((state, action))
        elif action[3]:  # right
            right_frames.append((state, action))
        elif action[0] and not any(action[1:]):  # pure gas-only
            if random.random() < gas_keep_ratio:
                gas_frames.append((state, action))

    print(f"Before balancing:")
    print(f"Gas-only: {len(gas_frames)} Brake: {len(brake_frames)} Left: {len(left_frames)} Right: {len(right_frames)}")

    def resample_to_target(data_list, target):
        if len(data_list) > target:
            # Downsample by random sampling without replacement
            return random.sample(data_list, target)
        elif len(data_list) < target:
            # Upsample by random sampling with replacement
            return data_list + random.choices(data_list, k=target - len(data_list))
        else:
            return data_list

    # Resample brake, left, right to exact target_count
    brake_frames = resample_to_target(brake_frames, target_count)
    left_frames = resample_to_target(left_frames, target_count)
    right_frames = resample_to_target(right_frames, target_count)

    print(f"After balancing:")
    print(f"Gas-only: {len(gas_frames)} Brake: {len(brake_frames)} Left: {len(left_frames)} Right: {len(right_frames)}")

    # Combine all after balancing
    final_states = [s for s, _ in gas_frames]
    final_actions = [a for _, a in gas_frames]

    final_states.extend([s for s, _ in brake_frames])
    final_actions.extend([a for _, a in brake_frames])

    final_states.extend([s for s, _ in left_frames])
    final_actions.extend([a for _, a in left_frames])

    final_states.extend([s for s, _ in right_frames])
    final_actions.extend([a for _, a in right_frames])

    # Final counts
    numbers = [0, 0, 0, 0]
    for action in final_actions:
        numbers = [x + y for x, y in zip(numbers, action)]

    print(f"\nFinal balanced dataset size: {len(final_states)}")
    print(f"Final action counts: {numbers}")

    return final_states, final_actions




def load_and_balance_recordings_only_gas(
    recordings_folder="clean-codes/recordings/",
    gas_keep_ratio=0.1
):
    import random
    import os
    import pickle

    all_states = []
    all_actions = []

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)
                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])

    # Separate frames by action type
    gas_frames = []
    brake_frames = []
    left_frames = []
    right_frames = []

    for state, action in zip(all_states, all_actions):
        if action[1]:  # brake
            brake_frames.append((state, action))
        elif action[2]:  # left
            left_frames.append((state, action))
        elif action[3]:  # right
            right_frames.append((state, action))
        elif action[0] and not any(action[1:]):  # pure gas-only
            if random.random() < gas_keep_ratio:
                gas_frames.append((state, action))

    print(f"After filtering gas-only frames:")
    print(f"Gas-only: {len(gas_frames)} Brake: {len(brake_frames)} Left: {len(left_frames)} Right: {len(right_frames)}")

    # Combine all after filtering
    final_states = [s for s, _ in gas_frames]
    final_actions = [a for _, a in gas_frames]

    final_states.extend([s for s, _ in brake_frames])
    final_actions.extend([a for _, a in brake_frames])

    final_states.extend([s for s, _ in left_frames])
    final_actions.extend([a for _, a in left_frames])

    final_states.extend([s for s, _ in right_frames])
    final_actions.extend([a for _, a in right_frames])

    # Final counts
    numbers = [0, 0, 0, 0]
    for action in final_actions:
        numbers = [x + y for x, y in zip(numbers, action)]

    print(f"\nFinal dataset size: {len(final_states)}")
    print(f"Final action counts: {numbers}")

    return final_states, final_actions
def load_and_balance_for_double(recordings_folder="clean-codes/recordings/"):
    import random
    import os
    import pickle

    all_states = []
    all_actions = []

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)
                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])

    # Separate frames by action type
    gas_brake = []
    left_right = []

    for state, action in zip(all_states, all_actions):
        # Normalize gas_brake: if both pressed, zero gas
        if not (action[0] and action[1]):
            gas_brake.append((state, [action[0], action[1]]))
        else:
            gas_brake.append((state, [0, action[1]]))

        # Normalize left_right: if both pressed, ignore left
        if not (action[2] and action[3]):
            left_right.append((state, [action[2], action[3]]))

    def count_actions(data):
        counts = [0, 0]
        for _, action in data:
            if action[0]:
                counts[0] += 1
            if action[1]:
                counts[1] += 1
        return counts

    # 1. Balance gas_brake inside gas_brake dataset
    gb_counts = count_actions(gas_brake)
    print("Before gas_brake balancing:", gb_counts)

    if gb_counts[0] > gb_counts[1]:
        diff = gb_counts[0] - gb_counts[1]
        minority_samples = [sample for sample in gas_brake if sample[1][1]]
        for _ in range(diff):
            gas_brake.append(random.choice(minority_samples))
    elif gb_counts[1] > gb_counts[0]:
        diff = gb_counts[1] - gb_counts[0]
        minority_samples = [sample for sample in gas_brake if sample[1][0]]
        for _ in range(diff):
            gas_brake.append(random.choice(minority_samples))

    gb_counts = count_actions(gas_brake)
    print("After gas_brake balancing:", gb_counts)

    # 2. Balance left_right inside left_right dataset
    lr_counts = count_actions(left_right)
    print("Before left_right balancing:", lr_counts)

    if lr_counts[0] > lr_counts[1]:
        diff = lr_counts[0] - lr_counts[1]
        minority_samples = [sample for sample in left_right if sample[1][1]]
        for _ in range(diff):
            left_right.append(random.choice(minority_samples))
    elif lr_counts[1] > lr_counts[0]:
        diff = lr_counts[1] - lr_counts[0]
        minority_samples = [sample for sample in left_right if sample[1][0]]
        for _ in range(diff):
            left_right.append(random.choice(minority_samples))

    lr_counts = count_actions(left_right)
    print("After left_right balancing:", lr_counts)

    # 3. Balance gas_brake and left_right datasets so both have roughly equal number of samples

    len_gb = len(gas_brake)
    len_lr = len(left_right)

    print(f"Before final balancing: gas_brake size = {len_gb}, left_right size = {len_lr}")

    if len_gb > len_lr:
        diff = len_gb - len_lr
        for _ in range(diff):
            left_right.append(random.choice(left_right))
    elif len_lr > len_gb:
        diff = len_lr - len_gb
        for _ in range(diff):
            gas_brake.append(random.choice(gas_brake))

    print(f"After final balancing: gas_brake size = {len(gas_brake)}, left_right size = {len(left_right)}")

    # print(len(gas_brake))
    # print(len(gas_brake[0]))

    # print(len(left_right))
    # print(len(left_right[0]))


    # Return or further process your balanced data if needed
    return gas_brake, left_right

def load_and_balance_for_double_maybe(recordings_folder="clean-codes/recordings/"):
    import random
    import os
    import pickle

    all_states = []
    all_actions = []

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)
                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])

    # Separate frames by action type
    gas_brake = []
    left_right = []

    for state, action in zip(all_states, all_actions):
        # Normalize gas_brake: if both pressed, zero gas
        if not (action[0] and action[1]):
            gas_brake.append((state, [action[0], action[1]]))
        else:
            gas_brake.append((state, [0, action[1]]))

        # Normalize left_right: if both pressed, ignore left
        if not (action[2] and action[3]):
            left_right.append((state, [action[2], action[3]]))

    def count_actions(data):
        counts = [0, 0]
        for _, action in data:
            if action[0]:
                counts[0] += 1
            if action[1]:
                counts[1] += 1
        return counts

    # 1. Balance gas_brake inside gas_brake dataset (modified)
    gb_counts = count_actions(gas_brake)
    print("Before gas_brake balancing:", gb_counts)

    gas_samples = [sample for sample in gas_brake if sample[1][0]]
    brake_samples = [sample for sample in gas_brake if sample[1][1]]
    neither_samples = [sample for sample in gas_brake if not (sample[1][0] or sample[1][1])]

    # Double the brake samples
    # brake_samples = brake_samples * 2

    # Trim gas samples randomly if they exceed brake count
    if len(gas_samples) > len(brake_samples):
        gas_samples = random.sample(gas_samples, len(brake_samples))

    # Recombine all samples
    gas_brake = gas_samples + brake_samples+ neither_samples

    gb_counts = count_actions(gas_brake)
    print("After gas_brake balancing:", gb_counts)

    # 2. Balance left_right inside left_right dataset (unchanged)
    lr_counts = count_actions(left_right)
    print("Before left_right balancing:", lr_counts)

    if lr_counts[0] > lr_counts[1]:
        diff = lr_counts[0] - lr_counts[1]
        minority_samples = [sample for sample in left_right if sample[1][1]]
        for _ in range(diff):
            left_right.append(random.choice(minority_samples))
    elif lr_counts[1] > lr_counts[0]:
        diff = lr_counts[1] - lr_counts[0]
        minority_samples = [sample for sample in left_right if sample[1][0]]
        for _ in range(diff):
            left_right.append(random.choice(minority_samples))

    lr_counts = count_actions(left_right)
    print("After left_right balancing:", lr_counts)

    # 3. Balance gas_brake and left_right datasets so both have roughly equal number of samples
    len_gb = len(gas_brake)
    len_lr = len(left_right)

    print(f"Before final balancing: gas_brake size = {len_gb}, left_right size = {len_lr}")

    # if len_gb > len_lr:
    #     diff = len_gb - len_lr
    #     for _ in range(diff):
    #         left_right.append(random.choice(left_right))
    # elif len_lr > len_gb:
    #     diff = len_lr - len_gb
    #     for _ in range(diff):
    #         gas_brake.append(random.choice(gas_brake))

    print(f"After final balancing: gas_brake size = {len(gas_brake)}, left_right size = {len(left_right)}")

    

    return gas_brake, left_right

def load_only(
    recordings_folder="clean-codes/recordings2/"
):
    import random
    import os
    import pickle

    all_states = []
    all_actions = []

    # Load all recordings
    for filename in os.listdir(recordings_folder):
        if filename.endswith(".pkl"):
            with open(os.path.join(recordings_folder, filename), "rb") as f:
                record = pickle.load(f)
                for state, action in record:
                    all_states.append(state)
                    all_actions.append([float(a) for a in action])

    # Separate frames by action type
    gas_brake = []
    left_right = []

    for state, action in zip(all_states, all_actions):
        # Normalize gas_brake: if both pressed, zero gas
        if not (action[0] and action[1]):
            gas_brake.append((state, [action[0], action[1]]))
        else:
            gas_brake.append((state, [0, action[1]]))

        # Normalize left_right: if both pressed, ignore left
        if not (action[2] and action[3]):
            left_right.append((state, [action[2], action[3]]))

    return gas_brake,left_right

