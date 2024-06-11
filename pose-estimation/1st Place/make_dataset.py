from pathlib import Path
import click

from helper import *


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


from scipy.signal import butter, lfilter
import copy
from scipy.spatial.transform import Rotation as R


def random_arrange_poses(poses, range0, max_distance_diff=99999999, max_angle_diff_deg=35):
    """
    Randomly arrange poses while ensuring distance and angle differences constraints.
    """
    num_poses = poses.shape[0]
    arranged_poses = [poses[0]]  # Start with the first pose
    current_pose = poses[0]
    poses = poses[1:]
    cand_id = 0
    #print(f'{poses[0]}, {poses[1]}: {poses[2]}, {poses[3]}..........')
    for _ in range(num_poses - 1):
        current_loc = current_pose[:3] * 1
        current_loc[0] = range0 - current_loc[0]
        candidates = []
        for idx in range(len(poses)):
            if idx != cand_id:
                selected_pose = poses[idx]

                selected_loc = selected_pose[:3] * 1
                selected_loc[0] = range0 - selected_loc[0]

                distance_diff = abs(np.linalg.norm(selected_loc)-np.linalg.norm((current_loc)))
                angle_diff = angle_between_quaternions(current_pose[3:], selected_pose[3:]) * 180.0 / np.pi
                #print(f'{distance_diff}, {angle_diff}..........')
                if distance_diff <= max_distance_diff and angle_diff <= max_angle_diff_deg:
                    candidates.append(idx)
        if not candidates:
            break
         
        cand_id = np.random.choice(candidates)
        selected_pose = poses[cand_id]
        poses = np.concatenate([poses[:cand_id], poses[cand_id+1:]], axis=0)
        arranged_poses.append(selected_pose)
        current_pose = selected_pose
    if len(arranged_poses) <= 90:
        print(f'???{len(arranged_poses)}')
    assert len(arranged_poses) > 90
    return np.stack(arranged_poses)

FEATS= [
    #cv2.SIFT_create(),
    cv2.ORB_create(scaleFactor=1.1, nlevels=12, edgeThreshold=10, patchSize=21),
    cv2.AKAZE_create(descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB, 
                     threshold=0.0005, nOctaves=8),
    #cv2.xfeatures2d.SURF_create(),
    #cv2.BRISK_create(),
    #cv2.FastFeatureDetector_create(),
    #cv2.GFTTDetector_create(),
    #cv2.SimpleBlobDetector_create()
]
# Create BFMatcher object
BFM = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
def get_rotations(keypoints_list, descriptors_list, rots, 
                  new_image, valid_count, all_count, intrinsics, lag):
    
    # Detect keypoints and compute descriptors
    for fi, feat in enumerate(FEATS):
        keypoints, descriptors = feat.detectAndCompute(new_image, None)
        keypoints_list[fi].append(keypoints)
        descriptors_list[fi].append(descriptors)
        
    if len(keypoints_list[0]) == 1:
        new_rot = np.array([[1.0, 0.0, 0.0, 0.0]*(lag+1)*N_ROTS]*len(FEATS))
    else:
        last_rot = rots[-1]
        quaternion_list = []
        for fi, feat in enumerate(FEATS):
            i = len(keypoints_list[0])-1
            _quaternion = []
            for li in range(lag+1):
                i0 = i - li - 1
                if i0 < 0:
                    _quaternion.append(_quaternion[-1])
                else:
                    #print(f'{fi}, {i0}, {len(keypoints_list)}, {len(keypoints_list[0])}')
                    keypoints1, keypoints2 = keypoints_list[fi][i0], keypoints_list[fi][i]
                    descriptors1, descriptors2 = descriptors_list[fi][i0], descriptors_list[fi][i]
                    try:
                        # Match keypoints
                        matches = BFM.match(descriptors1, descriptors2)

                        """
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.75 * n.distance:
                                good_matches.append(m)
                        """
                        # Sort matches by distance
                        matches = sorted(matches, key=lambda x: x.distance)

                        # Select top matches
                        good_matches = matches[:200]

                        # Extract corresponding keypoints
                        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                        retval, rotations, translation, normals = cv2.decomposeHomographyMat(H, intrinsics)

                        rotations = [rotation_matrix_to_quaternion(rot) for rot in rotations]
                        rotations = sorted(rotations, key=lambda x:-abs(x[0]))
                        if len(rotations) < N_ROTS:
                            rotations += [rotations[-1]] * (N_ROTS-len(rotations))
                        else:
                            rotations = rotations[:N_ROTS]
                        _quaternion.append(np.array(rotations).reshape(-1))
                        #_nan_ids.append(0)
                        valid_count += 1
                    except:
                        _quaternion.append([np.nan, np.nan, np.nan, np.nan]*N_ROTS)
                        #_nan_ids.append(1)
                    all_count += 1
                quaternion_list.append(np.array(_quaternion).reshape(-1))
                #nan_ids.append(_nan_ids)
        new_rot = np.array(quaternion_list)

    return keypoints_list, descriptors_list, new_rot, valid_count, all_count

import copy
class PE_Dataset_raw(torch.utils.data.Dataset):
    def __init__(self, df_file, data_folders, lag, mode='train'):
        self.data_folders = data_folders
        self.lag = lag
        self.mode = mode
        self.intrinsics = np.array([[5.2125371e+03, 0.0000000e+00, 6.4000000e+02],
            [0.0000000e+00, 6.2550444e+03, 5.1200000e+02],
            [0.0000000e+00, 0.0000000e+00, 1.0000000e+00]])
       
        self.data_paths = {}
        self.ranges = {}
        self.seq_ids = {}
        self.chain_ids = []
        self.labels = {}
        for ci, label in df_file.groupby('chain_id'):
            for folder in data_folders:
                paths = glob(f'{folder}{ci}/*.*')
                if len(paths) > 0:
                    break
            assert len(paths) > 0
            self.data_paths[ci] = sorted(paths)
            self.seq_ids[ci] = [int(path.split('/')[-1].split('\\')[-1][:-4]) for path in self.data_paths[ci]]
            label = label.sort_values('i')
            self.ranges[ci] = label['range'].values
            self.chain_ids.append(ci)
            if self.mode != 'test':
                self.labels[ci] = label[LABEL_NAMES].values
        
        
    def __len__(self):
        return len(self.chain_ids)
    
    def __getitem__(self, idc):
        ci = self.chain_ids[idc]
        used_paths = self.data_paths[ci]
        used_ranges = self.ranges[ci]
        used_seids = self.seq_ids[ci]
        used_labels = self.labels[ci]
        
        if MODE == 'shuf':
            labels_copy = copy.deepcopy(used_labels)
            np.random.shuffle(labels_copy)
            new_labels = random_arrange_poses(labels_copy, used_ranges[0])
            labels_dict = {tuple(pose):i for i, pose in enumerate(used_labels)}
            used_ids = []
            for pose in new_labels:
                used_ids.append(labels_dict[tuple(pose)])
        elif MODE == 'rev':
            used_ids = np.arange(len(used_paths))[::-1]
        else:
            used_ids = np.arange(len(used_paths))
        paths = np.array(used_paths)[used_ids]
        ranges = np.array(used_ranges)[used_ids]
        labels = np.array(used_labels)[used_ids]
        seids = np.array(used_seids)[used_ids]
        
        keypoints_list = [[] for _ in range(len(FEATS))]
        descriptors_list = [[] for _ in range(len(FEATS))]
        rots = []
        valid_count = 0
        all_count = 0
        for p in paths:
            image = cv2.imread(p)
            keypoints_list, descriptors_list, rot, valid_count, all_count = get_rotations(keypoints_list, 
                                                                                        descriptors_list, rots, 
                                                                                        image, valid_count,
                                                                                        all_count, self.intrinsics,
                                                                                        self.lag)
            rots.append(rot)
        #print(f'{ci} nan rate: {min([len(nanids[0][]) for i in nanids[0]])/len(rots[0])}')
        valid_rate = valid_count / all_count
        with open(f'data_cache/rots/{ci}', 'wb') as f:
            pickle.dump({'rots': rots, 'labels': labels, 'paths':paths, 
                         'labels':labels, 'seids':seids}, f)
        return rots, valid_rate

LAG = 0
N_ROTS = 4
MODE = 'none'#'shuf'

@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
def main(data_dir):
    data_dir = Path(data_dir).resolve()
    
    train = pd.read_csv(f'{data_dir}/train_labels.csv')
    ranges = pd.read_csv(f'{data_dir}/range.csv')
    train = train.merge(ranges, how='left', on=['chain_id', 'i'])
    train = train.groupby('chain_id').apply(lambda group: group.interpolate()).reset_index(drop=True)

    LABEL_NAMES = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
    DATA_FOLDERS = [
        f'{data_dir}/images/'
    ]

    
    os.system('mkdir data_cache')
    os.system('mkdir data_cache/rots')
    train_dataset = PE_Dataset_raw(train,
                            DATA_FOLDERS, LAG)

    val_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=None, num_workers=os.cpu_count(), shuffle=False, drop_last=False)

    mean_valid_rate = []
    for d in tqdm(val_loader):
        _, valid_rate = d
        print(f'valid_rate: {valid_rate}')
        mean_valid_rate.append(valid_rate) 
    print(f'valid_rate: {np.mean(mean_valid_rate)}')

    os.system(f'cp {data_dir}/range.csv data_cache/range.csv')
    os.system(f'cp {data_dir}/train_labels.csv data_cache/train_labels.csv')

    
if __name__ == "__main__":
    main()