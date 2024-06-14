from helper import *
from mathutils import Quaternion
from time import time
from sklearn.model_selection import KFold, GroupKFold
from transformers import T5Model, AutoConfig, AutoModel
import pickle
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.cuda import amp

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def apply_delta_quaternion_train(base_quaternion, delta_quaternion_angle):
    # Convert quaternion angles to mathutils Quaternions
    base_quaternion = Quaternion(base_quaternion)
    delta_quaternion = Quaternion(delta_quaternion_angle)
    
    # Apply the delta quaternion to the base quaternion
    result_quaternion = delta_quaternion @ base_quaternion
    
    # Convert the result quaternion to a tuple of quaternion angles
    result_quaternion_angle = (result_quaternion.w, result_quaternion.x,
                               result_quaternion.y, result_quaternion.z)
    
    return np.array(result_quaternion_angle)

# in case where mathutils is not available
''' 
def apply_delta_quaternion_train(base_quaternion, delta_quaternion_angle):
    base_quaternion = base_quaternion.astype(np.float32)
    delta_quaternion_angle = delta_quaternion_angle.astype(np.float32)
    
    # Apply the delta quaternion to the base quaternion
    result_quaternion_angle = quaternion_multiply(delta_quaternion_angle, base_quaternion)
    
    return result_quaternion_angle.astype(np.float64)
'''

def label_transform(labels):
    out = []
    l0 = labels[0]
    rot0 = l0[3:]
    true_loc0 = copy.deepcopy(l0[:3])
    true_rot0 = copy.deepcopy(l0[3:])
    true_loc0 *= -1
    true_rot0[1:] *= -1
    l0[3:] /= np.sum(l0[3:]**2)**0.5
    for l1 in labels:
        l1[3:] /= np.sum(l1[3:]**2)**0.5
        rot1 = l1[3:]
        true_loc1 = copy.deepcopy(l1[:3])
        true_rot1 = copy.deepcopy(l1[3:])
        true_loc1 *= -1
        true_rot1[1:] *= -1
        
        rot = apply_delta_quaternion_train(true_rot1, rot0)
        rot /= sum(rot**2) ** 0.5
        rot[1:] *= -1
        
        loc = true_loc1 - true_loc0
        loc = rotate_point_by_quaternion(loc, true_rot0)
        loc *= -1
        #rot[0] = -rot[0]
        out.append(np.concatenate([loc, rot], axis=0))
        
    #for i in range(len(out)):
    #    out[i][:3] = out[i][:3] - out[0][:3]
    return np.array(out)

class PE_Dataset_train(torch.utils.data.Dataset):
    def __init__(self, df_file, data_folders, mode='train'):
        self.data_folders = data_folders
        self.mode = mode
       
        self.data = {}
        self.ranges = {}
        self.ranges_scaled = {}
        self.seq_ids = {}
        self.chain_ids = []
        self.labels = {}
        for ci, label in df_file.groupby('chain_id'):
            for folder in data_folders:
                paths = glob(f'{folder}{ci}')
                if len(paths) > 0:
                    break
            assert len(paths) == 1
            with open(paths[0], 'rb') as f:
                rots = pickle.load(f)
            rots = np.array(rots['rots']).reshape(len(rots['rots']), -1)

            #nan_ids = np.isnan(rots[:, ::4])
            #rots[:, ::4][nan_ids] = 1.0
            #nan_ids = np.isnan(rots)
            #rots[nan_ids] = 0.0
            self.data[ci] = rots

            label = label.sort_values('i')
            self.seq_ids[ci] = label['i'].values
            self.ranges[ci] = label['range'].values
            self.ranges_scaled[ci] = label['range_scaled'].values
            self.chain_ids.append(ci)
            if self.mode != 'test':
                self.labels[ci] = label[LABEL_NAMES].values
        
    def __len__(self):
        return len(self.chain_ids)
    
    def decompose_label(self, label, r0):
        locs = label[:, :3]
        locs[:, 0] = r0 - locs[:, 0]
        uint_locs = locs / np.linalg.norm(locs, axis=-1, keepdims=True)
        rots = label[:, 3:]
        delta_rots = rots[:-1]
        #delta_rots[:, 1:] = -delta_rots[:, 1:]
        
        new_uint_locs = [uint_locs[0]]
        new_rots = [rots[0]]
        for i in range(len(delta_rots)): 
            new_rots.append(apply_delta_quaternion_train(rots[i+1], delta_rots[i]))
            new_uint_locs.append(rotate_point_by_quaternion(uint_locs[i+1], delta_rots[i]))
        return np.array(new_uint_locs), np.array(new_rots)   
        
    def recompose_label(self, uint_locs, rots, lens, r0):
        new_locs = [uint_locs[0]*lens[0]]
        new_rots = [rots[0]]
        new_locs[0][0] = r0 - new_locs[0][0]
        for ul, ro, le in zip(uint_locs[1:], rots[1:], lens[1:]):
            delta_rot = new_rots[-1] * -1
            delta_rot[0] = -delta_rot[0]
            new_loc = rotate_point_by_quaternion(ul, delta_rot) * le
            new_loc[0] = r0 - new_loc[0]
            new_locs.append(new_loc)
            new_rots.append(apply_delta_quaternion_train(ro, delta_rot))
        return np.concatenate([new_locs, new_rots], axis=-1)
    
    def __getitem__(self, idc):
        ci = self.chain_ids[idc]
        rots = self.data[ci]
        used_ranges = np.array(self.ranges[ci])
        used_ranges_scaled = np.array(self.ranges_scaled[ci])
        used_seids = np.array(self.seq_ids[ci])

        ranges = used_ranges
        ranges_scaled = used_ranges_scaled
        seids = used_seids[1:]
        if self.mode == 'test':
            rots = torch.tensor(np.array(rots), dtype=torch.float32)
            ranges_scaled = torch.tensor(np.array(ranges_scaled), dtype=torch.float32)
            ranges = torch.tensor(np.array(ranges), dtype=torch.float32)
            return rots, ranges_scaled, ranges, ci, seids
        
        used_labels = np.array(self.labels[ci])
        
        used_ids = np.arange(len(used_ranges))
        #print(f'p0: {used_ids.shape}')
        if self.mode == 'train':
            rev = np.random.rand() < 0.5
            if np.random.rand() < 0.4:
                offset = 0
                _used_ids = np.arange(1, len(used_ranges))
                if np.random.rand() < 1.0:
                    np.random.shuffle(_used_ids)
                else:
                    cuti = np.random.randint(len(_used_ids))
                    _used_ids = list(_used_ids[cuti:]) + list(_used_ids[:cuti])
                used_ids[1:] = np.array(_used_ids)
                #print(f'p1: {used_labels[:10]}, {ranges[:10]}')
                new_uint_locs, new_rots = self.decompose_label(used_labels[1:], ranges[0])
                new_lens_offset = np.random.rand(len(used_labels)-1)*2 - 1
                new_lens = [ranges[0]]
                for lo in new_lens_offset:
                    new_lens.append(new_lens[-1]+lo)
                used_labels[1:] = self.recompose_label(new_uint_locs[used_ids[1:]-1], new_rots[used_ids[1:]-1],
                                              new_lens, ranges[0])
                ranges = np.array(new_lens)
                aug_mask = np.random.rand(len(ranges)) < 0.1
                aug_offset = -np.random.rand(len(ranges))*6
                drop_mask = np.random.rand(len(ranges)) < 0.2
                drop_mask[0] = False
                ranges[aug_mask] += aug_offset[aug_mask]
                ranges[drop_mask] = np.nan
                ranges = pd.Series(ranges).fillna(method='ffill').values
                #print(ranges)
                
                #print(f'p2: {used_labels[:10]}, {ranges[:10]}')
                #a = 1/0
                rots = rots[used_ids]
                ranges = ranges[used_ids]
                ranges_scaled = ranges_scaled[used_ids]
            else:
                offset = np.random.randint(len(rots)//2)
                
        else:
            offset = 0
            rev = False
        
        #print(f'p1: {rots.shape}')
        ranges_scaled = ranges / 300
        rots = np.array([[rot[i:i+CFG.used_nrots*4] for i in range(0, len(rot), CFG.nrots*4)] for rot in rots])
        seq_nrots = rots.shape[1]
        pad = rots[0] * 1
        
        if rev:
            rots = rots[::-1]
            rots[1:] = rots[:-1]
            for i in range(CFG.used_nrots):
                rots[:, :, i*4+1:(i+1)*4] = -rots[:, :, i*4+1:(i+1)*4]
            ranges = ranges[::-1]
            ranges_scaled = ranges_scaled[::-1]
        rots = rots[offset:].copy()
        if offset > 0 or rev:
            assert CFG.lag == 0
            rots[0] = pad
        
        ranges = ranges[offset:].copy()
        ranges_scaled = ranges_scaled[offset:].copy()
        seids = seids[offset:].copy()
        
        # --add more features--
        new_rots = []
        for i0 in range(rots.shape[0]):
            _rot = rots[i0]
            _nan_ids = np.isnan(_rot)
            _rot[_nan_ids] = rots[i0-1][_nan_ids]
            rots[i0] = _rot
            #assert np.isnan(rots[i0]).sum() == 0
            new_rots.append([])
            for i1 in range(rots.shape[1]):
                new_rots[-1].append([])
                for i2 in range(0, rots.shape[2], 4):
                    _rot_cache = list(rots[i0, i1, i2:i2+4])
                    for newfeat_lag in CFG.newfeat_lags:
                        if i0 <= newfeat_lag:
                            _rot_cache += list(rots[i0, i1, i2:i2+4])
                        else:
                            _rot_cache += list(apply_delta_quaternion_train(rots[i0, i1, i2:i2+4],
                                                                     rots[i0-newfeat_lag, i1, :4]))
                    new_rots[-1][-1].append(_rot_cache)
        rots = np.array(new_rots)
        
        rots = rots.reshape([rots.shape[0], -1])
        
        #print([img.shape for img in images])
        rots = torch.tensor(np.array(rots), dtype=torch.float32)
        ranges_scaled = torch.tensor(np.array(ranges_scaled), dtype=torch.float32)
        ranges = torch.tensor(np.array(ranges), dtype=torch.float32)
        #print(rots.shape, ranges.shape)
        if rev:
            used_labels = used_labels[::-1]
        used_labels = used_labels[offset:]
        if offset>0 or rev:
            used_labels = label_transform(used_labels)
        labels = used_labels[1:]
        labels = torch.tensor(np.array(labels), dtype=torch.float32)
        
        return rots, ranges_scaled, ranges, labels, ci, seids

class PE_Model(nn.Module):
    def __init__(self):
        super().__init__()
        #self.encoder = tv.models.get_model(CFG.encoder_name, weights="DEFAULT")
        self.decoder = T5Model.from_pretrained(CFG.decoder_name).decoder
        #self.decoder.block += T5Model.from_pretrained(CFG.decoder_name).decoder.block
        self.decoder.config.num_layers = len(self.decoder.block)
        #print(self.encoder)
        """
        h_size = self.encoder.classifier[-1].weight.shape[-1]
        self.encoder.classifier = torch.nn.Linear(h_size, self.decoder.config.d_model-1)
        
        h_size = self.encoder.head[-1].weight.shape[-1]
        self.encoder.head[-1] = torch.nn.Linear(h_size, self.decoder.config.d_model-1)
        h_size = self.encoder.fc.weight.shape[-1]
        self.encoder.fc = torch.nn.Linear(h_size, self.decoder.config.d_model-1)
         
        h_size = self.encoder.classifier[-1].weight.shape[-1]
        self.encoder.classifier[-1] = torch.nn.Linear(h_size, self.decoder.config.d_model-1)
        """
        self.trans = torch.nn.Linear((CFG.lag+1)*CFG.used_nrots*CFG.n_feats*4*(len(CFG.newfeat_lags)+1),
                                     self.decoder.config.d_model-1) 
        self.head = torch.nn.Linear(self.decoder.config.d_model, 256)
        self.head2 = torch.nn.Linear(256, 3)
        
    def unit_vectors_to_quaternions(self, unit_vectors):
        # Ensure unit vectors have unit length
        unit_vectors = unit_vectors / torch.norm(unit_vectors, dim=1, keepdim=True)

        # Calculate rotation angle around axis
        angles = torch.acos(unit_vectors[:, :, 0])  # Angle between unit vectors and positive x-axis

        # Calculate quaternion components
        qw = torch.cos(angles / 2)
        qx = torch.sin(angles / 2) * unit_vectors[:, :, 1]
        qy = torch.sin(angles / 2) * unit_vectors[:, :, 2]
        qz = torch.sin(angles / 2) * unit_vectors[:, :, 3]

        # Return quaternions
        return torch.stack((qw, qx, qy, qz), dim=-1)
        
    def forward(self, xs, rs, r): 
        hs = torch.cat([self.trans(xs), rs[..., None]], dim=-1)
        
        hs = self.decoder(inputs_embeds=hs[None]).last_hidden_state[0, 1:]
        out = self.head(hs)
        loc_offset = self.head2(out)
        out2 = out[:, -4:] / torch.linalg.norm(out[:, -4:], dim=-1, keepdim=True)
        #out = self.head2(out) * 400
        base_loc = torch.tensor([[-r[0], 0, 0]]*out2.shape[0], dtype=out2.dtype, 
                                     device=out2.device)
        out = rotate_vector_by_quaternion(base_loc, out2) - base_loc
        out = torch.cat([out+loc_offset, out2], dim=1)
        return out

def train_one_fold(CFG, val_fold, train_all, output_path, model=None, data_ids=None):
    """Main"""
    torch.backends.cudnn.benchmark = True
    device = torch.device(CFG.device)
    
    #train_path_label, val_path_label, _, _ = get_path_label(val_fold, train_all)
    #train_transform, val_transform = get_transforms(CFG)
    
    if data_ids is None:
        train_dataset = PE_Dataset_train(train_all[train_all['fold']!=val_fold], DATA_FOLDERS)
    else:
        train_dataset = PE_Dataset_train(train_all[train_all['fold']!=val_fold].iloc[data_ids],
                                        DATA_FOLDERS)
    val_dataset = PE_Dataset_train(train_all[train_all['fold']==val_fold], DATA_FOLDERS, mode='valid')
    print(f'train len: {len(train_dataset)}, valid len: {len(val_dataset)}')
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=None, num_workers=os.cpu_count(), shuffle=True, drop_last=False)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=None, num_workers=os.cpu_count(), shuffle=False, drop_last=False)
    
    if model is None:
        model = PE_Model()
        model = model.to(device)
        #model = nn.DataParallel(model)
    
    optimizer = optim.AdamW(params=model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer=optimizer, epochs=CFG.max_epoch,
        pct_start=0.3, steps_per_epoch=round(len(train_loader)/CFG.batch_size),
        max_lr=CFG.lr, div_factor=25, final_div_factor=4.0e-01
    )
    
    loss_func = my_loss
    #loss_func.to(device)
    loss_func_val = my_loss
    
    use_amp = CFG.enable_amp
    scaler = amp.GradScaler(enabled=use_amp)
    
    best_val_score = 9999999
    best_epoch = 0    
    saved_dict = {}
    for epoch in range(1, CFG.max_epoch + 1):
        train_loss = []
        train_loss2 = []
        epoch_start = time()
        model.train()
        ga_count = 0
        do_optimize = True
        for batch in tqdm(train_loader):
            #batch = to_device(batch, device)
            x, rs, r, t, _, _ = batch
            x = x.to(device)
            r = r.to(device)
            rs = rs.to(device)
            t = t.to(device)
            #with amp.autocast(use_amp):
            y = model(x, rs, r)
            #print(torch.isnan(x).sum(), torch.isnan(y).sum(), torch.isnan(t).sum())

            #print(x.shape, y.shape, t.shape)
            loss = loss_func(y, t, r)
            loss2 = loss2_func(t, r)
            #print(loss)
            loss = loss / CFG.batch_size
            scaler.scale(loss).backward()
            for name, param in model.named_parameters():
                if not param.grad is None:
                    nan_mask = torch.isnan(param.grad)
                    if nan_mask.any():
                        #print(f"nan gradient found: {name}")
                        param.grad[nan_mask] = 0
                        #do_optimize = False
                        #break
            ga_count += 1
            if ga_count == CFG.batch_size:
                if do_optimize:
                    scaler.step(optimizer)
                    scaler.update()
                do_optimize = True
                scheduler.step()
                optimizer.zero_grad()
                train_loss.append(loss.item())
                train_loss2.append([loss2[0].item(), loss2[1].item()])
            ga_count = 0
        if ga_count > 0:
            optimizer.zero_grad()
            scaler.step(optimizer)
            train_loss.append(loss.item())
            train_loss2.append([loss2[0].item(), loss2[1].item()])
            ga_count = 0
        
        train_loss = np.mean(train_loss)
        train_loss2 = np.mean(train_loss2, axis=0)
        elapsed_time = time() - epoch_start
        print(
            f"[epoch {epoch}] train loss: {train_loss: .6f} elapsed_time: {elapsed_time: .3f}")
        #print(f'loss2: {train_loss2}')
        
        if epoch > 0:
            val_loss = []
            val_loss2 = []
            model.eval()
            ycache = []
            tcache = []
            cicache = []
            seicache = []
            for batch in tqdm(val_loader):
                x, rs, r, t, ci, seis = batch
                x = x.to(device)
                r = r.to(device)
                rs = rs.to(device)
                t = t.to(device)
                with torch.no_grad():
                    #y = generate_one_by_one(x, r, model)
                    y = model(x, rs, r)
                #y = y.detach().cpu().to(torch.float32)
                val_loss.append(loss_func_val(y, t, r).item())
                loss2 = loss2_func(y, r)
                val_loss2.append([loss2[0].item(), loss2[1].item()])
                ycache.append(y.detach().cpu().numpy())
                tcache.append(t.detach().cpu().numpy())
                cicache.append([ci]*(len(seis)))
                seicache.append(seis)            
            val_loss = np.mean(val_loss) 
            val_loss2 = np.mean(val_loss2, axis=0)
            ycache = np.concatenate(ycache, axis=0)
            tcache = np.concatenate(tcache, axis=0)
            cicache = np.concatenate(cicache, axis=0)
            seicache = np.concatenate(seicache, axis=0)
            score, error_dict, predicted_df, actual_df = get_score(ycache, tcache, cicache, seicache)
            #val_loss2[1] /= range_penalty_rate
            print(f'valid loss2: {val_loss2}')

            if best_val_score > score:
                best_epoch = epoch
                best_val_score = score
                # print("save model")
                #
                #predicted_df.to_csv(f'fold{val_fold}_predicted_df.csv', index=False)
                #actual_df.to_csv(f'fold{val_fold}_actual_df.csv', index=False)
                #score_df = pd.DataFrame([{'chain_id':k, 'score':error_dict[k]} for k in error_dict])
                #score_df.to_csv(f'fold{val_fold}_score.csv', index=False)

                save_path = str(output_path / f'{CFG.model_name}.pth')
                torch.save(model, save_path)      
                
            print(f'val loss: {val_loss: .6f}')
            if epoch - best_epoch > CFG.es_patience:
                print("Early Stopping!")
                break
    
    return val_fold, best_epoch, best_val_score, model


CFG = MODEL_CONFIGS[0]
DATA_DIR = 'data_cache'
N_FOLDS = 5
LABEL_NAMES = ['x', 'y', 'z', 'qw', 'qx', 'qy', 'qz']
DATA_FOLDERS = [f'{DATA_DIR}/rots/']
def main():
    global CFG
    os.system('mkdir assets/models')
    output_path = Path(f"assets/models")
    output_path.mkdir(exist_ok=True)
    
    train = pd.read_csv(f'{DATA_DIR}/train_labels.csv')
    ranges = pd.read_csv(f'{DATA_DIR}/range.csv')
    types = pd.read_csv('assets/data_type.csv')
    train = train.merge(ranges, how='left', on=['chain_id', 'i'])
    #train = train.groupby('chain_id').apply(lambda group: group.interpolate()).reset_index(drop=True)
    train['range_scaled'] = train['range'].fillna(0)
    mean_range = train['range'].mean()

    def interpolate_func(group):
        assert not np.isnan(group['range'].iloc[0])
        group['range'] = group['range'].fillna(method='ffill')
        return group
    train = train.groupby('chain_id').apply(interpolate_func).reset_index(drop=True)

    #seed_everything(1086)

    score_list = []
    epoch_list = []
    for cfg in MODEL_CONFIGS:
        CFG = cfg
        print('start training: ', CFG.__dict__)
        RANDAM_SEED = CFG.seed + CFG.fold
        seed_everything(RANDAM_SEED)

        train["fold"] = -1
        train["type"] = 'n'
        chain_ids = train['chain_id'].unique()
        np.random.shuffle(chain_ids)
        chain_types = np.array([types[types['chain_id']==ci]['type'].values[0] for ci in chain_ids])
        #sgkf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDAM_SEED)
        #sgkf = GroupKFold(n_splits=N_FOLDS)

        for fold_id, used_types in enumerate(FOLDS):
            val_idx = np.where(np.isin(chain_types, used_types))[0]
            train.loc[train['chain_id'].isin(chain_ids[val_idx]), "fold"] = fold_id
            if CFG.fold == fold_id:
                #print(chain_ids[val_idx][:10])
                print(f'fold{fold_id}: ', np.unique(chain_types[val_idx]))
        
        print(f"training {CFG.model_name}")
        _, best_epoch, best_val_score, _ = train_one_fold(CFG, CFG.fold, train, output_path)
        epoch_list.append(best_epoch)
        score_list.append(best_val_score)
        print('end training: ', CFG.__dict__)
        print(f"score: {best_val_score}")
        print("-----------------------------------------------------------------------")
        
    print(f'avg best epoch: {np.mean(epoch_list)}')
    print(f'avg best score: {np.mean(score_list)}')
    
if __name__ == "__main__":
    main()