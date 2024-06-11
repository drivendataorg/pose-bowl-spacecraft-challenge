from pathlib import Path
import click
from helper import *

@click.command()
@click.argument(
    "data_dir",
    type=click.Path(exists=True, file_okay=False, dir_okay=True),
)
@click.argument(
    "output_path",
    type=click.Path(exists=False),
)
def main(data_dir, output_path):
    data_dir = Path(data_dir).resolve()
    output_path = Path(output_path).resolve()
    
    # load ranges
    ranges = pd.read_csv(f'{data_dir}/range.csv')    
    #print(ranges)
    
    # load model
    model_paths = glob('assets/models/*.pth')
    models = [torch.load(p, map_location=torch.device('cpu')).eval() for p in model_paths]

    val_dataset = PE_Dataset(ranges, [f'{data_dir}/images/'], models)
    
    ycache = []
    tcache = []
    cicache = []
    seicache = []
    vrcache = []
    for data in tqdm(val_dataset):
        y, ci, seis, valid_rate = data
        ycache.append(y)
        cicache.append([ci]*(len(seis)))
        seicache.append(seis)
        vrcache.append(valid_rate)
        print('past one')
    ycache = np.concatenate(ycache, axis=0)
    cicache = np.concatenate(cicache, axis=0)
    seicache = np.concatenate(seicache, axis=0)
    
    vrcache = np.mean(vrcache)
    #assert vrcache > 0.98
    
    submission = get_pred_df(ycache, cicache, seicache)
    submission.to_csv(output_path, index=False)
    
if __name__ == "__main__":
    main()