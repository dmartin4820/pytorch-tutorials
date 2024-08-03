import os
from pathlib import Path

outdir = "data/celeba"
dirname = "/mnt/c/Users/l/Downloads/CelebA/unzipped/CelebA_Spoof/Data/train"
outpath = Path(outdir)
dirpath = Path(dirname)

for dir in dirpath.iterdir():
  outpath_sub = Path(outpath/f'{dir.stem}')
  if not outpath_sub.exists():
    #E.g. data/celeba/1, data/celeba/2, etc.
    outpath_sub.mkdir()

    
    fnjpgs = sorted(Path(dir/'live').glob('*.jpg'))
    for fnjpg in fnjpgs:
      Path(outpath_sub/f'{fnjpg.stem}.jpg').symlink_to(fnjpg)


