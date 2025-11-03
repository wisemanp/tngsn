import pandas as pd
from utils.utils import get
import glob
from tqdm import tqdm
import sys
sim = sys.argv[1]
baseUrl = 'http://www.tng-project.org/api/'
bases = {'TNG50-1':baseUrl+'TNG50-1/snapshots/99/',
        'TNG100-1':baseUrl+'TNG100-1/snapshots/99/'}

star_formers = {}
star_bursts = {}
passives = {}

dirs = glob.glob(f'data/{sim}/*/')

for dirname in tqdm(dirs):
    subhaloId = dirname.split('/')[-2]
    thissub =get(bases[sim]+f'subhalos/{subhaloId}')
    
    mass_stars = thissub['mass_stars']
    
    if mass_stars>0:
        sfr = thissub['sfr']
        ssfr = sfr/mass_stars
        if ssfr>10:
            star_bursts[subhaloId]=[mass_stars,sfr,ssfr]
        elif ssfr <0.1:
            passives[subhaloId]=[mass_stars,sfr,ssfr]
        else:
            star_formers[subhaloId]=[mass_stars,sfr,ssfr]

SFmeta = pd.DataFrame(star_formers)
SFmeta = SFmeta.T
SFmeta.rename(columns={0:'mass_stars',1:'sfr',2:'ssfr'},inplace=True)

SBmeta = pd.DataFrame(star_bursts)
SBmeta = SBmeta.T
SBmeta.rename(columns={0:'mass_stars',1:'sfr',2:'ssfr'},inplace=True)

passivemeta = pd.DataFrame(passives)
passivemeta = passivemeta.T
passivemeta.rename(columns={0:'mass_stars',1:'sfr',2:'ssfr'},inplace=True)

galmeta = pd.concat([SFmeta,SBmeta,passivemeta])
galmeta.to_csv(f'data/{sim}/galmeta.csv')