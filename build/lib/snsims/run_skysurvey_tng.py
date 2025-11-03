import pandas as pd
import numpy as np
import h5py
#import illustris_python as il
import matplotlib.pyplot as plt
from matplotlib import ticker
#import skysurvey_pw as skysurvey
import skysurvey
import requests
import glob
import os
import multiprocessing as mp

import h5py
from astropy.cosmology import Planck15 as cosmo

from photutils.aperture import CircularAperture,aperture_photometry
from astropy.io import fits

import warnings
from tqdm import tqdm

import pandas as pd
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

def convert_phot(aperture_sum,zp):
    return -2.5*np.log10(aperture_sum) + zp
def convert_phot_fnu(aperture_sum,zp):
    return 10**(np.log10(aperture_sum)-0.4*(zp-8.9))
    
def do_ap_phot(img_path,snx=0,sny=0,pix_list=None,pos_list=None,zp=0,ax=None,band=0,plot_colour='w',plot=False):
    #zp does not matter if we are just doing colours.
    img = fits.getdata(img_path)[band]
    h = fits.getheader(img_path)
    if not pix_list:
        pix_list=[snx,sny]
    if not pos_list:
        pos_list=[snx,sny]
    pixscale_physical = 1000/h['CDELT1']/cosmo.h
    ap = CircularAperture(pix_list,r=pixscale_physical)
    phot_df = aperture_photometry(img, ap,error=0.1*img).to_pandas()
    
    res =convert_phot(phot_df['aperture_sum'],zp)
    res_err = convert_phot(phot_df['aperture_sum_err'],zp)
    return ap,res, res_err


# get d_DLR

def get_DLR_ABT(x_pos, y_pos,  A_IMAGE, B_IMAGE, orientation_ellip, angsep):
    '''Function for calculating the DLR of a galaxy - SN pair (taken from dessne)'''
    #print('Inputs: ',x_pos, y_pos,  A_IMAGE, B_IMAGE, orientation_ellip, angsep)
   
    # angle between RA-axis and SN-host vector
    GAMMA = np.arctan(-y_pos/x_pos)
    #print('GAMMA',GAMMA)
    # angle between semi-major axis of host and SN-host vector
    PHI = orientation_ellip + GAMMA # angle between semi-major axis of host and SN-host vector
    #print('PHI',PHI)
    rPHI = A_IMAGE*B_IMAGE/np.sqrt((A_IMAGE*np.sin(PHI))**2 +
                                     (B_IMAGE*np.cos(PHI))**2)
    #print('rPHI',rPHI)
    # directional light radius
    #  where 2nd moments are bad, set d_DLR = 99.99
    d_DLR = angsep/rPHI

    return d_DLR
def angles(x_pos,y_pos,orientation_ellip):# angle between RA-axis and SN-host vector
    GAMMA = np.arctan(-y_pos/x_pos)
    #print('GAMMA',GAMMA)
    # angle between semi-major axis of host and SN-host vector
    PHI = orientation_ellip + GAMMA # angle between semi-major axis of host and SN-host vector
    print(GAMMA,PHI)
    print(np.rad2deg(GAMMA),np.rad2deg(PHI))

def dtd_pl(age, A=2.11e-13, beta=-1.13, t0=40):
        """Default delay time distribution function."""
        p = ((age*1000 < t0)*0) + ((age*1000>t0)*A * (age**beta))
        #if age*1000 < t0:
        #    return 0
        return p

def get_hostid(hostfunc=dtd_pl,ages=None):
    rates = hostfunc(ages,)
    print(rates)
    ids = np.random.choice(np.arange(len(ages)),p=rates/np.sum(rates))
    print(ids)
    return ids

# abstract it all into a single func

def get_host(ages,A=2.11e-13, beta=-1.13, t0=40):
    p = ((ages*1000 < t0)*0) + ((ages*1000>t0)*A * (ages**beta))
    return np.random.choice(np.arange(len(ages)),p=p/np.sum(p))


def get_metallicity(hostId,galaxy_catalog):
    return galaxy_catalog['metallicity'].iloc[hostId]
def get_progage(hostId,galaxy_catalog):
    return galaxy_catalog['age'].iloc[hostId]
from scipy.special import expit
from scipy.stats import norm
def x1_g24_age_metallicity(**kwargs):
    age =kwargs['age']
    metallicity = kwargs['metallicity']
    mu1,sig1,mu2,sig2,Km,rred,rblue,Kc,agesplit = [i for i in kwargs['params']]
    r = rred+((rblue-rred)*expit((age-agesplit)/Kc))
    
    #r=np.array(r)
    #print(r)
    Zsol=0
    x1low = (Km*(metallicity-Zsol))+mu2
    x1high = (Km*(metallicity-Zsol))+mu1
    norm1 = norm(x1high,sig1)
    norm2 = norm(x1low,sig2)
    choice = (np.random.rand(len(age))>r).astype(int)
    #choice = np.random.choice([0,1],p=[r,1-r])
    #print(choice)
    #print(choice.shape)
    n1rvs = norm1.rvs()
    n2rvs = norm2.rvs()
    #print('norm 1',n1rvs.shape,np.mean(n1rvs))
    #print('norm 2',n2rvs.shape,np.mean(n2rvs))
    x1=((choice==0)*n1rvs )+((choice==1)*n2rvs )
    #print(x1.shape)
    return x1

def age_step(**kwargs):
    age =kwargs['age']
    agesplit = kwargs['agesplit']
    Ka = kwargs['Ka']
    r = expit((age-agesplit)/Ka)
    choice = (np.random.rand(len(age))>r).astype(int)
    #print(age[:10],choice[:10])
    step = ((choice==0)*0.1)+((choice==1)*-0.1)
    #print(step[:10])
    magabs = -19.3+step
    #print(magabs[:10])
    return magabs

def distmod(mobs,x1,c,mabs=-19.3,alpha=-0.14,beta=3.15):
    return mobs-mabs -(alpha*x1 + beta*c)
def hubres(distmod,z,cosmo=None):
    if cosmo==None:
        from astropy.cosmology import Planck18 as cosmo
    return distmod - cosmo.distmod(z).value

    
g24_agemet_params = [0.25,0.55,-1.33,0.63,-0.3,0.18,0.98,-0.128,5]

baseUrl = 'http://www.tng-project.org/api/'
headers = {"api-key":"eba6365a63ff327c63d284349c914245"}
def get(path, params=None,savepath=None):
    # make HTTP GET request to path
    r = requests.get(path, params=params, headers=headers)

    # raise exception if response code is not HTTP SUCCESS (200)
    r.raise_for_status()

    if r.headers['content-type'] == 'application/json':
        return r.json() # parse json responses automatically
    if 'content-disposition' in r.headers:
         if savepath==None:
             savepath=''
         
         filename = savepath+r.headers['content-disposition'].split("filename=")[1]
         #print('writing to file',filename)
         with open(filename, 'wb') as f:
             f.write(r.content)
         return filename # return the filename string
    return r
bases = {'TNG50-1':baseUrl+'TNG50-1/snapshots/99/',
        'TNG100-1':baseUrl+'TNG100-1/snapshots/99/'}
sim = 'TNG100-1'
with h5py.File(f'data/{sim}/morphs_i.hdf5') as f:
    subfind_ids = np.array(f['subfind_id'])
    elongation_asymmetry = np.array(f['elongation_asymmetry'])
    ellipticity_asymmetry = np.array(f['ellipticity_asymmetry'])
    orientation_asymmetry = np.array(f['orientation_asymmetry'])
    rhalf_ellip = np.array(f['rhalf_ellip']) 
    sersic_ellip = np.array(f['sersic_ellip'])
    sersic_n = np.array(f['sersic_n'])
    sersic_rhalf = np.array(f['sersic_rhalf'])
    sersic_theta = np.array(f['sersic_theta'])
    galaxy_morphs = pd.DataFrame(np.array([subfind_ids,  elongation_asymmetry,ellipticity_asymmetry,    orientation_asymmetry,    rhalf_ellip,    sersic_ellip,    sersic_n,sersic_rhalf, sersic_theta]).T,
                                     columns=['subfind_id','elongation_asymmetry', 'ellipticity_asymmetry','orientation_asymmetry','rhalf_ellip','sersic_ellip','sersic_n','sersic_rhalf','sersic_theta',])



def process_subhalo(subhaloId,):
    subhaloId = int(subhaloId)
    snap_fn = f'data/{sim}/{subhaloId}/cutout_{subhaloId}.hdf5'
    
    if not os.path.isfile(snap_fn):
        print(f'{subhaloId} has no snapshot, exiting')
        return None
    
    subinfo = get(bases[sim] + f'subhalos/{subhaloId}')
    
    with h5py.File(snap_fn, 'r') as f:
        stars = f['PartType4']
        star_inds = np.array(stars['GFM_StellarFormationTime']) > 0
        metallicity = np.log10(stars['GFM_Metallicity'][star_inds] / 0.0127)
        initial_mass = stars['GFM_InitialMass'][star_inds] * 1e10 / cosmo.h
        stellar_mass = stars['Masses'][star_inds] * 1e10 / cosmo.h
        Ur = stars['GFM_StellarPhotometrics'][star_inds, 0] - (stars['GFM_StellarPhotometrics'][star_inds, 5] - 0.16)
        V = stars['GFM_StellarPhotometrics'][star_inds, 2]
        gz = stars['GFM_StellarPhotometrics'][star_inds, 4] - stars['GFM_StellarPhotometrics'][star_inds, 7]
        x = stars['Coordinates'][star_inds, 0] - subinfo['pos_x']
        y = stars['Coordinates'][star_inds, 1] - subinfo['pos_y']
        stfs = np.array(stars['GFM_StellarFormationTime'][star_inds])
        
        if not os.path.isfile(f'data/{sim}/{subhaloId}/{subhaloId}_ages.dat'):
            print(f'Calculating ages for {subhaloId}')
            zs = (1 / stfs) - 1
            t = cosmo.lookback_time(zs)
            t = np.clip(t.value, a_min=0.05, a_max=None)
            np.savetxt(f'data/{sim}/{subhaloId}/{subhaloId}_ages.dat', t)
    
    ages = np.loadtxt(f'data/{sim}/{subhaloId}/{subhaloId}_ages.dat', dtype=float)
    if len(ages) > 0:
        n_samp = max(1, int(len(ages) / minlen))
        galaxy_catalog = pd.DataFrame(np.array([ages, metallicity, initial_mass, stellar_mass, V, Ur, gz, x, y]).T,
                                      columns=['age', 'metallicity', 'initial_mass', 'mass', 'V', 'U-r', 'g-z', 'x_pos', 'y_pos'])
        
        masses = galaxy_catalog['initial_mass'].values
        A = 2.11e-13
        beta = -1.13
        t0 = 40
        # Only compute once
        valid = (ages * 1000 > t0)
        DTD_weights = np.zeros_like(ages)
        DTD_weights[valid] = (masses[valid] * A * (ages[valid] ** beta))

        # Normalize
        DTD_weights /= DTD_weights.sum()
        full_model = {"hostId":{"func":np.random.choice,"kwargs":{"a":np.arange(len(ages)),
                            "p":DTD_weights},
                              "as":"hostId"},
               'hostmet':{"func":get_metallicity,"kwargs":{"hostId":"@hostId","galaxy_catalog":galaxy_catalog},'as':"hostmet"},
               'progage':{"func":get_progage,"kwargs":{"hostId":"@hostId","galaxy_catalog":galaxy_catalog},'as':"progage"},
               "x1":{"func":x1_g24_age_metallicity,"kwargs":{"age":"@progage","metallicity":"@hostmet","params":g24_agemet_params},},
               "mabs":{"func":age_step,"kwargs":{"age":"@progage","agesplit":3,"Ka":-0.128},}
              }
        #tqdm.write(f'Sampling {n_samp} SNe from {subhaloId}')
        snia = skysurvey.SNeIa.from_draw(size=n_samp, model=full_model, magabs={"mabs": "@mabs"})
        drawn_gals = galaxy_catalog.iloc[snia.data['hostId']]
        sim_data = snia.data.join(drawn_gals.reset_index(drop=True))
        sim_data['MURES'] = hubres(distmod(snia.data['magobs'], snia.data['x1'], snia.data['c']), snia.data['z'])
        sim_data['MUERR'] = 0.1 # TODO: get real errors
        sim_data['globalmass'] = galmeta.loc[subhaloId]['mass_stars']
        sim_data['globalsfr'] = galmeta.loc[subhaloId]['sfr']
        sim_data['globalssfr'] = galmeta.loc[subhaloId]['ssfr']
        try:
            img_path = f'data/{sim}/{subhaloId}/'
            if not os.path.isfile(img_path):
                get(subinfo['supplementary_data']['skirt_images']['fits_pogs'],savepath=img_path)
            hdul = fits.open(f'data/{sim}/{subhaloId}/broadband_{subhaloId}.fits')
            h = hdul[0].header
        except:
            #tqdm.write(f'{subhaloId} has no image, exiting')
            return sim_data
        

        sim_data['x_pix'] = (sim_data['x_pos'] * 1000 / h['CDELT1'] / cosmo.h) + (h['NAXIS1'] / 2)
        sim_data['y_pix'] = (sim_data['y_pos'] * 1000 / h['CDELT2'] / cosmo.h) + (h['NAXIS2'] / 2)
        sim_data['r_pix'] = np.sqrt((sim_data['x_pos'] * 1000 / h['CDELT1'] / cosmo.h) ** 2 + (sim_data['y_pos'] * 1000 / h['CDELT2'] / cosmo.h) ** 2)
        try:
            this_subhalo_morph = galaxy_morphs.iloc[np.where(subfind_ids==subhaloId)[0]]
            A_IMAGE =this_subhalo_morph['rhalf_ellip'].values*(1/0.68)
            B_IMAGE = this_subhalo_morph['rhalf_ellip'].values*(1/0.68)/this_subhalo_morph['elongation_asymmetry'].values[0]
            sim_data['d_DLR'] = get_DLR_ABT(sim_data['x_pos'].values,sim_data['y_pos'].values,A_IMAGE,B_IMAGE,this_subhalo_morph['orientation_asymmetry'].values[0],sim_data['r_pix'].values)
        except:
            print(f'{subhaloId} has no morphs, returning NaN for dDLR')
            sim_data['d_DLR'] = np.nan
        pos_list = sim_data[['x_pos', 'y_pos']].values.tolist()
        pix_list = sim_data[['x_pix', 'y_pix']].values.tolist()
        band_mags = {}
        for counter, band in enumerate(['g', 'r', 'i', 'z']):
            apps, mags, magerrs = do_ap_phot(f'data/{sim}/{subhaloId}/broadband_{subhaloId}.fits',
                                             pos_list=pos_list,
                                             pix_list=pix_list,
                                             band=counter, plot=False)
            band_mags[band] = mags
        sim_data['localrestframe_gz'] = band_mags['g'].values - band_mags['z'].values
        sim_data['localrestframe_gz'] = band_mags['g'].values - band_mags['z'].values
        
        #print(f'{subhaloId} Done')
        sim_data.to_hdf(f'simout/{subhaloId}_skysurvey_out.h5', key='main',mode='w')
        return sim_data
galmeta=pd.read_csv(f'data/{sim}/galmeta.csv',index_col=0)
galmeta['mass_stars_true'] = galmeta['mass_stars']*1E10/ 0.6774
galmeta_mcut = galmeta[galmeta['mass_stars_true']>5E9]
idx =galmeta_mcut.mass_stars.idxmin()
subhaloId = int(idx)
snap_fn =f'data/{sim}/{subhaloId}/cutout_{subhaloId}.hdf5'

with h5py.File(snap_fn) as f:
    stars=f['PartType4']
    star_inds = np.array(stars['GFM_StellarFormationTime'])>0
    metallicity = np.log10(stars['GFM_Metallicity'][star_inds]/0.0127)
    initial_mass = stars['GFM_InitialMass'][star_inds]*1e10/cosmo.h
    stellar_mass = stars['Masses'][star_inds]*1e10/cosmo.h
    minlen = len(stellar_mass)   
if __name__=="__main__":
    import sys
    try:
        n = int(sys.argv[1])
    except IndexError: 
        n = len(galmeta_mcut)
    num_cores = min(mp.cpu_count(), 16)
    subhalo_ids = galmeta_mcut.index.values[:n-1]

    with mp.Pool(num_cores) as pool:
        results = list(tqdm(pool.imap_unordered(process_subhalo, subhalo_ids), total=len(subhalo_ids),leave=True,position=0))

    all_sim = pd.concat([r for r in results if r is not None])
    all_sim.to_hdf(f'simout/{sim}_skysurvey_out.h5', key='main',mode='w')