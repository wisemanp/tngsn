import requests
import sys
import os
import numpy as np
from tqdm import tqdm
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
if sys.argv[1]:
    sim = sys.argv[1]
else:
    sim = 'TNG100-1'

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