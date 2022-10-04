from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import scipy as sp

from copy import deepcopy


# =============================================================================
# SCENE
# =============================================================================

class scene():
    
    def __init__(self,
                 path,
                 time=None,
                 wave=None):
        
        # Load scene from Haystacks.
        hdul = pyfits.open(path)
        N_EXT = hdul[0].header['N_EXT']
        
        # Load times.
        self.time = np.array([0.])
        self.Ntime = len(self.time)
        
        # Load wavelengths.
        waves = hdul[N_EXT+1].data # micron
        Nwaves = len(waves)
        if (wave is None):
            self.wave = waves # micron
        else:
            self.wave = wave # micron
        self.Nwave = len(self.wave)
        
        # Load star.
        self.angdiam = 0.465 # mas
        self.pixscale = hdul[0].header['PIXSCALE']/hdul[0].header['DIST']*1e3 # mas
        self.xystar = np.array([1666, 1666])[np.newaxis, :] # pix
        fstar = np.array(hdul[N_EXT+2].data)[np.newaxis, :] # Jy
        fstar_interp = sp.interpolate.interp1d(waves, fstar[0], kind='cubic')
        self.fstar = fstar_interp(self.wave)[np.newaxis, :] # Jy
        
        # Load disk.
        self.diskwaves = waves # micron
        self.Ndiskwaves = len(self.diskwaves)
        disk = []
        for i in range(N_EXT):
            disk += [hdul[i+1].data]
        disk = np.array(disk)
        
        # Interpolate in log-space to avoid negative values.
        inds = np.searchsorted(self.diskwaves, self.wave)-1
        fracinds = (inds+(np.log(self.wave)-np.log(self.diskwaves[inds]))/(np.log(self.diskwaves[inds+1])-np.log(self.diskwaves[inds])))
        disk_interp = sp.interpolate.interp1d(np.arange(self.Ndiskwaves), disk, kind='cubic', axis=0)
        disk = disk_interp(fracinds)
        self.disk = deepcopy(disk)[np.newaxis, :] # Jy/pix (microns, pix, pix)
        del disk
        
        # Close scene from exoVista.
        hdul.close()
        
        pass
