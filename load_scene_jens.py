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


# =============================================================================
# SCENE
# =============================================================================

class scene():
    
    def __init__(self,
                 path,
                 time=None,
                 wave=None):
        
        # Load scene from exoVista.
        hdul = pyfits.open(path)
        
        # Load times.
        temp = hdul[3].data
        if (len(temp.shape) == 1):
            temp = temp[np.newaxis, :]
        times = np.array(temp[:, 0]) # yr
        Ntimes = len(times)
        if ((time is None) or (Ntimes == 1)):
            self.time = np.array([times[0]]) # yr
        else:
            self.time = time # yr
        self.Ntime = len(self.time)
        
        # Load wavelengths.
        waves = hdul[0].data # micron
        Nwaves = len(waves)
        if (wave is None):
            self.wave = waves # micron
        else:
            self.wave = wave # micron
        self.Nwave = len(self.wave)
        
        # Load star.
        self.angdiam = hdul[3].header['ANGDIAM'] # mas
        self.pixscale = hdul[3].header['PXSCLMAS'] # mas
        if (self.Ntime == 1):
            self.xystar = np.array(temp[0, 1:3])[np.newaxis, :] # pix
            self.bposstar = np.array(temp[0, 9:12])[np.newaxis, :] # au
            self.bvelstar = np.array(temp[0, 12:15])[np.newaxis, :] # au/yr
            fstar = np.array(temp[0, 15:])[np.newaxis, :] # Jy
            fstar_interp = sp.interpolate.interp1d(waves, fstar[0], kind='cubic')
            self.fstar = fstar_interp(self.wave)[np.newaxis, :] # Jy
            self.fstar_V = fstar_interp(0.540) # Jy, Johnson V-band
        else:
            xystar = np.array(temp[:, 1:3]) # pix
            xystar_x_interp = sp.interpolate.interp1d(times, xystar[:, 0], kind='cubic')
            xystar_y_interp = sp.interpolate.interp1d(times, xystar[:, 1], kind='cubic')
            self.xystar = np.array([xystar_x_interp(self.time), xystar_y_interp(self.time)]).T # pix
            bposstar = np.array(temp[:, 9:12]) # au
            bposstar_x_interp = sp.interpolate.interp1d(times, bposstar[:, 0], kind='cubic')
            bposstar_y_interp = sp.interpolate.interp1d(times, bposstar[:, 1], kind='cubic')
            bposstar_z_interp = sp.interpolate.interp1d(times, bposstar[:, 2], kind='cubic')
            self.bposstar = np.array([bposstar_x_interp(self.time), bposstar_y_interp(self.time), bposstar_z_interp(self.time)]).T # au
            bvelstar = np.array(temp[:, 12:15]) # au/yr
            bvelstar_x_interp = sp.interpolate.interp1d(times, bvelstar[:, 0], kind='cubic')
            bvelstar_y_interp = sp.interpolate.interp1d(times, bvelstar[:, 1], kind='cubic')
            bvelstar_z_interp = sp.interpolate.interp1d(times, bvelstar[:, 2], kind='cubic')
            self.bvelstar = np.array([bvelstar_x_interp(self.time), bvelstar_y_interp(self.time), bvelstar_z_interp(self.time)]).T # au/yr
            fstar = np.array(temp[:, 15:]) # Jy
            fstar_interp = sp.interpolate.interp2d(waves, times, fstar, kind='quintic')
            self.fstar = fstar_interp(self.wave, self.time) # Jy
            self.fstar_V = fstar_interp(0.540, 0.) # Jy, Johnson V-band
        self.dstar = hdul[3].header['DIST'] # pc
        self.Lstar = hdul[3].header['LSTAR'] # Lsun
        self.Vstar = hdul[3].header['M_V'] # absmag
        
        # Load planets.
        self.Nplanets = len(hdul)-4
        self.xyplanet = np.zeros((self.Nplanets, self.Ntime, 2)) # pix
        self.orbplanet = np.zeros((self.Nplanets, self.Ntime, 6)) # sma (au), ecc, inc (deg), pan (deg), aop (deg), man (deg)
        self.bposplanet = np.zeros((self.Nplanets, self.Ntime, 3)) # au
        self.bvelplanet = np.zeros((self.Nplanets, self.Ntime, 3)) # au/yr
        self.fplanet = np.zeros((self.Nplanets, self.Ntime, self.Nwave)) # Jy
        for i in range(self.Nplanets):
            temp = hdul[4+i].data
            if (len(temp.shape) == 1):
                temp = temp[np.newaxis, :]
            if (self.Ntime == 1):
                self.xyplanet[i, 0] = temp[0, 1:3] # pix
                self.orbplanet[i, 0] = temp[0, 3:9] # sma (au), ecc, inc (deg), pan (deg), aop (deg), man (deg)
                self.bposplanet[i, 0] = temp[0, 9:12] # au
                self.bvelplanet[i, 0] = temp[0, 12:15] # au/yr
                fplanet_interp = sp.interpolate.interp1d(waves, np.multiply(temp[0, 15:], fstar[0]), kind='cubic')
                self.fplanet[i, 0] = fplanet_interp(self.wave)[np.newaxis, :] # Jy
            else:
                xyplanet_x_interp = sp.interpolate.interp1d(times, temp[:, 1], kind='cubic')
                xyplanet_y_interp = sp.interpolate.interp1d(times, temp[:, 2], kind='cubic')
                self.xyplanet[i] = np.array([xyplanet_x_interp(self.time), xyplanet_y_interp(self.time)]).T # pix
                orbplanet_a_interp = sp.interpolate.interp1d(times, temp[:, 3], kind='cubic')
                orbplanet_e_interp = sp.interpolate.interp1d(times, temp[:, 4], kind='cubic')
                orbplanet_i_interp = sp.interpolate.interp1d(times, temp[:, 5], kind='cubic')
                orbplanet_p_interp = sp.interpolate.interp1d(times, temp[:, 6], kind='cubic')
                orbplanet_o_interp = sp.interpolate.interp1d(times, temp[:, 7], kind='cubic')
                orbplanet_m_interp = sp.interpolate.interp1d(times, temp[:, 8], kind='cubic')
                self.orbplanet[i] = np.array([orbplanet_a_interp(self.time), orbplanet_e_interp(self.time), orbplanet_i_interp(self.time), orbplanet_p_interp(self.time), orbplanet_o_interp(self.time), orbplanet_m_interp(self.time)]).T # sma (au), ecc, inc (deg), pan (deg), aop (deg), man (deg)
                bposplanet_x_interp = sp.interpolate.interp1d(times, temp[:, 9], kind='cubic')
                bposplanet_y_interp = sp.interpolate.interp1d(times, temp[:, 10], kind='cubic')
                bposplanet_z_interp = sp.interpolate.interp1d(times, temp[:, 11], kind='cubic')
                self.bposplanet[i] = np.array([bposplanet_x_interp(self.time), bposplanet_y_interp(self.time), bposplanet_z_interp(self.time)]).T # au
                bvelplanet_x_interp = sp.interpolate.interp1d(times, temp[:, 12], kind='cubic')
                bvelplanet_y_interp = sp.interpolate.interp1d(times, temp[:, 13], kind='cubic')
                bvelplanet_z_interp = sp.interpolate.interp1d(times, temp[:, 14], kind='cubic')
                self.bvelplanet[i] = np.array([bvelplanet_x_interp(self.time), bvelplanet_y_interp(self.time), bvelplanet_z_interp(self.time)]).T # au/yr
                fplanet_interp = sp.interpolate.interp2d(waves, times, np.multiply(temp[:, 15:], fstar), kind='quintic')
                self.fplanet[i] = fplanet_interp(self.wave, self.time) # Jy
        
        # Load disk.
        self.diskwaves = hdul[1].data # micron
        self.Ndiskwaves = len(self.diskwaves)
        disk = hdul[2].data[:-1]
        
        # Interpolate in log-space to avoid negative values.
        inds = np.searchsorted(self.diskwaves, self.wave)-1
        fracinds = (inds+(np.log(self.wave)-np.log(self.diskwaves[inds]))/(np.log(self.diskwaves[inds+1])-np.log(self.diskwaves[inds])))
        disk_interp = sp.interpolate.interp1d(np.arange(self.Ndiskwaves), disk, kind='cubic', axis=0)
        disk = disk_interp(fracinds)
        self.disk = np.zeros((self.Ntime, self.Nwave, disk.shape[1], disk.shape[2]))
        disk = disk.T
        for i in range(self.Ntime):
            self.disk[i] = np.multiply(disk, self.fstar[i]).T # Jy/pix (microns, pix, pix)
        
        # Close scene from exoVista.
        hdul.close()
        
        pass
    
    def replace_disk(self,
                     Nzodi=1.):
        
        # plt.figure()
        # plt.imshow(np.log10(self.disk[0, 0]), origin='lower')
        # plt.colorbar()
        # plt.show()
        
        # Compute angular separation of HZ.
        a_HZ = 1.*np.sqrt(self.Lstar/1.) # au
        r_HZ = a_HZ/self.dstar # arcsec
        
        # Make face-on 1/r^2 disk.
        ramp = np.arange(self.disk.shape[2])-self.disk.shape[2]/2.+0.5
        xx, yy = np.meshgrid(ramp, ramp)
        disk = 1./(xx**2+yy**2)
        
        # Compute disk flux at V-band.
        F0 = 3781 # Jy, Johnson V-band zero point
        Vmag_sun = 4.83 # absmag
        Vmag_1zodi = 22. # absmag/arcsec^2
        Vflux_1zodi = F0*10.**(-Vmag_1zodi/2.5) # Jy/arcsec^2
        Vflux_Nzodi = Vflux_1zodi*Nzodi*10.**(-(self.Vstar-Vmag_sun)/2.5) # Jy/arcsec^2
        
        # Compute disk flux at respective wavelength.
        Xflux_Nzodi = Vflux_Nzodi*self.fstar/self.fstar_V # Jy/arcsec^2
        Xflux_Nzodi *= (self.pixscale/1e3)**2 # Jy/pix
        
        # Renormalize disk flux at angular separation of HZ.
        self.disk = np.zeros(self.disk.shape)
        for i in range(self.disk.shape[0]):
            for j in range(self.disk.shape[1]):
                rnfact = Xflux_Nzodi[i, j]*(r_HZ/(self.pixscale/1e3))**2
                self.disk[i, j] = disk.copy()*rnfact
        
        # plt.figure()
        # plt.imshow(np.log10(self.disk[0, 0]), origin='lower')
        # plt.colorbar()
        # plt.show()
        
        pass
