from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import matplotlib.patheffects as PathEffects
import os
import sys

from scipy.ndimage import shift, zoom
from scipy.optimize import leastsq
from scipy.special import lambertw

sys.path.append('/Users/jkammerer/Documents/Code/emccd_detect')
from emccd_detect.emccd_detect import EMCCDDetect
from photon_count.corr_photon_count import get_count_rate, get_counts_uncorrected

import util

rad2mas = 180./np.pi*3600.*1000.
mas2rad = np.pi/180./3600./1000.


# =============================================================================
# DETECTOR
# =============================================================================

class det():
    
    def __init__(self,
                 iwa=8.5,
                 owa=26.):
        
        print('Initializing detector...')
        
        # Simulation parameters.
        self.iwa = iwa # lambda/D
        self.owa = owa # lambda/D
        # self.iwa = 8.5 # lambda/D
        # self.owa = 26. # lambda/D
        self.iwa = 2.5 # lambda/D
        self.owa = 26. # lambda/D
        # self.imsz = 100
        self.imsz = 250
        self.imsc_scene = 2. # mas
        self.fill = 1e-100
        self.gain = 5000. # e-/ph, electron multiplying gain
        self.fw_img = 60000. # e-, image area full well capacity
        self.fw_ser = 100000. # e-, serial register full well capacity
        # self.dc = 0. # e-/pix/s, dark current
        self.dc = 3e-5 # e-/pix/s, dark current
        # self.cic = 0. # e-/pix/frame, clock-induced charge
        self.cic = 1.3e-3 # e-/pix/frame, clock-induced charge
        self.rn = 0. # e-/pix/frame
        self.bias = 10000. # e-, bias
        # self.qe = 1. # quantum efficiency
        self.qe = 0.9 # quantum efficiency
        self.crr = 0. # hits/cm^2/s, cosmic ray rate
        self.prob = 0.99 # probability to detect photon
        
        pass
    
    def simpol(self,
               x0,
               l0,
               rho,
               phi,
               sgn,
               sym):
        
        pol_r = 0.
        pol_p = 0.
        for i in range(len(l0)):
            if (l0[i][0:2] == 'r-'):
                pol_r += x0[i]*rho**float(l0[i][2:])
            elif (l0[i][0:2] == 'p+'):
                pol_p += x0[i]*phi**float(l0[i][2:])
        
        disk = pol_p/pol_r
        if ((sym == 'odd') or (sym == 'both')):
            disk *= sgn
        
        return disk
    
    def chi_simpol(self,
                   x0,
                   l0,
                   xx,
                   yy,
                   inc, # deg
                   Omega, # deg
                   sym,
                   imgs,
                   mask,
                   errs=None):
        
        if (('inc' in l0) and ('Omega' in l0)):
            ww_inc = np.where(l0 == 'inc')[0][0]
            inc = x0[ww_inc]
            ww_Omega = np.where(l0 == 'Omega')[0][0]
            Omega = x0[ww_Omega]
        elif ('inc' in l0):
            ww_inc = np.where(l0 == 'inc')[0][0]
            inc = x0[ww_inc]
        elif ('Omega' in l0):
            ww_Omega = np.where(l0 == 'Omega')[0][0]
            Omega = x0[ww_Omega]
        
        if ((np.abs(inc) > 90.) or (np.abs(Omega) > 180.)):
            
            return np.ones_like(imgs[mask])*np.inf
        
        else:
            if (sym == 'even'):
                rho, phi, sgn = util.proj_sa(xx, yy, inc, Omega)
            elif (sym == 'odd'):
                rho, phi, sgn = util.proj_pc(xx, yy, inc, Omega)
            elif (sym == 'both'):
                rho, phi, sgn = util.proj_qc(xx, yy, inc, Omega)
            disk = self.simpol(x0,
                               l0,
                               rho[mask],
                               phi[mask],
                               sgn[mask],
                               sym)
            
            if (errs is None):
                return np.abs(imgs[mask]-disk)
            else:
                return np.abs(imgs[mask]-disk)/errs[mask]
    
    def nlimgs(self,
               name,
               odir,
               tags,
               overwrite=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if detector images were already computed.
            ofile = odir+name+'/DET/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing detector images...')
                
                # Load raw images.
                ifile = odir+name+'/RAW/'+tag+'.fits'
                hdul = pyfits.open(ifile)
                if (np.min(hdul[0].data) < self.fill):
                    raise UserWarning('Input image cannot have negative pixels')
                time = hdul['TIME'].data # yr
                Ntime = time.shape[0]
                wave = hdul['WAVE'].data*1e-6 # m
                Nwave = wave.shape[0]
                
                # Compute image scale (0.5 lambda/D at 0.5 micron and 90% circumscribed primary mirror diameter)
                self.imsc = 0.5*0.5e-6/(0.9*hdul[0].header['DIAM'])*rad2mas # mas
                
                # Compute wavelength-dependent zoom factor.
                fact = hdul[0].header['PIXSCALE']*wave/hdul[0].header['DIAM']*rad2mas/self.imsc
                norm = np.sum(hdul[0].data, axis=(2, 3)) # ph/s
                
                # Go through time and wavelength steps.
                imgs = np.zeros((Ntime, Nwave, self.imsz, self.imsz)) # ph/s
                for i in range(Ntime):
                    for j in range(Nwave):
                        
                        # Scale image to imsc.
                        temp = np.exp(zoom(np.log(hdul[0].data[i, j]), fact[j], mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                        temp *= norm[i, j]/np.sum(temp) # ph/s
                        
                        # Center image so that (imsz-1)/2 is center.
                        if (((temp.shape[0] % 2 == 0) and (self.imsz % 2 != 0)) or ((temp.shape[0] % 2 != 0) and (self.imsz % 2 == 0))):
                            temp = np.pad(temp, ((0, 1), (0, 1)), mode='edge')
                            temp = np.exp(shift(np.log(temp), (0.5, 0.5), order=5)) # interpolate in log-space to avoid negative values
                            temp = temp[1:-1, 1:-1]
                        
                        # Crop image to imsz.
                        if (temp.shape[0] > self.imsz):
                            nn = (temp.shape[0]-self.imsz)//2
                            temp = temp[nn:-nn, nn:-nn]
                        else:
                            nn = (self.imsz-temp.shape[0])//2
                            temp = np.pad(temp, ((nn, nn), (nn, nn)), mode='edge')
                        imgs[i, j] = temp
                
                # Save detector images.
                hdul[0].data = imgs # ph/s
                hdul[0].header['IMSZ'] = self.imsz # pix
                hdul[0].header['IMSC'] = self.imsc # mas
                path = odir+name+'/DET/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(ofile, output_verify='fix', overwrite=True)
                hdul.close()
        
        pass
    
    def nlphot(self,
               path,
               papt, # pix
               rapt, # lambda/D
               oversample=100,
               time_comp=0., # yr
               wave_comp=0.5, # micron
               fdir=None):
        
        # Load detector images.
        hdul = pyfits.open(path)
        if (np.min(hdul[0].data) < self.fill):
            raise UserWarning('Input image cannot have negative pixels')
        ww_time = np.argmin(np.abs(hdul['TIME'].data-time_comp))
        ww_wave = np.argmin(np.abs(hdul['WAVE'].data-wave_comp))
        imgs = hdul[0].data[ww_time, ww_wave].astype(float)
        imsz = hdul[0].header['IMSZ'] # pix
        imsc = hdul[0].header['IMSC'] # mas
        diam = hdul[0].header['DIAM'] # m
        time = hdul['TIME'].data[ww_time] # yr
        wave = hdul['WAVE'].data[ww_wave]*1e-6 # m
        
        # Compute aperture position and radius in mas.
        pos_mas = papt*self.imsc_scene # mas
        pos_mas[0] *= -1. # mas
        rad_mas = rapt*wave/diam*rad2mas # mas
        
        # Compute aperture position and radius on subarray in pixels.
        Npix = int(np.ceil(3*rad_mas/imsc))
        pos_pix = (pos_mas/imsc+(imsz-1)/2.).astype(int)
        subarr = np.fliplr(imgs)[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
        pos_subarr = (pos_mas/imsc+(imsz-1)/2.)-pos_pix+Npix
        rad_subarr = rad_mas/imsc
        
        # Compute aperture position and radius on oversampled subarray in pixels.
        norm = np.sum(subarr)
        subarr_zoom = zoom(subarr, oversample, mode='nearest', order=5)
        subarr_zoom *= norm/np.sum(subarr_zoom)
        pos_subarr_zoom = pos_subarr*oversample+(oversample-1.)/2.
        rad_subarr_zoom = rad_subarr*oversample
        
        # Compute aperture on oversampled subarray in pixels.
        ramp = np.arange(subarr_zoom.shape[0])
        xx, yy = np.meshgrid(ramp, ramp)
        aptr = np.sqrt((xx-pos_subarr_zoom[0])**2+(yy-pos_subarr_zoom[1])**2) <= rad_subarr_zoom
        
        # Compute aperture count rate.
        cr = np.sum(subarr_zoom[aptr])
        
        # Compute mask.
        ramp = np.arange(imsz)-(imsz-1)/2. # pix
        xx, yy = np.meshgrid(ramp, ramp) # pix
        dist = np.sqrt(xx**2+yy**2) # pix
        iwa = self.iwa*wave/diam*rad2mas/imsc # pix
        owa = self.owa*wave/diam*rad2mas/imsc # pix
        mask = (dist > iwa) & (dist < owa)
        mask_subarr = np.fliplr(mask)[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
        mask_subarr_zoom = zoom(mask_subarr, oversample, mode='nearest', order=0)
        
        # Plot.
        if (True):
            ext = imsz/2.*imsc # mas
            f, ax = plt.subplots(1, 4, figsize=(4.8*4, 3.6*1))
            temp0 = imgs.copy()
            temp0[mask < 0.5] = np.nan
            temp0[temp0 == 0.] = np.nan
            vmin0 = np.log10(np.nanmin(temp0))
            vmax0 = np.log10(np.nanmax(temp0))
            p0 = ax[0].imshow(np.fliplr(np.log10(temp0)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
            c0 = plt.colorbar(p0, ax=ax[0])
            c0.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
            a0 = plt.Circle(pos_mas, rad_mas, fc='none', ec='red')
            ax[0].add_patch(a0)
            ax[0].invert_xaxis()
            ax[0].set_xlabel('$\Delta$RA [mas]')
            ax[0].set_ylabel('$\Delta$DEC [mas]')
            ax[0].set_title('Scene')
            temp1 = np.abs(subarr.copy())
            temp1[mask_subarr < 0.5] = np.nan
            temp1[temp1 == 0.] = np.nan
            vmin1 = np.log10(np.nanmin(temp1))
            vmax1 = np.log10(np.nanmax(temp1))
            p1 = ax[1].imshow(np.log10(temp1), origin='lower', vmin=vmin1, vmax=vmax1)
            c1 = plt.colorbar(p1, ax=ax[1])
            c1.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
            a1 = plt.Circle(pos_subarr, rad_subarr, fc='none', ec='red')
            ax[1].add_patch(a1)
            ax[1].invert_xaxis()
            ax[1].set_xlabel('$\Delta$RA [pix]')
            ax[1].set_ylabel('$\Delta$DEC [pix]')
            ax[1].set_title('PSF')
            temp2 = np.abs(subarr_zoom.copy())
            temp2[mask_subarr_zoom < 0.5] = np.nan
            temp2[temp2 == 0.] = np.nan
            p2 = ax[2].imshow(np.log10(temp2), origin='lower', vmin=vmin1-2.*np.log10(oversample), vmax=vmax1-2.*np.log10(oversample))
            c2 = plt.colorbar(p2, ax=ax[2])
            c2.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
            a2 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
            ax[2].add_patch(a2)
            ax[2].invert_xaxis()
            ax[2].set_xlabel('$\Delta$RA [pix]')
            ax[2].set_ylabel('$\Delta$DEC [pix]')
            ax[2].set_title('Oversampled PSF')
            p3 = ax[3].imshow(aptr, origin='lower')
            c3 = plt.colorbar(p3, ax=ax[3])
            c3.set_label('Transmission', rotation=270, labelpad=20)
            a3 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
            ax[3].add_patch(a3)
            text = ax[3].text(aptr.shape[0]-1., 0., 'CR = %.3e ph/s' % cr, color='white', ha='left', va='bottom', size=10)
            text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
            ax[3].invert_xaxis()
            ax[3].set_xlabel('$\Delta$RA [pix]')
            ax[3].set_ylabel('$\Delta$DEC [pix]')
            ax[3].set_title('Oversampled aperture')
            plt.tight_layout()
            temp = fdir+path[path.find('/'):path.rfind('/')+1]
            if (not os.path.exists(temp)):
                os.makedirs(temp)
            # plt.savefig(temp+path[path.rfind('/')+1:-5]+'.pdf')
            # plt.show()
            plt.close()
        
        return cr
    
    def pnimgs(self,
               name,
               odir,
               tags,
               tint, # s
               time_comp=0., # yr
               wave_comp=0.5, # micron
               Nobs=1,
               overwrite=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if photon noise images were already computed.
            ofile = odir+name+'/PHN/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing photon noise images...')
                
                # Load detector images.
                ifile = odir+name+'/DET/'+tag+'.fits'
                hdul = pyfits.open(ifile)
                if (np.min(hdul[0].data) < self.fill):
                    raise UserWarning('Input image cannot have negative pixels')
                ww_time = np.argmin(np.abs(hdul['TIME'].data-time_comp))
                ww_wave = np.argmin(np.abs(hdul['WAVE'].data-wave_comp))
                imgs = hdul[0].data[ww_time, ww_wave].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                time = hdul['TIME'].data[ww_time] # yr
                wave = hdul['WAVE'].data[ww_wave]*1e-6 # m
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                imgs[mask < 0.5] = 0.
                
                # Compute frame time and number of frames.
                cr_peak = np.max(imgs) # ph/s
                tframe = np.real(-1./cr_peak*(1.+lambertw(-self.prob/np.exp(1.), -1))) # s
                Nframe = int(np.ceil(tint/tframe))
                tframe = float(tint)/Nframe # s
                print('   Frame time = %.3f s, integration time = %.0f s --> %.0f frames' % (tframe, tint, Nframe))
                
                # Go through number of observations.
                # imgs *= tframe # ph
                imgs *= tint # ph
                data = [] # ph
                for i in range(Nobs):
                    
                    # # Go through number of frames.
                    # frames = [] # ph
                    # for j in range(Nframe):
                    #     frames += [np.random.poisson(lam=imgs)] # ph
                    # frames = np.array(frames) # ph
                    # data += [np.sum(frames, axis=0)] # ph
                    data += [np.random.poisson(lam=imgs)] # ph
                data = np.array(data) # ph
                
                # Save photon noise images.
                hdul[0].data = data[:, np.newaxis, np.newaxis, :, :].astype(float) # ph
                hdul[0].header['IMSZ'] = self.imsz # pix
                hdul[0].header['IMSC'] = self.imsc # mas
                hdul[0].header['TINT'] = tint # s
                hdul[0].header['TFRAME'] = tframe # s
                hdul['TIME'].data = np.array([time]) # yr
                hdul['WAVE'].data = np.array([wave*1e6]) # micron
                path = odir+name+'/PHN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(ofile, output_verify='fix', overwrite=True)
                hdul.close()
        
        pass
    
    def pnimgs_all(self,
                   name,
                   odir,
                   tags,
                   tint, # s
                   Nobs=1,
                   overwrite=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if photon noise images were already computed.
            ofile = odir+name+'/PHN/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing photon noise images...')
                
                # Load detector images.
                ifile = odir+name+'/DET/'+tag+'.fits'
                hdul = pyfits.open(ifile)
                if (np.min(hdul[0].data) < self.fill):
                    raise UserWarning('Input image cannot have negative pixels')
                data = np.zeros_like(hdul[0].data) # ph
                for i in range(hdul[0].data.shape[0]):
                    for j in range(hdul[0].data.shape[1]):
                        imgs = hdul[0].data[i, j].astype(float)
                        imsz = hdul[0].header['IMSZ'] # pix
                        imsc = hdul[0].header['IMSC'] # mas
                        diam = hdul[0].header['DIAM'] # m
                        time = hdul['TIME'].data[i] # yr
                        wave = hdul['WAVE'].data[j]*1e-6 # m
                        
                        # Compute mask.
                        ramp = np.arange(imsz)-(imsz-1)/2. # pix
                        xx, yy = np.meshgrid(ramp, ramp) # pix
                        dist = np.sqrt(xx**2+yy**2) # pix
                        iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                        owa = self.owa*wave/diam*rad2mas/imsc # pix
                        mask = (dist > iwa) & (dist < owa)
                        imgs[mask < 0.5] = 0.
                        
                        # Compute frame time and number of frames.
                        cr_peak = np.max(imgs) # ph/s
                        tframe = np.real(-1./cr_peak*(1.+lambertw(-self.prob/np.exp(1.), -1))) # s
                        Nframe = int(np.ceil(tint/tframe))
                        tframe = float(tint)/Nframe # s
                        # print('   Frame time = %.3f s, integration time = %.0f s --> %.0f frames' % (tframe, tint, Nframe))
                        
                        # Go through number of observations.
                        # imgs *= tframe # ph
                        imgs *= tint # ph
                        data[i, j] = np.random.poisson(lam=imgs) # ph
                
                # Save photon noise images.
                hdul[0].data = data.astype(float) # ph
                hdul[0].header['IMSZ'] = self.imsz # pix
                hdul[0].header['IMSC'] = self.imsc # mas
                hdul[0].header['TINT'] = tint # s
                hdul[0].header['TFRAME'] = tframe # s
                # hdul['TIME'].data = np.array([time]) # yr
                # hdul['WAVE'].data = np.array([wave*1e6]) # micron
                path = odir+name+'/PHN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(ofile, output_verify='fix', overwrite=True)
                hdul.close()
        
        pass
    
    def pncals(self,
               name,
               odir,
               tags_in1,
               tags_in2,
               tags_out,
               overwrite=False):
        
        # Go through all tags.
        for ii, tag in enumerate(tags_out):
            
            # Check if calibrated images were already computed.
            ofile = odir+name+'/PHN/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing calibrated images...')
                
                # Load photon noise images.
                ifile = odir+name+'/PHN/'+tags_in1[ii]+'.fits'
                hdul1 = pyfits.open(ifile)
                ifile = odir+name+'/PHN/'+tags_in2[ii]+'.fits'
                hdul2 = pyfits.open(ifile)
                
                # Save calibrated images.
                temp = hdul1[0].data.astype(float)-hdul2[0].data.astype(float) # ph
                hdul1[0].data = temp # ph
                hdul1.writeto(ofile, output_verify='fix', overwrite=True)
                hdul1.close()
                hdul2.close()
        
        pass
    
    def pnphot(self,
               name,
               odir,
               tags,
               papt, # pix
               rapt, # lambda/D
               inds=[0],
               oversample=1,
               fdir=None,
               apt2=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if photon noise images were already computed.
            ifile = odir+name+'/PHN/'+tag+'.fits'
            if (os.path.exists(ifile)):
                
                print('Computing photon noise photometry...')
                
                # Load photon noise images.
                hdul = pyfits.open(ifile)
                imgs = hdul[0].data[:, 0, 0].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                time = hdul['TIME'].data[0] # yr
                wave = hdul['WAVE'].data[0]*1e-6 # m
                
                # Go through indicies.
                cr_mean = [] # ph
                cr_sdev = [] # ph
                for ind in inds:
                    
                    # Compute aperture position and radius in mas.
                    pos_mas = papt*self.imsc_scene # mas
                    pos_mas[0] *= -1. # mas
                    rad_mas = rapt*wave/diam*rad2mas # mas
                    
                    # Compute aperture position and radius on subarray in pixels.
                    Npix = int(np.ceil(3*rad_mas/imsc))
                    pos_pix = (pos_mas/imsc+(imsz-1)/2.).astype(int)
                    subarr = np.fliplr(imgs[ind].copy())[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                    if (apt2 == True):
                        subarr2 = imgs[ind].copy()[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                    pos_subarr = (pos_mas/imsc+(imsz-1)/2.)-pos_pix+Npix
                    rad_subarr = rad_mas/imsc
                    
                    # Compute aperture position and radius on oversampled subarray in pixels.
                    if (oversample == 1):
                        subarr_zoom = subarr.copy()
                        if (apt2 == True):
                            subarr2_zoom = subarr2.copy()
                    else:
                        norm = np.sum(subarr)
                        subarr_zoom = zoom(subarr, oversample, mode='nearest', order=5)
                        subarr_zoom *= norm/np.sum(subarr_zoom)
                        if (apt2 == True):
                            norm2 = np.sum(subarr2)
                            subarr2_zoom = zoom(subarr2, oversample, mode='nearest', order=5)
                            subarr2_zoom *= norm2/np.sum(subarr2_zoom)
                    pos_subarr_zoom = pos_subarr*oversample+(oversample-1.)/2.
                    rad_subarr_zoom = rad_subarr*oversample
                    
                    # Compute aperture on oversampled subarray in pixels.
                    ramp = np.arange(subarr_zoom.shape[0])
                    xx, yy = np.meshgrid(ramp, ramp)
                    aptr = np.sqrt((xx-pos_subarr_zoom[0])**2+(yy-pos_subarr_zoom[1])**2) <= rad_subarr_zoom
                    
                    # Compute aperture count rate.
                    if (apt2 == True):
                        cr_mean += [np.mean(subarr_zoom[aptr]), np.mean(subarr2_zoom[aptr])] # ph
                        cr_sdev += [np.std(subarr_zoom[aptr]), np.std(subarr2_zoom[aptr])] # ph
                    else:
                        cr_mean += [np.mean(subarr_zoom[aptr])] # ph
                        cr_sdev += [np.std(subarr_zoom[aptr])] # ph
                path = fdir+name+'/PHN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                np.save(path+tag+'_cr_mean', np.array(cr_mean))
                np.save(path+tag+'_cr_sdev', np.array(cr_sdev))
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                mask_subarr = np.fliplr(mask)[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                mask_subarr_zoom = zoom(mask_subarr, oversample, mode='nearest', order=0)
                
                # Plot.
                if (fdir is not None):
                    ext = imsz/2.*imsc # mas
                    f, ax = plt.subplots(1, 4, figsize=(4.8*4, 3.6*1))
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp0[mask < 0.5] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin0 = max(np.log10(np.nanmin(temp0)), 0.)
                    vmax0 = np.log10(np.nanmax(temp0))
                    p0 = ax[0].imshow(np.fliplr(np.log10(temp0)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    c0 = plt.colorbar(p0, ax=ax[0])
                    c0.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a0 = plt.Circle(pos_mas, rad_mas, fc='none', ec='red')
                    ax[0].add_patch(a0)
                    ax[0].invert_xaxis()
                    ax[0].set_xlabel('$\Delta$RA [mas]')
                    ax[0].set_ylabel('$\Delta$DEC [mas]')
                    ax[0].set_title('Scene')
                    temp1 = np.abs(subarr.copy())
                    temp1[mask_subarr < 0.5] = np.nan
                    temp1[temp1 == 0.] = np.nan
                    vmin1 = max(np.log10(np.nanmin(temp1)), 0)
                    vmax1 = np.log10(np.nanmax(temp1))
                    p1 = ax[1].imshow(np.log10(temp1), origin='lower', vmin=vmin1, vmax=vmax1)
                    c1 = plt.colorbar(p1, ax=ax[1])
                    c1.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a1 = plt.Circle(pos_subarr, rad_subarr, fc='none', ec='red')
                    ax[1].add_patch(a1)
                    ax[1].invert_xaxis()
                    ax[1].set_xlabel('$\Delta$RA [pix]')
                    ax[1].set_ylabel('$\Delta$DEC [pix]')
                    ax[1].set_title('PSF')
                    temp2 = np.abs(subarr_zoom.copy())
                    temp2[mask_subarr_zoom < 0.5] = np.nan
                    temp2[temp2 == 0.] = np.nan
                    p2 = ax[2].imshow(np.log10(temp2), origin='lower', vmin=vmin1-2.*np.log10(oversample), vmax=vmax1-2.*np.log10(oversample))
                    c2 = plt.colorbar(p2, ax=ax[2])
                    c2.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a2 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
                    ax[2].add_patch(a2)
                    ax[2].invert_xaxis()
                    ax[2].set_xlabel('$\Delta$RA [pix]')
                    ax[2].set_ylabel('$\Delta$DEC [pix]')
                    ax[2].set_title('Oversampled PSF')
                    p3 = ax[3].imshow(aptr, origin='lower')
                    c3 = plt.colorbar(p3, ax=ax[3])
                    c3.set_label('Transmission', rotation=270, labelpad=20)
                    a3 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
                    ax[3].add_patch(a3)
                    text = ax[3].text(aptr.shape[0]-1., 0.08*aptr.shape[1], 'CR (mean) = %.3e ph/s' % cr_mean[-1], color='white', ha='left', va='bottom', size=10)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                    text = ax[3].text(aptr.shape[0]-1., 0., 'CR (sdev) = %.3e ph/s' % cr_sdev[-1], color='white', ha='left', va='bottom', size=10)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                    ax[3].invert_xaxis()
                    ax[3].set_xlabel('$\Delta$RA [pix]')
                    ax[3].set_ylabel('$\Delta$DEC [pix]')
                    ax[3].set_title('Oversampled aperture')
                    plt.tight_layout()
                    plt.savefig(path+tag+'.pdf')
                    plt.close()
            
            else:
                raise UserWarning('Photon noise images were not computed yet.')
        
        pass
    
    def pnplot(self,
                name,
                odir,
                tags,
                rapt, # lambda/D
                cr_star, # ph/s
                cr_plan, # ph/s
                cr_disk, # ph/s
                cr_detn, # ph/s
                fdir):
        
        print('Plotting photon noise photometry...')
        
        # Go through all tags.
        cr_mean = {}
        cr_sdev = {}
        mins = []
        maxs = []
        for tag in tags:
            
            # Check if photon noise images were already computed.
            ifile = odir+name+'/PHN/'+tag+'.fits'
            if (os.path.exists(ifile)):
                
                # Load photon noise images.
                hdul = pyfits.open(ifile)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                tint = hdul[0].header['TINT'] # s
                time = hdul['TIME'].data[0] # yr
                wave = hdul['WAVE'].data[0]*1e-6 # m
                
                # Compute number of pixels in aperture.
                Npix = (np.pi*(rapt*wave/diam*rad2mas)**2)/imsc**2
                
                # Get count rate.
                path = fdir+name+'/PHN/'
                cr_mean[tag] = np.load(path+tag+'_cr_mean.npy') # ph
                cr_sdev[tag] = np.load(path+tag+'_cr_sdev.npy') # ph
                mins += [np.min(cr_sdev[tag])] # ph
                maxs += [np.max(cr_sdev[tag])] # ph
            
            else:
                raise UserWarning('Photon noise images were not computed yet.')
        
        # Plot.
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        plt.figure()
        ax = plt.gca()
        bins = np.linspace(np.min(mins), np.max(maxs), 30)
        for ii, tag in enumerate(tags):
            ax.hist(cr_sdev[tag], bins=bins, color=colors[ii], histtype='step', lw=3, label=tag)
            ax.axvline(np.median(cr_sdev[tag]), color=colors[ii], ls='--')
        cr_nois = 0.
        if (cr_star is not None):
            cr_nois += cr_star
        if (cr_plan is not None):
            cr_nois += cr_plan
        if (cr_disk is not None):
            cr_nois += cr_disk
        if (cr_detn is not None):
            cr_nois += cr_detn
        ax.axvline(np.sqrt((cr_nois+cr_star)*tint/Npix), color='black', label='expected')
        ax.axvline(np.sqrt(2.*cr_nois*tint/Npix), color='black')
        ax.set_ylim([0., 1.5*ax.get_ylim()[1]])
        ax.set_axisbelow(True)
        ax.yaxis.grid()
        ax.set_xlabel('Pixel-to-pixel noise [ph]')
        ax.set_ylabel('Number')
        ax.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(path+'_cr_sdev.pdf')
        plt.close()
        
        pass
    
    # def pnplot_all(self,
    #                names,
    #                odir,
    #                tags,
    #                rapt, # lambda/D
    #                cr_stars, # ph/s
    #                cr_plans, # ph/s
    #                cr_disks, # ph/s
    #                cr_detns, # ph/s
    #                fdir):
        
    #     print('Plotting photon noise photometry...')
        
    #     cr_measured_ref = []
    #     cr_expected_ref = []
    #     cr_measured_010 = []
    #     cr_expected_010 = []
    #     cr_measured_180 = []
    #     cr_expected_180 = []
    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     xtitle = ['inc = 0 deg', 'inc = 30 deg', 'inc = 60 deg']
    #     ytitle = ['z = 1 zodi', 'z = 2 zodi', 'z = 5 zodi', 'z = 10 zodi', 'z = 20 zodi', 'z = 50 zodi', 'z = 100 zodi', 'z = 200 zodi', 'z = 500 zodi', 'z = 1000 zodi']
    #     labels = ['Ref sub', '10 deg', '180 deg']
    #     f, ax = plt.subplots(10, 3, figsize=(3*6.4, 5*4.8))
    #     for i in range(len(names)):
            
    #         # Go through all tags.
    #         cr_mean = {}
    #         cr_sdev = {}
    #         mins = []
    #         maxs = []
    #         for tag in tags:
                
    #             # Check if photon noise images were already computed.
    #             ifile = odir+names[i]+'/PHN/'+tag+'.fits'
    #             if (os.path.exists(ifile)):
                    
    #                 # Load photon noise images.
    #                 hdul = pyfits.open(ifile)
    #                 imsz = hdul[0].header['IMSZ'] # pix
    #                 imsc = hdul[0].header['IMSC'] # mas
    #                 diam = hdul[0].header['DIAM'] # m
    #                 tint = hdul[0].header['TINT'] # s
    #                 time = hdul['TIME'].data[0] # yr
    #                 wave = hdul['WAVE'].data[0]*1e-6 # m
                    
    #                 # Compute number of pixels in aperture.
    #                 Npix = (np.pi*(rapt*wave/diam*rad2mas)**2)/imsc**2
                    
    #                 # Get count rate.
    #                 path = fdir+names[i]+'/PHN/'
    #                 cr_mean[tag] = np.load(path+tag+'_cr_mean.npy') # ph
    #                 cr_sdev[tag] = np.load(path+tag+'_cr_sdev.npy') # ph
    #                 mins += [np.min(cr_sdev[tag])] # ph
    #                 maxs += [np.max(cr_sdev[tag])] # ph
                
    #             else:
    #                 raise UserWarning('Photon noise images were not computed yet.')
            
    #         # Plot.
    #         row = i//3
    #         col = i % 3
    #         bins = np.linspace(np.min(mins), np.max(maxs), 15)
    #         for ii, tag in enumerate(tags):
    #             ax[row, col].hist(cr_sdev[tag], bins=bins, color=colors[ii], histtype='step', label=labels[ii])
    #             ax[row, col].axvline(np.median(cr_sdev[tag]), color=colors[ii], ls='--', lw=3)
    #             if (ii == 0):
    #                 cr_measured_ref += [np.median(cr_sdev[tag])]
    #             if (ii == 1):
    #                 cr_measured_010 += [np.median(cr_sdev[tag])]
    #             if (ii == 2):
    #                 cr_measured_180 += [np.median(cr_sdev[tag])]
    #         cr_nois = 0.
    #         if (cr_stars is not None):
    #             cr_nois += cr_stars[i]
    #         if (cr_plans is not None):
    #             cr_nois += cr_plans[i]
    #         if (cr_disks is not None):
    #             cr_nois += cr_disks[i]
    #         if (cr_detns is not None):
    #             cr_nois += cr_detns[i]
    #         ax[row, col].axvline(np.sqrt((cr_nois+cr_stars[i])*tint/Npix), color=colors[0], lw=3)
    #         cr_expected_ref += [np.sqrt((cr_nois+cr_stars[i])*tint/Npix)]
    #         ax[row, col].axvline(np.sqrt(2.*cr_nois*tint/Npix), color=colors[2], lw=3)
    #         cr_expected_010 += [np.sqrt(2.*cr_nois*tint/Npix)]
    #         cr_expected_180 += [np.sqrt(2.*cr_nois*tint/Npix)]
    #         ax[row, col].set_ylim([0., 1.25*ax[row, col].get_ylim()[1]])
    #         ax[row, col].yaxis.tick_right()
    #         ax[row, col].set_axisbelow(True)
    #         ax[row, col].yaxis.grid()
    #         if (row == 0 and col == 0):
    #             ax[row, col].legend(loc='upper left')
    #         if (row == 0):
    #             ax[row, col].set_title(xtitle[col])
    #         if (row == 9):
    #             ax[row, col].set_xlabel('Pixel-to-pixel sdev [ph]')
    #         if (col == 0):
    #             ax[row, col].set_ylabel(ytitle[row], size=18)
    #         if (col == 2):
    #             ax[row, col].yaxis.set_label_position('right')
    #             ax[row, col].set_ylabel('Number', rotation=270, labelpad=20)
    #     plt.tight_layout()
    #     plt.savefig('noise_performance.pdf')
    #     # plt.savefig('noise_performance_30pm.pdf')
    #     # plt.savefig('noise_performance_90pm.pdf')
    #     # plt.savefig('noise_performance_15pc.pdf')
    #     plt.close()
        
    #     cr_measured_ref = np.array(cr_measured_ref)
    #     cr_expected_ref = np.array(cr_expected_ref)
    #     f_ref = np.true_divide(cr_measured_ref, cr_expected_ref)
    #     cr_measured_010 = np.array(cr_measured_010)
    #     cr_expected_010 = np.array(cr_expected_010)
    #     f_010 = np.true_divide(cr_measured_010, cr_expected_010)
    #     cr_measured_180 = np.array(cr_measured_180)
    #     cr_expected_180 = np.array(cr_expected_180)
    #     f_180 = np.true_divide(cr_measured_180, cr_expected_180)
    #     f, ax = plt.subplots(1, 3, figsize=(3*6.4, 1*4.8))
    #     xx = np.array([1, 2, 5, 10, 20, 50, 100, 200, 500, 1000])
    #     for i in range(3):
    #         ax[i].plot(xx, f_ref[i::3], label=labels[0])
    #         ax[i].plot(xx, f_010[i::3], label=labels[1])
    #         ax[i].plot(xx, f_180[i::3], label=labels[2])
    #         ax[i].axhline(1., ls='--', color='black')
    #         ax[i].set_xscale('log')
    #         ax[i].yaxis.grid()
    #         ax[i].set_xlabel('z [zodi]')
    #         if (i == 0):
    #             ax[i].set_ylabel(r'$\rm{N}_{\rm{measured}}/\rm{N}_{\rm{photon}}$')
    #             ax[i].legend(loc='upper left')
    #         ax[i].set_title(xtitle[i], y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
    #     plt.suptitle('10 pc dist -- 10 pm RMS WFE', y=0.925)
    #     # plt.suptitle('10 pc dist -- 30 pm RMS WFE', y=0.925)
    #     # plt.suptitle('10 pc dist -- 90 pm RMS WFE', y=0.925)
    #     # plt.suptitle('15 pc dist -- 10 pm RMS WFE', y=0.925)
    #     plt.tight_layout()
    #     plt.savefig('noise_ratio.pdf')
    #     # plt.savefig('noise_ratio_30pm.pdf')
    #     # plt.savefig('noise_ratio_90pm.pdf')
    #     # plt.savefig('noise_ratio_15pc.pdf')
    #     plt.close()        
        
    #     pass
    
    def pndisk(self,
               name,
               odir,
               tags,
               inds=[0],
               inc=0., # deg
               fitinc=False,
               Omegas=None, # deg
               x0=None,
               l0=None,
               burnin=10,
               sym='even',
               fdir=None):
        
        # Get fit model and prior.
        if (sym == 'even'):
            if (l0 is None):
                if (fitinc == False):
                    l0 = np.array(['r-1', 'r-2', 'r-3', 'r-4', 'r-5', 'p+0', 'p+1', 'p+2', 'p+3', 'p+4', 'p+5', 'p+6', 'p+7', 'p+8', 'p+9', 'p+10'])
                else:
                    l0 = np.array(['r-1', 'r-2', 'r-3', 'r-4', 'r-5', 'p+0', 'p+1', 'p+2', 'p+3', 'p+4', 'p+5', 'p+6', 'p+7', 'p+8', 'p+9', 'p+10', 'inc'])
            if (x0 is None):
                x0 = np.ones(l0.shape)
        elif ((sym == 'odd') or (sym == 'both')):
            if (l0 is None):
                l0 = np.array(['r-1', 'r-2', 'r-3', 'r-4', 'r-5', 'p+2', 'p+3', 'p+4', 'p+5', 'p+6', 'p+7', 'p+8', 'p+9', 'p+10'])
            if (x0 is None):
                x0 = np.ones(l0.shape)
        else:
            raise UserWarning(sym+' is an unknown symmetry')
        if (len(l0) != len(x0)):
            raise UserWarning('Fit model and prior need to have same number of elements')
        if (Omegas is None):
            Omegas = np.zeros((len(inds)))
        
        # Go through all tags.
        for ii, tag in enumerate(tags):
            
            # Check if photon noise images were already computed.
            ifile = odir+name+'/PHN/'+tag+'.fits'
            if (os.path.exists(ifile)):
                
                print('Fitting and subtracting disk...')
                
                # Load photon noise images.
                hdul = pyfits.open(ifile)
                imgs = hdul[0].data[:, 0, 0].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                time = hdul['TIME'].data[0] # yr
                wave = hdul['WAVE'].data[0]*1e-6 # m
                tint = hdul[0].header['TINT'] # s
                
                # Load noiseless images.
                if ('cal_psf_imgs' in ifile):
                    rfile1 = ifile.replace('PHN', 'DET').replace('cal_psf_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PHN', 'DET').replace('cal_psf_imgs', 'ref_psf_imgs')
                elif ('cal_010_imgs' in ifile):
                    rfile1 = ifile.replace('PHN', 'DET').replace('cal_010_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PHN', 'DET').replace('cal_010_imgs', 'ref_010_imgs')
                elif ('cal_180_imgs' in ifile):
                    rfile1 = ifile.replace('PHN', 'DET').replace('cal_180_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PHN', 'DET').replace('cal_180_imgs', 'ref_180_imgs')
                else:
                    raise UserWarning('Cannot find noiseless images for this data')
                rimgs1 = pyfits.getdata(rfile1)[1, 1].astype(float)
                rimgs2 = pyfits.getdata(rfile2)[1, 1].astype(float)
                errs = np.sqrt(tint*(rimgs1+rimgs2))
                
                # box = 1
                # test = np.zeros_like(errs)
                # ramp = np.arange(test.shape[0])
                # xx, yy = np.meshgrid(ramp, ramp)
                # for i in range(test.shape[0]):
                #     for j in range(test.shape[1]):
                #         ww = (np.abs(xx-j) <= box) & (np.abs(yy-i) <= box)
                #         test[i, j] = np.nanstd(imgs[0][ww])
                # f, ax = plt.subplots(1, 2, figsize=(2*6.4, 1*4.8))
                # p0 = ax[0].imshow(test, origin='lower', vmax=100)
                # plt.colorbar(p0, ax=ax[0])
                # p1 = ax[1].imshow(errs, origin='lower', vmax=100)
                # plt.colorbar(p1, ax=ax[1])
                # plt.tight_layout()
                # plt.show()
                # import pdb; pdb.set_trace()
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                
                Omega = Omegas[ii]
                
                # Burn-in.
                erel_super = [1.]
                x0_super = [x0.copy()]
                if (burnin is not None):
                    if (len(inds) < 3*burnin):
                        raise UserWarning('Not enough frames for burn-in')
                    for i in range(3):
                        erel_temp = 1.
                        x0_temp = x0.copy()
                        for j in range(burnin):
                            # errs = np.maximum(np.sqrt(np.abs(imgs[inds[i*burnin+j]].copy())), 1.) # FIXME
                            # errs = None
                            res = leastsq(self.chi_simpol,
                                          x0=x0_temp,
                                          args=(l0, xx, yy, inc, Omega, sym, imgs[inds[i*burnin+j]].copy(), mask, errs),
                                          full_output=True)
                            res = {'x': res[0], 'fun': res[2]['fvec']}
                            if ('inc' in l0):
                                ww_inc = np.where(l0 == 'inc')[0][0]
                                inc = res['x'][ww_inc]
                            if ('Omega' in l0):
                                ww_Omega = np.where(l0 == 'Omega')[0][0]
                                Omega = res['x'][ww_Omega]
                            if (sym == 'even'):
                                rho, phi, sgn = util.proj_sa(xx, yy, inc, Omega)
                            elif (sym == 'odd'):
                                rho, phi, sgn = util.proj_pc(xx, yy, inc, Omega)
                            elif (sym == 'both'):
                                rho, phi, sgn = util.proj_qc(xx, yy, inc, Omega)
                            disk = self.simpol(res['x'], l0, rho, phi, sgn, sym)
                            temp = np.abs(imgs[inds[i*burnin+j]].copy())
                            temp[temp == 0.] = np.nan
                            temp = np.nanmedian(res['fun']/temp[mask])
                            if (temp < erel_temp):
                                erel_temp = temp
                                x0_temp = res['x']
                        erel_super += [erel_temp]
                        x0_super += [x0_temp]
                x0_temp = x0_super[np.argmin(erel_super)]
                # x0_temp = x0.copy() # ignore burn-in
                
                # Go through indicies.
                disk = []
                erel = []
                incs = []
                for ind in inds:
                    
                    # Fit disk.
                    if (fitinc == True):
                        inc = np.rad2deg(np.arccos(2.*np.random.rand()-1.))
                        x0_temp[-1] = inc
                    # errs = np.maximum(np.sqrt(np.abs(imgs[ind].copy())), 1.) # FIXME
                    # errs = None
                    res = leastsq(self.chi_simpol,
                                  x0=x0_temp,
                                  args=(l0, xx, yy, inc, Omega, sym, imgs[ind].copy(), mask, errs),
                                  full_output=True)
                    res = {'x': res[0], 'fun': res[2]['fvec']}
                    if ('inc' in l0):
                        ww_inc = np.where(l0 == 'inc')[0][0]
                        inc = res['x'][ww_inc]
                    if ('Omega' in l0):
                        ww_Omega = np.where(l0 == 'Omega')[0][0]
                        Omega = res['x'][ww_Omega]
                    if (sym == 'even'):
                        rho, phi, sgn = util.proj_sa(xx, yy, inc, Omega)
                    elif (sym == 'odd'):
                        rho, phi, sgn = util.proj_pc(xx, yy, inc, Omega)
                    elif (sym == 'both'):
                        rho, phi, sgn = util.proj_qc(xx, yy, inc, Omega)
                    disk += [self.simpol(res['x'], l0, rho, phi, sgn, sym)]
                    temp = np.abs(imgs[ind].copy())
                    temp[temp == 0.] = np.nan
                    erel += [np.nanmedian(np.abs(imgs[ind][mask]-disk[-1][mask])/temp[mask])]
                    incs += [inc]
                disk = np.array(disk)
                erel = np.array(erel)
                incs = np.array(incs)
                
                # Plot.
                if (fdir is not None):
                    ext = imsz/2.*imsc # mas
                    f, ax = plt.subplots(1, 3, figsize=(3*6.4, 1*4.8))
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp0[mask < 0.5] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin0 = np.log10(np.nanpercentile(temp0, 10.))
                    vmax0 = np.log10(np.nanmax(temp0))
                    # vmin0 = 0.4
                    # vmax0 = 2.2
                    p0 = ax[0].imshow(np.fliplr(np.log10(temp0)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    
                    # temp0p = imgs[inds[-1]].copy()
                    # temp0p[mask < 0.5] = np.nan
                    # temp0p[temp0p < 1.] = np.nan
                    # temp0n = imgs[inds[-1]].copy()
                    # temp0n[mask < 0.5] = np.nan
                    # temp0n[temp0n > -1.] = np.nan
                    # temp00 = imgs[inds[-1]].copy()
                    # temp00[:] = np.nan
                    # ax[0].imshow(np.fliplr(np.log10(np.abs(temp0p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[0].imshow(np.fliplr(np.log10(np.abs(temp0n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p0 = ax[0].imshow(temp00, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c0 = plt.colorbar(p0, ax=ax[0])
                    c0.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    ax[0].invert_xaxis()
                    ax[0].set_xlabel('$\Delta$RA [mas]')
                    ax[0].set_ylabel('$\Delta$DEC [mas]')
                    if (sym == 'even'):
                        ax[0].set_title('Ref star subtracted')
                    elif (sym == 'odd'):
                        ax[0].set_title('10 deg roll subtracted')
                    elif (sym == 'both'):
                        ax[0].set_title('180 deg roll subtracted')
                    temp1 = np.abs(disk[-1].copy())
                    temp1[mask < 0.5] = np.nan
                    temp1[temp1 == 0.] = np.nan
                    p1 = ax[1].imshow(np.fliplr(np.log10(temp1)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    
                    # temp1p = disk[-1].copy()
                    # temp1p[mask < 0.5] = np.nan
                    # temp1p[temp1p <= 0.] = np.nan
                    # temp1n = disk[-1].copy()
                    # temp1n[mask < 0.5] = np.nan
                    # temp1n[temp1n >= 0.] = np.nan
                    # temp11 = disk[-1].copy()
                    # temp11[:] = np.nan
                    # ax[1].imshow(np.fliplr(np.log10(np.abs(temp1p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[1].imshow(np.fliplr(np.log10(np.abs(temp1n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p1 = ax[1].imshow(temp11, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c1 = plt.colorbar(p1, ax=ax[1])
                    c1.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    # if (sym == 'even'):
                    #     ax[1].axvline(0., color='green', lw=3)
                    # elif (sym == 'odd'):
                    #     tempx = ax[1].get_xlim()
                    #     tempy = ax[1].get_ylim()
                    #     ax[1].plot([-21.872165881481001, 21.872165881481001], [-250., 250.], color='green', ls='--', lw=3)
                    #     ax[1].set_xlim(tempx)
                    #     ax[1].set_ylim(tempy)
                    # elif (sym == 'both'):
                    #     ax[1].axvline(0., color='green', lw=3)
                    #     ax[1].axhline(0., color='green', ls='--', lw=3)
                    text = ax[1].text(ext-imsc, -ext+imsc, 'inc = %.1f deg, PA = %.1f deg' % (inc, Omega), color='black', ha='left', va='bottom', size=14)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
                    ax[1].invert_xaxis()
                    ax[1].set_xlabel('$\Delta$RA [mas]')
                    ax[1].set_ylabel('$\Delta$DEC [mas]')
                    ax[1].set_title('Parametric disk model')
                    temp2 = np.abs(imgs[inds[-1]]-disk[-1])
                    temp2[mask < 0.5] = np.nan
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp2[temp0 == 0.] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin2 = np.log10(np.nanpercentile(temp2, 10.))
                    vmax2 = np.log10(np.nanmax(temp2))
                    p2 = ax[2].imshow(np.fliplr(np.log10(temp2)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin2, vmax=vmax2)
                    
                    # temp2p = imgs[inds[-1]]-disk[-1]
                    # temp2p[mask < 0.5] = np.nan
                    # temp0 = np.abs(imgs[inds[-1]].copy())
                    # temp2p[temp0 == 0.] = np.nan
                    # temp2p[temp2p <= 0.] = np.nan
                    # temp2n = imgs[inds[-1]]-disk[-1]
                    # temp2n[mask < 0.5] = np.nan
                    # temp0 = np.abs(imgs[inds[-1]].copy())
                    # temp2n[temp0 == 0.] = np.nan
                    # temp2n[temp2n >= 0.] = np.nan
                    # temp22 = imgs[inds[-1]]-disk[-1]
                    # temp22[:] = np.nan
                    # ax[2].imshow(np.fliplr(np.log10(np.abs(temp2p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[2].imshow(np.fliplr(np.log10(np.abs(temp2n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p2 = ax[2].imshow(temp22, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c2 = plt.colorbar(p2, ax=ax[2])
                    c2.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    text = ax[2].text(ext-imsc, -ext+imsc, 'Med. rel. error = %.3e' % erel[-1], color='black', ha='left', va='bottom', size=14)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
                    # circ = plt.Circle((49.77782752*2., 0.42865107*2.), 25, facecolor='none', edgecolor='green', lw=3)
                    # ax[2].add_patch(circ)
                    # if (sym == 'odd'):
                    #     alpha = np.arctan(0.42865107/49.77782752)
                    #     alpha -= 10./180.*np.pi
                    #     x_new = np.sqrt((49.77782752*2.)**2+(0.42865107*2.)**2)*np.cos(alpha)
                    #     y_new = np.sqrt((49.77782752*2.)**2+(0.42865107*2.)**2)*np.sin(alpha)
                    #     circ = plt.Circle((x_new, y_new), 25, facecolor='none', edgecolor='green', ls='--', lw=3)
                    #     ax[2].add_patch(circ)
                    # elif (sym == 'both'):
                    #     circ = plt.Circle((-49.77782752*2., -0.42865107*2.), 25, facecolor='none', edgecolor='green', ls='--', lw=3)
                    #     ax[2].add_patch(circ)
                    ax[2].invert_xaxis()
                    ax[2].set_xlabel('$\Delta$RA [mas]')
                    ax[2].set_ylabel('$\Delta$DEC [mas]')
                    ax[2].set_title('Absolute error')
                    plt.tight_layout()
                    path = fdir+name+'/PHN/'
                    if (not os.path.exists(path)):
                        os.makedirs(path)
                    plt.savefig(path+tag+'_dfit.pdf')
                    plt.close()
                    
                    np.save(path+tag+'_incs.npy', incs)
                
                # Save disk-subtracted images.
                temp = hdul[0].data[inds].astype(float)-disk[:, np.newaxis, np.newaxis, :, :] # ph
                hdul[0].data[inds] = temp # ph
                hdux = pyfits.ImageHDU(erel)
                hdux.header['EXTNAME'] = 'EREL'
                hdul.append(hdux)
                path = odir+name+'/PHN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(path+tag+'_dsub.fits', output_verify='fix', overwrite=True)
                hdul.close()
            
            else:
                raise UserWarning('Photon noise images were not computed yet.')
        
        pass
    
    def dnimgs(self,
               name,
               odir,
               tags,
               tint, # s
               time_comp=0., # yr
               wave_comp=0.5, # micron
               Nobs=1,
               overwrite=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if detector noise images were already computed.
            ofile = odir+name+'/PDN/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing detector noise images...')
                
                # Load detector images.
                ifile = odir+name+'/DET/'+tag+'.fits'
                hdul = pyfits.open(ifile)
                if (np.min(hdul[0].data) < self.fill):
                    raise UserWarning('Input image cannot have negative pixels')
                ww_time = np.argmin(np.abs(hdul['TIME'].data-time_comp))
                ww_wave = np.argmin(np.abs(hdul['WAVE'].data-wave_comp))
                imgs = hdul[0].data[ww_time, ww_wave].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                time = hdul['TIME'].data[ww_time] # yr
                wave = hdul['WAVE'].data[ww_wave]*1e-6 # m
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                imgs[mask < 0.5] = 0.
                temp = np.zeros_like(imgs)
                
                # Compute frame time and number of frames.
                cr_peak = np.max(imgs) # ph/s
                tframe = np.real(-1./cr_peak*(1.+lambertw(-self.prob/np.exp(1.), -1))) # s
                Nframe = int(np.ceil(tint/tframe))
                tframe = float(tint)/Nframe # s
                print('   Frame time = %.3f s, integration time = %.0f s --> %.0f frames' % (tframe, tint, Nframe))
                
                # Set EMCCD parameters.
                emccd = EMCCDDetect(em_gain=self.gain, # e-/ph, electron multiplying gain
                                    full_well_image=self.fw_img, # e-, image area full well capacity
                                    full_well_serial=self.fw_ser, # e-, serial register full well capacity
                                    dark_current=self.dc, # e-/pix/s, dark current
                                    cic=self.cic, # e-/pix/frame, clock-induced charge
                                    read_noise=self.rn, # e-/pix/frame, read noise
                                    bias=self.bias, # e-, bias
                                    qe=self.qe, # quantum efficiency
                                    cr_rate=self.crr, # hits/cm^2/s, cosmic ray rate
                                    pixel_pitch=13e-6, # m, distance between pixel centers
                                    eperdn=1., # e- per dn
                                    nbits=64, # 1-64, number of bits used by the ADC readout
                                    numel_gain_register=604) # number of gain register elements
                
                # Go through number of observations.
                data = [] # ph
                for i in range(Nobs):
                    
                    # Go through number of frames.
                    frames = [] # e-
                    darks = [] # e-
                    thresh = emccd.em_gain/10. # e-/ph
                    for j in range(Nframe):
                        frame = emccd.sim_sub_frame(fluxmap=imgs, frametime=tframe) # dn
                        frame = frame*emccd.eperdn-emccd.bias # e-
                        frames += [frame] # e-
                        dark = emccd.sim_sub_frame(fluxmap=temp, frametime=tframe) # dn
                        dark = dark*emccd.eperdn-emccd.bias # e-
                        darks += [dark] # e-
                    frames = np.array(frames) # e-
                    darks = np.array(darks) # e-
                    data += [np.sum(get_counts_uncorrected(frames, thresh, emccd.em_gain), axis=0)-np.sum(get_counts_uncorrected(darks, thresh, emccd.em_gain), axis=0)] # ph
                data = np.array(data) # ph
                
                # Save detector noise images.
                hdul[0].data = data[:, np.newaxis, np.newaxis, :, :].astype(float) # ph
                hdul[0].header['IMSZ'] = self.imsz # pix
                hdul[0].header['IMSC'] = self.imsc # mas
                hdul[0].header['TINT'] = tint # s
                hdul[0].header['TFRAME'] = tframe # s
                hdul[0].header['GAIN'] = self.gain # e-/ph
                hdul[0].header['DC'] = self.dc # e-/pix/s
                hdul[0].header['CIC'] = self.cic # e-/pix/frame
                hdul[0].header['RN'] = self.rn # e-/pix/frame
                hdul[0].header['BIAS'] = self.bias # e-
                hdul[0].header['QE'] = self.qe
                hdul[0].header['CRR'] = self.crr # hits/cm^2/s
                hdul['TIME'].data = np.array([time]) # yr
                hdul['WAVE'].data = np.array([wave*1e6]) # micron
                path = odir+name+'/PDN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(ofile, output_verify='fix', overwrite=True)
                hdul.close()
        
        pass
    
    def dncals(self,
               name,
               odir,
               tags_in1,
               tags_in2,
               tags_out,
               overwrite=False):
        
        # Go through all tags.
        for ii, tag in enumerate(tags_out):
            
            # Check if calibrated images were already computed.
            ofile = odir+name+'/PDN/'+tag+'.fits'
            if ((overwrite == True) or (not os.path.exists(ofile))):
                
                print('Computing calibrated images...')
                
                # Load detector noise images.
                ifile = odir+name+'/PDN/'+tags_in1[ii]+'.fits'
                hdul1 = pyfits.open(ifile)
                ifile = odir+name+'/PDN/'+tags_in2[ii]+'.fits'
                hdul2 = pyfits.open(ifile)
                
                # Save calibrated images.
                temp = hdul1[0].data.astype(float)-hdul2[0].data.astype(float) # ph
                hdul1[0].data = temp # ph
                hdul1.writeto(ofile, output_verify='fix', overwrite=True)
                hdul1.close()
                hdul2.close()
        
        pass
    
    def dnphot(self,
               name,
               odir,
               tags,
               papt, # pix
               rapt, # lambda/D
               inds=[0],
               oversample=1,
               fdir=None,
               apt2=False):
        
        # Go through all tags.
        for tag in tags:
            
            # Check if detector noise images were already computed.
            ifile = odir+name+'/PDN/'+tag+'.fits'
            if (os.path.exists(ifile)):
                
                print('Computing detector noise photometry...')
                
                # Load detector noise images.
                hdul = pyfits.open(ifile)
                imgs = hdul[0].data[:, 0, 0].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                tframe = hdul[0].header['TFRAME'] # s
                time = hdul['TIME'].data[0] # yr
                wave = hdul['WAVE'].data[0]*1e-6 # m
                
                # Go through indicies.
                cr_mean = [] # ph
                cr_sdev = [] # ph
                for ind in inds:
                    
                    # Compute aperture position and radius in mas.
                    pos_mas = papt*self.imsc_scene # mas
                    pos_mas[0] *= -1. # mas
                    rad_mas = rapt*wave/diam*rad2mas # mas
                    
                    # Compute aperture position and radius on subarray in pixels.
                    Npix = int(np.ceil(3*rad_mas/imsc))
                    pos_pix = (pos_mas/imsc+(imsz-1)/2.).astype(int)
                    subarr = np.fliplr(imgs[ind].copy())[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                    if (apt2 == True):
                        subarr2 = imgs[ind].copy()[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                    pos_subarr = (pos_mas/imsc+(imsz-1)/2.)-pos_pix+Npix
                    rad_subarr = rad_mas/imsc
                    
                    # Compute aperture position and radius on oversampled subarray in pixels.
                    if (oversample == 1):
                        subarr_zoom = subarr.copy()
                        if (apt2 == True):
                            subarr2_zoom = subarr2.copy()
                    else:
                        norm = np.sum(subarr)
                        subarr_zoom = zoom(subarr, oversample, mode='nearest', order=5)
                        subarr_zoom *= norm/np.sum(subarr_zoom)
                        if (apt2 == True):
                            norm2 = np.sum(subarr2)
                            subarr2_zoom = zoom(subarr2, oversample, mode='nearest', order=5)
                            subarr2_zoom *= norm2/np.sum(subarr2_zoom)
                    pos_subarr_zoom = pos_subarr*oversample+(oversample-1.)/2.
                    rad_subarr_zoom = rad_subarr*oversample
                    
                    # Compute aperture on oversampled subarray in pixels.
                    ramp = np.arange(subarr_zoom.shape[0])
                    xx, yy = np.meshgrid(ramp, ramp)
                    aptr = np.sqrt((xx-pos_subarr_zoom[0])**2+(yy-pos_subarr_zoom[1])**2) <= rad_subarr_zoom
                    
                    # Compute aperture count rate.
                    if (apt2 == True):
                        cr_mean += [np.mean(subarr_zoom[aptr]), np.mean(subarr2_zoom[aptr])] # ph
                        cr_sdev += [np.std(subarr_zoom[aptr]), np.std(subarr2_zoom[aptr])] # ph
                    else:
                        cr_mean += [np.mean(subarr_zoom[aptr])] # ph
                        cr_sdev += [np.std(subarr_zoom[aptr])] # ph
                path = fdir+name+'/PDN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                np.save(path+tag+'_cr_mean', np.array(cr_mean))
                np.save(path+tag+'_cr_sdev', np.array(cr_sdev))
                np.save(path+tag+'_tframe', np.array([tframe]))
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                mask_subarr = np.fliplr(mask)[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
                mask_subarr_zoom = zoom(mask_subarr, oversample, mode='nearest', order=0)
                
                # Plot.
                if (fdir is not None):
                    ext = imsz/2.*imsc # mas
                    f, ax = plt.subplots(1, 4, figsize=(4.8*4, 3.6*1))
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp0[mask < 0.5] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin0 = np.log10(np.nanmin(temp0))
                    vmax0 = np.log10(np.nanmax(temp0))
                    p0 = ax[0].imshow(np.fliplr(np.log10(temp0)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    c0 = plt.colorbar(p0, ax=ax[0])
                    c0.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a0 = plt.Circle(pos_mas, rad_mas, fc='none', ec='red')
                    ax[0].add_patch(a0)
                    ax[0].invert_xaxis()
                    ax[0].set_xlabel('$\Delta$RA [mas]')
                    ax[0].set_ylabel('$\Delta$DEC [mas]')
                    ax[0].set_title('Scene')
                    temp1 = np.abs(subarr.copy())
                    temp1[mask_subarr < 0.5] = np.nan
                    temp1[temp1 == 0.] = np.nan
                    vmin1 = np.log10(np.nanmin(temp1))
                    vmax1 = np.log10(np.nanmax(temp1))
                    p1 = ax[1].imshow(np.log10(temp1), origin='lower', vmin=vmin1, vmax=vmax1)
                    c1 = plt.colorbar(p1, ax=ax[1])
                    c1.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a1 = plt.Circle(pos_subarr, rad_subarr, fc='none', ec='red')
                    ax[1].add_patch(a1)
                    ax[1].invert_xaxis()
                    ax[1].set_xlabel('$\Delta$RA [pix]')
                    ax[1].set_ylabel('$\Delta$DEC [pix]')
                    ax[1].set_title('PSF')
                    temp2 = np.abs(subarr_zoom.copy())
                    temp2[mask_subarr_zoom < 0.5] = np.nan
                    temp2[temp2 == 0.] = np.nan
                    p2 = ax[2].imshow(np.log10(temp2), origin='lower', vmin=vmin1-2.*np.log10(oversample), vmax=vmax1-2.*np.log10(oversample))
                    c2 = plt.colorbar(p2, ax=ax[2])
                    c2.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    a2 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
                    ax[2].add_patch(a2)
                    ax[2].invert_xaxis()
                    ax[2].set_xlabel('$\Delta$RA [pix]')
                    ax[2].set_ylabel('$\Delta$DEC [pix]')
                    ax[2].set_title('Oversampled PSF')
                    p3 = ax[3].imshow(aptr, origin='lower')
                    c3 = plt.colorbar(p3, ax=ax[3])
                    c3.set_label('Transmission', rotation=270, labelpad=20)
                    a3 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
                    ax[3].add_patch(a3)
                    text = ax[3].text(aptr.shape[0]-1., 0.08*aptr.shape[1], 'CR (mean) = %.3e ph/s' % cr_mean[-1], color='white', ha='left', va='bottom', size=10)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                    text = ax[3].text(aptr.shape[0]-1., 0., 'CR (sdev) = %.3e ph/s' % cr_sdev[-1], color='white', ha='left', va='bottom', size=10)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='black')])
                    ax[3].invert_xaxis()
                    ax[3].set_xlabel('$\Delta$RA [pix]')
                    ax[3].set_ylabel('$\Delta$DEC [pix]')
                    ax[3].set_title('Oversampled aperture')
                    plt.tight_layout()
                    plt.savefig(path+tag+'.pdf')
                    plt.close()
            
            else:
                raise UserWarning('Detector noise images were not computed yet.')
        
        pass
    
    # def dnplot(self,
    #            name,
    #            odir,
    #            tags,
    #            rapt, # lambda/D
    #            cr_star, # ph/s
    #            cr_plan, # ph/s
    #            cr_disk, # ph/s
    #            cr_detn, # ph/s
    #            fdir):
        
    #     print('Plotting detector noise photometry...')
        
    #     # Go through all tags.
    #     tframe = []
    #     cr_mean = {}
    #     cr_sdev = {}
    #     mins = []
    #     maxs = []
    #     for tag in tags:
            
    #         # Check if detector noise images were already computed.
    #         ifile = odir+name+'/PDN/'+tag+'.fits'
    #         if (os.path.exists(ifile)):
                
    #             # Load photon noise images.
    #             hdul = pyfits.open(ifile)
    #             imsz = hdul[0].header['IMSZ'] # pix
    #             imsc = hdul[0].header['IMSC'] # mas
    #             diam = hdul[0].header['DIAM'] # m
    #             tint = hdul[0].header['TINT'] # s
    #             tframe += [hdul[0].header['TFRAME']] # s
    #             time = hdul['TIME'].data[0] # yr
    #             wave = hdul['WAVE'].data[0]*1e-6 # m
                
    #             # Compute number of pixels in aperture.
    #             Npix = (np.pi*(rapt*wave/diam*rad2mas)**2)/imsc**2
                
    #             # Get count rate.
    #             path = fdir+name+'/PDN/'
    #             cr_mean[tag] = np.load(path+tag+'_cr_mean.npy') # ph
    #             cr_sdev[tag] = np.load(path+tag+'_cr_sdev.npy') # ph
    #             mins += [np.min(cr_sdev[tag])] # ph
    #             maxs += [np.max(cr_sdev[tag])] # ph
            
    #         else:
    #             raise UserWarning('Detector noise images were not computed yet.')
        
    #     # Plot.
    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     plt.figure()
    #     ax = plt.gca()
    #     bins = np.linspace(np.min(mins), np.max(maxs), 30)
    #     for ii, tag in enumerate(tags):
    #         ax.hist(cr_sdev[tag], bins=bins, color=colors[ii], histtype='step', lw=3, label=tag)
    #         ax.axvline(np.median(cr_sdev[tag]), color=colors[ii], ls='--')
    #     cr_nois = 0.
    #     if (cr_star is not None):
    #         cr_nois += cr_star
    #     if (cr_plan is not None):
    #         cr_nois += cr_plan
    #     if (cr_disk is not None):
    #         cr_nois += cr_disk
    #     if (cr_detn is not None):
    #         cr_nois += cr_detn
        
    #     lam1 = tframe[0]*cr_nois/Npix
    #     lam2 = tframe[0]*(cr_star+cr_detn)/Npix
    #     ax.axvline(np.sqrt((cr_nois+cr_star+cr_detn)*tint/Npix)*(1.-np.exp(-lam1))/lam1*(1.-np.exp(-lam2))/lam2, color='black', label='expected')
    #     ax.axvline(np.sqrt(2.*cr_nois*tint/Npix)*((1.-np.exp(-lam1))/lam1)**2, color='black')
    #     ax.set_ylim([0., 1.5*ax.get_ylim()[1]])
    #     ax.set_axisbelow(True)
    #     ax.yaxis.grid()
    #     ax.set_xlabel('Pixel-to-pixel noise [ph]')
    #     ax.set_ylabel('Number')
    #     ax.legend(loc='upper right')
    #     plt.tight_layout()
    #     plt.savefig(path+'_cr_sdev.pdf')
    #     plt.close()
        
    #     pass
    
    # def dnplot_all(self,
    #                names,
    #                odir,
    #                tags,
    #                rapt, # lambda/D
    #                cr_stars, # ph/s
    #                cr_plans, # ph/s
    #                cr_disks, # ph/s
    #                cr_detns, # ph/s
    #                fdir):
        
    #     print('Plotting detector noise photometry...')
        
    #     cr_measured_ref = []
    #     cr_expected_ref = []
    #     cr_measured_010 = []
    #     cr_expected_010 = []
    #     cr_measured_180 = []
    #     cr_expected_180 = []
    #     colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    #     xtitle = ['inc = 0 deg', 'inc = 30 deg', 'inc = 60 deg']
    #     ytitle = ['z = 1 zodi', 'z = 2 zodi', 'z = 5 zodi', 'z = 10 zodi', 'z = 20 zodi', 'z = 50 zodi', 'z = 100 zodi']
    #     labels = ['Ref sub', '10 deg', '180 deg']
    #     f, ax = plt.subplots(10, 3, figsize=(3*6.4, 5*7./10.*4.8))
    #     for i in range(len(names)):
            
    #         # Go through all tags.
    #         tframe = []
    #         cr_mean = {}
    #         cr_sdev = {}
    #         mins = []
    #         maxs = []
    #         for tag in tags:
                
    #             # Check if photon noise images were already computed.
    #             ifile = odir+names[i]+'/PDN/'+tag+'.fits'
    #             if (os.path.exists(ifile)):
                    
    #                 # Load photon noise images.
    #                 hdul = pyfits.open(ifile)
    #                 imsz = hdul[0].header['IMSZ'] # pix
    #                 imsc = hdul[0].header['IMSC'] # mas
    #                 diam = hdul[0].header['DIAM'] # m
    #                 tint = hdul[0].header['TINT'] # s
    #                 tframe += [hdul[0].header['TFRAME']] # s
    #                 time = hdul['TIME'].data[0] # yr
    #                 wave = hdul['WAVE'].data[0]*1e-6 # m
                    
    #                 # Compute number of pixels in aperture.
    #                 Npix = (np.pi*(rapt*wave/diam*rad2mas)**2)/imsc**2
                    
    #                 # Get count rate.
    #                 path = fdir+names[i]+'/PDN/'
    #                 cr_mean[tag] = np.load(path+tag+'_cr_mean.npy') # ph
    #                 cr_sdev[tag] = np.load(path+tag+'_cr_sdev.npy') # ph
    #                 mins += [np.min(cr_sdev[tag])] # ph
    #                 maxs += [np.max(cr_sdev[tag])] # ph
                
    #             else:
    #                 raise UserWarning('Detector noise images were not computed yet.')
            
    #         # Plot.
    #         row = i//3
    #         col = i % 3
    #         bins = np.linspace(np.min(mins), np.max(maxs), 15)
    #         for ii, tag in enumerate(tags):
    #             ax[row, col].hist(cr_sdev[tag], bins=bins, color=colors[ii], histtype='step', label=labels[ii])
    #             ax[row, col].axvline(np.median(cr_sdev[tag]), color=colors[ii], ls='--', lw=3)
    #             if (ii == 0):
    #                 cr_measured_ref += [np.median(cr_sdev[tag])]
    #             if (ii == 1):
    #                 cr_measured_010 += [np.median(cr_sdev[tag])]
    #             if (ii == 2):
    #                 cr_measured_180 += [np.median(cr_sdev[tag])]
    #         cr_nois = 0.
    #         if (cr_stars is not None):
    #             cr_nois += cr_stars[i]
    #         if (cr_plans is not None):
    #             cr_nois += cr_plans[i]
    #         if (cr_disks is not None):
    #             cr_nois += cr_disks[i]
    #         if (cr_detns is not None):
    #             cr_nois += cr_detns[i]
    #         lam1 = tframe[0]*cr_nois/Npix
    #         lam2 = tframe[0]*(cr_stars[i]+cr_detns[i])/Npix
    #         ax[row, col].axvline(np.sqrt((cr_nois+cr_stars[i]+cr_detns[i])*tint/Npix)*(1.-np.exp(-lam1))/lam1*(1.-np.exp(-lam2))/lam2, color='black', label='expected')
    #         cr_expected_ref += [np.sqrt((cr_nois+cr_stars[i]+cr_detns[i])*tint/Npix)*(1.-np.exp(-lam1))/lam1*(1.-np.exp(-lam2))/lam2]
    #         ax[row, col].axvline(np.sqrt(2.*cr_nois*tint/Npix)*((1.-np.exp(-lam1))/lam1)**2, color='black')
    #         cr_expected_010 += [np.sqrt(2.*cr_nois*tint/Npix)*((1.-np.exp(-lam1))/lam1)**2]
    #         cr_expected_180 += [np.sqrt(2.*cr_nois*tint/Npix)*((1.-np.exp(-lam1))/lam1)**2]            
    #         ax[row, col].set_ylim([0., 1.25*ax[row, col].get_ylim()[1]])
    #         ax[row, col].yaxis.tick_right()
    #         ax[row, col].set_axisbelow(True)
    #         ax[row, col].yaxis.grid()
    #         if (row == 0 and col == 0):
    #             ax[row, col].legend(loc='upper left')
    #         if (row == 0):
    #             ax[row, col].set_title(xtitle[col])
    #         if (row == 9):
    #             ax[row, col].set_xlabel('Pixel-to-pixel sdev [ph]')
    #         if (col == 0):
    #             ax[row, col].set_ylabel(ytitle[row], size=18)
    #         if (col == 2):
    #             ax[row, col].yaxis.set_label_position('right')
    #             ax[row, col].set_ylabel('Number', rotation=270, labelpad=20)
    #     plt.tight_layout()
    #     plt.savefig('noise_performance_PDN.pdf')
    #     plt.close()
        
    #     cr_measured_ref = np.array(cr_measured_ref)
    #     cr_expected_ref = np.array(cr_expected_ref)
    #     f_ref = np.true_divide(cr_measured_ref, cr_expected_ref)
    #     cr_measured_010 = np.array(cr_measured_010)
    #     cr_expected_010 = np.array(cr_expected_010)
    #     f_010 = np.true_divide(cr_measured_010, cr_expected_010)
    #     cr_measured_180 = np.array(cr_measured_180)
    #     cr_expected_180 = np.array(cr_expected_180)
    #     f_180 = np.true_divide(cr_measured_180, cr_expected_180)
    #     f, ax = plt.subplots(1, 3, figsize=(3*6.4, 1*4.8))
    #     xx = np.array([1, 2, 5, 10, 20, 50, 100])
    #     for i in range(3):
    #         ax[i].plot(xx, f_ref[i::3], label=labels[0])
    #         ax[i].plot(xx, f_010[i::3], label=labels[1])
    #         ax[i].plot(xx, f_180[i::3], label=labels[2])
    #         ax[i].axhline(1., ls='--', color='black')
    #         ax[i].set_xscale('log')
    #         ax[i].yaxis.grid()
    #         ax[i].set_xlabel('z [zodi]')
    #         if (i == 0):
    #             ax[i].set_ylabel(r'$\rm{N}_{\rm{measured}}/\rm{N}_{\rm{photon}}$')
    #             ax[i].legend(loc='upper left')
    #         ax[i].set_title(xtitle[i], y=1., pad=-20, bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='round'))
    #     plt.suptitle('10 pc dist -- 10 pm RMS WFE', y=0.925)
    #     plt.tight_layout()
    #     plt.savefig('noise_ratio_PDN.pdf')
    #     plt.close()        
        
    #     pass
    
    def dndisk(self,
               name,
               odir,
               tags,
               inds=[0],
               inc=0., # deg
               Omegas=None, # deg
               x0=None,
               l0=None,
               burnin=10,
               sym='even',
               fdir=None):
        
        # Get fit model and prior.
        if (sym == 'even'):
            if (l0 is None):
                l0 = np.array(['r-1', 'r-2', 'r-3', 'r-4', 'r-5', 'p+0', 'p+1', 'p+2', 'p+3', 'p+4', 'p+5', 'p+6', 'p+7', 'p+8', 'p+9', 'p+10'])
            if (x0 is None):
                x0 = np.ones(l0.shape)
        elif ((sym == 'odd') or (sym == 'both')):
            if (l0 is None):
                l0 = np.array(['r-1', 'r-2', 'r-3', 'r-4', 'r-5', 'p+2', 'p+3', 'p+4', 'p+5', 'p+6', 'p+7', 'p+8', 'p+9', 'p+10'])
            if (x0 is None):
                x0 = np.ones(l0.shape)
        else:
            raise UserWarning(sym+' is an unknown symmetry')
        if (len(l0) != len(x0)):
            raise UserWarning('Fit model and prior need to have same number of elements')
        if (Omegas is None):
            Omegas = np.zeros((len(inds)))
        
        # Go through all tags.
        for ii, tag in enumerate(tags):
            
            # Check if photon noise images were already computed.
            ifile = odir+name+'/PDN/'+tag+'.fits'
            if (os.path.exists(ifile)):
                
                print('Fitting and subtracting disk...')
                
                # Load photon noise images.
                hdul = pyfits.open(ifile)
                imgs = hdul[0].data[:, 0, 0].astype(float)
                imsz = hdul[0].header['IMSZ'] # pix
                imsc = hdul[0].header['IMSC'] # mas
                diam = hdul[0].header['DIAM'] # m
                time = hdul['TIME'].data[0] # yr
                wave = hdul['WAVE'].data[0]*1e-6 # m
                tint = hdul[0].header['TINT'] # s
                
                # Load noiseless images.
                if ('cal_psf_imgs' in ifile):
                    rfile1 = ifile.replace('PDN', 'DET').replace('cal_psf_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PDN', 'DET').replace('cal_psf_imgs', 'ref_psf_imgs')
                elif ('cal_010_imgs' in ifile):
                    rfile1 = ifile.replace('PDN', 'DET').replace('cal_010_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PDN', 'DET').replace('cal_010_imgs', 'ref_010_imgs')
                elif ('cal_180_imgs' in ifile):
                    rfile1 = ifile.replace('PDN', 'DET').replace('cal_180_imgs', 'sci_imgs')
                    rfile2 = ifile.replace('PDN', 'DET').replace('cal_180_imgs', 'ref_180_imgs')
                else:
                    raise UserWarning('Cannot find noiseless images for this data')
                rimgs1 = pyfits.getdata(rfile1)[1, 1].astype(float)
                rimgs2 = pyfits.getdata(rfile2)[1, 1].astype(float)
                errs = np.sqrt(tint*(rimgs1+rimgs2))
                
                # Compute mask.
                ramp = np.arange(imsz)-(imsz-1)/2. # pix
                xx, yy = np.meshgrid(ramp, ramp) # pix
                dist = np.sqrt(xx**2+yy**2) # pix
                iwa = self.iwa*wave/diam*rad2mas/imsc # pix
                owa = self.owa*wave/diam*rad2mas/imsc # pix
                mask = (dist > iwa) & (dist < owa)
                
                Omega = Omegas[ii]
                
                # Burn-in.
                erel_super = [1.]
                x0_super = [x0.copy()]
                if (burnin is not None):
                    if (len(inds) < 3*burnin):
                        raise UserWarning('Not enough frames for burn-in')
                    for i in range(3):
                        erel_temp = 1.
                        x0_temp = x0.copy()
                        for j in range(burnin):
                            # errs = np.maximum(np.sqrt(np.abs(imgs[inds[i*burnin+j]].copy())), 1.) # FIXME
                            # errs = None
                            res = leastsq(self.chi_simpol,
                                          x0=x0_temp,
                                          args=(l0, xx, yy, inc, Omega, sym, imgs[inds[i*burnin+j]].copy(), mask, errs),
                                          full_output=True)
                            res = {'x': res[0], 'fun': res[2]['fvec']}
                            if ('inc' in l0):
                                ww_inc = np.where(l0 == 'inc')[0][0]
                                inc = res['x'][ww_inc]
                            if ('Omega' in l0):
                                ww_Omega = np.where(l0 == 'Omega')[0][0]
                                Omega = res['x'][ww_Omega]
                            if (sym == 'even'):
                                rho, phi, sgn = util.proj_sa(xx, yy, inc, Omega)
                            elif (sym == 'odd'):
                                rho, phi, sgn = util.proj_pc(xx, yy, inc, Omega)
                            elif (sym == 'both'):
                                rho, phi, sgn = util.proj_qc(xx, yy, inc, Omega)
                            disk = self.simpol(res['x'], l0, rho, phi, sgn, sym)
                            temp = np.abs(imgs[inds[i*burnin+j]].copy())
                            temp[temp == 0.] = np.nan
                            temp = np.nanmedian(res['fun']/temp[mask])
                            if (temp < erel_temp):
                                erel_temp = temp
                                x0_temp = res['x']
                        erel_super += [erel_temp]
                        x0_super += [x0_temp]
                x0_temp = x0_super[np.argmin(erel_super)]
                
                # Go through indicies.
                disk = []
                erel = []
                for ind in inds:
                    
                    # Fit disk.
                    # errs = np.maximum(np.sqrt(np.abs(imgs[inds[i*burnin+j]].copy())), 1.) # FIXME
                    # errs = None
                    res = leastsq(self.chi_simpol,
                                  x0=x0_temp,
                                  args=(l0, xx, yy, inc, Omega, sym, imgs[ind].copy(), mask, errs),
                                  full_output=True)
                    res = {'x': res[0], 'fun': res[2]['fvec']}
                    if ('inc' in l0):
                        ww_inc = np.where(l0 == 'inc')[0][0]
                        inc = res['x'][ww_inc]
                    if ('Omega' in l0):
                        ww_Omega = np.where(l0 == 'Omega')[0][0]
                        Omega = res['x'][ww_Omega]
                    if (sym == 'even'):
                        rho, phi, sgn = util.proj_sa(xx, yy, inc, Omega)
                    elif (sym == 'odd'):
                        rho, phi, sgn = util.proj_pc(xx, yy, inc, Omega)
                    elif (sym == 'both'):
                        rho, phi, sgn = util.proj_qc(xx, yy, inc, Omega)
                    disk += [self.simpol(res['x'], l0, rho, phi, sgn, sym)]
                    temp = np.abs(imgs[ind].copy())
                    temp[temp == 0.] = np.nan
                    erel += [np.nanmedian(res['fun']/temp[mask])]
                disk = np.array(disk)
                erel = np.array(erel)
                
                # Plot.
                if (fdir is not None):
                    ext = imsz/2.*imsc # mas
                    f, ax = plt.subplots(1, 3, figsize=(3*6.4, 1*4.8))
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp0[mask < 0.5] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin0 = np.log10(np.nanpercentile(temp0, 10.))
                    vmax0 = np.log10(np.nanmax(temp0))
                    # vmin0 = 0.4
                    # vmax0 = 2.2
                    p0 = ax[0].imshow(np.fliplr(np.log10(temp0)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    
                    # temp0p = imgs[inds[-1]].copy()
                    # temp0p[mask < 0.5] = np.nan
                    # temp0p[temp0p < 1.] = np.nan
                    # temp0n = imgs[inds[-1]].copy()
                    # temp0n[mask < 0.5] = np.nan
                    # temp0n[temp0n > -1.] = np.nan
                    # temp00 = imgs[inds[-1]].copy()
                    # temp00[:] = np.nan
                    # ax[0].imshow(np.fliplr(np.log10(np.abs(temp0p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[0].imshow(np.fliplr(np.log10(np.abs(temp0n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p0 = ax[0].imshow(temp00, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c0 = plt.colorbar(p0, ax=ax[0])
                    c0.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    ax[0].invert_xaxis()
                    ax[0].set_xlabel('$\Delta$RA [mas]')
                    ax[0].set_ylabel('$\Delta$DEC [mas]')
                    if (sym == 'even'):
                        ax[0].set_title('Ref star subtracted')
                    elif (sym == 'odd'):
                        ax[0].set_title('10 deg roll subtracted')
                    elif (sym == 'both'):
                        ax[0].set_title('180 deg roll subtracted')
                    temp1 = np.abs(disk[-1].copy())
                    temp1[mask < 0.5] = np.nan
                    temp1[temp1 == 0.] = np.nan
                    p1 = ax[1].imshow(np.fliplr(np.log10(temp1)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin0, vmax=vmax0)
                    
                    # temp1p = disk[-1].copy()
                    # temp1p[mask < 0.5] = np.nan
                    # temp1p[temp1p <= 0.] = np.nan
                    # temp1n = disk[-1].copy()
                    # temp1n[mask < 0.5] = np.nan
                    # temp1n[temp1n >= 0.] = np.nan
                    # temp11 = disk[-1].copy()
                    # temp11[:] = np.nan
                    # ax[1].imshow(np.fliplr(np.log10(np.abs(temp1p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[1].imshow(np.fliplr(np.log10(np.abs(temp1n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p1 = ax[1].imshow(temp11, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c1 = plt.colorbar(p1, ax=ax[1])
                    c1.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    # if (sym == 'even'):
                    #     ax[1].axvline(0., color='green', lw=3)
                    # elif (sym == 'odd'):
                    #     tempx = ax[1].get_xlim()
                    #     tempy = ax[1].get_ylim()
                    #     ax[1].plot([-21.872165881481001, 21.872165881481001], [-250., 250.], color='green', ls='--', lw=3)
                    #     ax[1].set_xlim(tempx)
                    #     ax[1].set_ylim(tempy)
                    # elif (sym == 'both'):
                    #     ax[1].axvline(0., color='green', lw=3)
                    #     ax[1].axhline(0., color='green', ls='--', lw=3)
                    text = ax[1].text(ext-imsc, -ext+imsc, 'inc = %.1f deg, PA = %.1f deg' % (inc, Omega), color='black', ha='left', va='bottom', size=14)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
                    ax[1].invert_xaxis()
                    ax[1].set_xlabel('$\Delta$RA [mas]')
                    ax[1].set_ylabel('$\Delta$DEC [mas]')
                    ax[1].set_title('Parametric disk model')
                    temp2 = np.abs(imgs[inds[-1]]-disk[-1])
                    temp2[mask < 0.5] = np.nan
                    temp0 = np.abs(imgs[inds[-1]].copy())
                    temp2[temp0 == 0.] = np.nan
                    temp0[temp0 == 0.] = np.nan
                    vmin2 = np.log10(np.nanpercentile(temp2, 10.))
                    vmax2 = np.log10(np.nanmax(temp2))
                    p2 = ax[2].imshow(np.fliplr(np.log10(temp2)), origin='lower', extent=(-ext, ext, -ext, ext), vmin=vmin2, vmax=vmax2)
                    
                    # temp2p = imgs[inds[-1]]-disk[-1]
                    # temp2p[mask < 0.5] = np.nan
                    # temp0 = np.abs(imgs[inds[-1]].copy())
                    # temp2p[temp0 == 0.] = np.nan
                    # temp2p[temp2p <= 0.] = np.nan
                    # temp2n = imgs[inds[-1]]-disk[-1]
                    # temp2n[mask < 0.5] = np.nan
                    # temp0 = np.abs(imgs[inds[-1]].copy())
                    # temp2n[temp0 == 0.] = np.nan
                    # temp2n[temp2n >= 0.] = np.nan
                    # temp22 = imgs[inds[-1]]-disk[-1]
                    # temp22[:] = np.nan
                    # ax[2].imshow(np.fliplr(np.log10(np.abs(temp2p))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Reds', vmin=vmin0, vmax=vmax0)
                    # ax[2].imshow(np.fliplr(np.log10(np.abs(temp2n))), origin='lower', extent=(-ext, ext, -ext, ext), cmap='Blues', vmin=vmin0, vmax=vmax0)
                    # p2 = ax[2].imshow(temp22, origin='lower', extent=(-ext, ext, -ext, ext), cmap='Greys', vmin=vmin0, vmax=vmax0)
                    
                    c2 = plt.colorbar(p2, ax=ax[2])
                    c2.set_label('$\log_{10}$(ph/pix)', rotation=270, labelpad=20)
                    text = ax[2].text(ext-imsc, -ext+imsc, 'Med. rel. error = %.3e' % erel[-1], color='black', ha='left', va='bottom', size=14)
                    text.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='white')])
                    # circ = plt.Circle((49.77782752*2., 0.42865107*2.), 25, facecolor='none', edgecolor='green', lw=3)
                    # ax[2].add_patch(circ)
                    # if (sym == 'odd'):
                    #     alpha = np.arctan(0.42865107/49.77782752)
                    #     alpha -= 10./180.*np.pi
                    #     x_new = np.sqrt((49.77782752*2.)**2+(0.42865107*2.)**2)*np.cos(alpha)
                    #     y_new = np.sqrt((49.77782752*2.)**2+(0.42865107*2.)**2)*np.sin(alpha)
                    #     circ = plt.Circle((x_new, y_new), 25, facecolor='none', edgecolor='green', ls='--', lw=3)
                    #     ax[2].add_patch(circ)
                    # elif (sym == 'both'):
                    #     circ = plt.Circle((-49.77782752*2., -0.42865107*2.), 25, facecolor='none', edgecolor='green', ls='--', lw=3)
                    #     ax[2].add_patch(circ)
                    ax[2].invert_xaxis()
                    ax[2].set_xlabel('$\Delta$RA [mas]')
                    ax[2].set_ylabel('$\Delta$DEC [mas]')
                    ax[2].set_title('Absolute error')
                    plt.tight_layout()
                    path = fdir+name+'/PDN/'
                    if (not os.path.exists(path)):
                        os.makedirs(path)
                    plt.savefig(path+tag+'_dfit.pdf')
                    plt.close()
                
                # Save disk-subtracted images.
                temp = hdul[0].data[inds].astype(float)-disk[:, np.newaxis, np.newaxis, :, :] # ph
                hdul[0].data[inds] = temp # ph
                hdux = pyfits.ImageHDU(erel)
                hdux.header['EXTNAME'] = 'EREL'
                hdul.append(hdux)
                path = odir+name+'/PDN/'
                if (not os.path.exists(path)):
                    os.makedirs(path)
                hdul.writeto(path+tag+'_dsub.fits', output_verify='fix', overwrite=True)
                hdul.close()
            
            else:
                raise UserWarning('Photon noise images were not computed yet.')
        
        pass
    
    def tint(self,
             name,
             odir,
             tag,
             papt, # pix
             rapt, # lambda/D
             oversample=100,
             time_comp=0., # yr
             wave_comp=0.5, # micron
             snr_targ=7.,
             path_star=None,
             path_plan=None,
             path_disk=None,
             detn=False,
             fdir=None):
        
        print('Computing integration time...')
                
        # Load detector images.
        ifile = odir+name+'/DET/'+tag+'.fits'
        hdul = pyfits.open(ifile)
        if (np.min(hdul[0].data) < self.fill):
            raise UserWarning('Input image cannot have negative pixels')
        ww_time = np.argmin(np.abs(hdul['TIME'].data-time_comp))
        ww_wave = np.argmin(np.abs(hdul['WAVE'].data-wave_comp))
        imgs = hdul[0].data[ww_time, ww_wave]
        imsz = hdul[0].header['IMSZ'] # pix
        imsc = hdul[0].header['IMSC'] # mas
        diam = hdul[0].header['DIAM'] # m
        time = hdul['TIME'].data[ww_time] # yr
        wave = hdul['WAVE'].data[ww_wave]*1e-6 # m
        
        # Compute mask.
        ramp = np.arange(imsz)-(imsz-1)/2. # pix
        xx, yy = np.meshgrid(ramp, ramp) # pix
        dist = np.sqrt(xx**2+yy**2) # pix
        iwa = self.iwa*wave/diam*rad2mas/imsc # pix
        owa = self.owa*wave/diam*rad2mas/imsc # pix
        mask = (dist > iwa) & (dist < owa)
        imgs[mask < 0.5] = 0.
        
        # Compute peak count rate.
        cr_peak = np.max(imgs) # ph/s
        tframe = np.real(-1./cr_peak*(1.+lambertw(-self.prob/np.exp(1.), -1))) # s
        
        # Compute number of pixels in aperture.
        Npix = (np.pi*(rapt*wave/diam*rad2mas)**2)/imsc**2
        
        # Compute star count rate.
        if (path_star is not None):
            # OLD: use aperture radius defined by user.
            # cr_star = self.nlphot(path_star,
            #                       papt=papt, # pix
            #                       rapt=rapt, # lambda/D
            #                       oversample=oversample,
            #                       time_comp=time_comp, # yr
            #                       wave_comp=wave_comp, # micron
            #                       fdir=fdir)
            # NEW: use aperture radius of 3 lambda/D and scale to aperture radius defined by user.
            rapt_star = 3.0 # lambda/D
            # rapt_star = 0.8 # lambda/D
            cr_star = self.nlphot(path_star,
                                  papt=papt, # pix
                                  rapt=rapt_star, # lambda/D
                                  oversample=oversample,
                                  time_comp=time_comp, # yr
                                  wave_comp=wave_comp, # micron
                                  fdir=fdir)
            cr_star *= rapt**2/rapt_star**2
        else:
            cr_star = 0. # ph/s
        
        # Compute planet count rate.
        if (path_plan is not None):
            cr_plan = self.nlphot(path_plan,
                                  papt=papt, # pix
                                  rapt=rapt, # lambda/D
                                  oversample=oversample,
                                  time_comp=time_comp, # yr
                                  wave_comp=wave_comp, # micron
                                  fdir=fdir)
        else:
            cr_plan = 0. # ph/s
        
        # Compute disk count rate.
        if (path_disk is not None):
            cr_disk = self.nlphot(path_disk,
                                  papt=papt, # pix
                                  rapt=rapt, # lambda/D
                                  oversample=oversample,
                                  time_comp=time_comp, # yr
                                  wave_comp=wave_comp, # micron
                                  fdir=fdir)
        else:
            cr_disk = 0. # ph/s
        
        # Compute detector count rate.
        if (detn == True):
            cr_detn = Npix*(self.dc+self.rn**2/tframe-self.cic*cr_peak*(1.+np.real(lambertw(-self.prob/np.exp(1.), -1)))**(-1)) # ph/s
        else:
            cr_detn = 0. # ph/s
        
        # Multiply astrophysical sources by quantum efficiency.
        if (detn == True):
            cr_star *= self.qe
            cr_plan *= self.qe
            cr_disk *= self.qe
        
        # Compute integration time.
        cr_back = cr_star+cr_disk+cr_detn # ph/s
        cr_nois = cr_plan+2.*cr_back # ph/s
        tint = snr_targ**2*cr_nois/cr_plan**2 # s
        print('   CR star = %.3e ph/s' % cr_star)
        print('   CR plan = %.3e ph/s' % cr_plan)
        print('   CR disk = %.3e ph/s' % cr_disk)
        print('   CR detn = %.3e ph/s' % cr_detn)
        print('   CR nois = %.3e ph/s' % cr_nois)
        
        return tint, cr_star, cr_plan, cr_disk, cr_detn
