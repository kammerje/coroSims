from __future__ import division

import matplotlib
matplotlib.rcParams.update({'font.size': 14})


# =============================================================================
# IMPORTS
# =============================================================================

import astropy.io.fits as pyfits
import matplotlib.pyplot as plt
import numpy as np

import os
import sys

from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import rotate, shift, zoom

import load_scene_jens
import load_scene_haystacks
import util

rad2mas = 180./np.pi*3600.*1000.
mas2rad = np.pi/180./3600./1000.


# =============================================================================
# CORONAGRAPH
# =============================================================================

class coro():
    
    def __init__(self,
                 cdir,
                 verbose=True):
        
        self.verbose = verbose
        if (self.verbose == True):
            print('Initializing coronagraph...')
        
        # Load on-axis data.
        self.stellar_intens_1 = pyfits.getdata(cdir+'stellar_intens_1.fits', 0)
        self.stellar_intens_diam_list = pyfits.getdata(cdir+'stellar_intens_diam_list.fits', 0) # lambda/D
        self.imsc = pyfits.getheader(cdir+'stellar_intens_1.fits', 0)['PIXSCALE'] # lambda/D
        
        # Load reference data.
        try:
            self.stellar_intens_2 = pyfits.getdata(cdir+'stellar_intens_2.fits', 0)
            self.hasref = True
        except:
            self.hasref = False
        
        # Load off-axis data.
        self.offax_psf = pyfits.getdata(cdir+'offax_psf.fits', 0)
        self.offax_psf_offset_list = pyfits.getdata(cdir+'offax_psf_offset_list.fits', 0) # lambda/D
        if ((self.offax_psf_offset_list.shape[1] != 2) and (self.offax_psf_offset_list.shape[0] == 2)):
            self.offax_psf_offset_list = self.offax_psf_offset_list.T
        if (self.offax_psf_offset_list.shape[1] != 2):
            raise UserWarning('Array offax_psf_offset_list should have 2 columns')
        self.xx = np.unique(self.offax_psf_offset_list[:, 0])
        self.yy = np.unique(self.offax_psf_offset_list[:, 1])
        if ((len(self.xx) == 1) and (self.xx[0] == 0)):
            self.type = '1dy'
            self.cang = 90. # deg
            print('   Radially symmetric response --> rotating PSFs ('+self.type+')')
        elif ((len(self.yy) == 1) and (self.yy[0] == 0)):
            self.type = '1dx'
            self.cang = 0. # deg
            print('   Radially symmetric response --> rotating PSFs ('+self.type+')')
        elif (len(self.xx) == 1):
            self.type = '1dyo'
            self.cang = 90. # deg
            print('   Radially symmetric response --> rotating PSFs ('+self.type+')')
        elif (len(self.yy) == 1):
            self.type = '1dxo'
            self.cang = 0. # deg
            print('   Radially symmetric response --> rotating PSFs ('+self.type+')')
        elif (np.min(self.offax_psf_offset_list) >= 0):
            self.type = '2dq'
            self.cang = 0. # deg
            print('   Quarterly symmetric response --> reflecting PSFs ('+self.type+')')
        else:
            self.type = '2df'
            self.cang = 0. # deg
            print('   Full 2D response ('+self.type+')')
        
        # Set position angle.
        self.set_pang(0.)
        
        # Center coronagraph model so that image size is odd and central pixel is center.
        if (('LUVOIR-A_APLC_10bw_smallFPM_2021-05-05_Dyn10pm-nostaticabb' in cdir) or ('LUVOIR-A_APLC_18bw_medFPM_2021-05-07_Dyn10pm-nostaticabb' in cdir) or ('LUVOIR-B-VC6_timeseries' in cdir) or ('LUVOIR-B_VC6_timeseries' in cdir)):
            self.stellar_intens_1 = self.stellar_intens_1[:, 1:, 1:]
            if (self.hasref == True):
                self.stellar_intens_2 = self.stellar_intens_2[:, 1:, 1:]
            self.offax_psf = self.offax_psf[:, :-1, 1:]
        else:
            raise UserWarning('Please validate centering for this unknown coronagraph model')
        
        # Simulation parameters.
        self.cdir = cdir
        self.imsz = self.stellar_intens_1.shape[1]
        self.fill = np.log(1e-100)
        
        # Interpolate coronagraph model.
        self.inter_onaxis_1 = interp1d(self.stellar_intens_diam_list, np.log(self.stellar_intens_1), kind='cubic', axis=0, bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        if (self.hasref == True):
            self.inter_onaxis_2 = interp1d(self.stellar_intens_diam_list, np.log(self.stellar_intens_2), kind='cubic', axis=0, bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        if ((self.type == '1dx') or (self.type == '1dxo')):
            self.inter_offaxis = interp1d(self.offax_psf_offset_list[:, 0], np.log(self.offax_psf), kind='cubic', axis=0, bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        elif ((self.type == '1dy') or (self.type == '1dyo')):
            self.inter_offaxis = interp1d(self.offax_psf_offset_list[:, 1], np.log(self.offax_psf), kind='cubic', axis=0, bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        elif (self.type == '2dq'):
            zz_temp = self.offax_psf.reshape(self.xx.shape[0], self.yy.shape[0], self.offax_psf.shape[1], self.offax_psf.shape[2])
            
            # Reflect PSFs to cover the x = 0 and y = 0 axes.
            xx = np.append(-self.xx[0], self.xx)
            yy = np.append(-self.yy[0], self.yy)
            zz = np.pad(zz_temp, ((1, 0), (1, 0), (0, 0), (0, 0)))
            zz[0, 1:] = zz_temp[0, :, ::-1, :]
            zz[1:, 0] = zz_temp[:, 0, :, ::-1]
            zz[0, 0] = zz_temp[0, 0, ::-1, ::-1]
            
            self.inter_offaxis = RegularGridInterpolator((xx, yy), np.log(zz), method='linear', bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        else:
            zz = self.offax_psf.reshape(self.xx.shape[0], self.yy.shape[0], self.offax_psf.shape[1], self.offax_psf.shape[2])
            self.inter_offaxis = RegularGridInterpolator((xx, yy), np.log(zz), method='linear', bounds_error=False, fill_value=self.fill) # interpolate in log-space to avoid negative values
        
        # Photometric parameters.
        head = pyfits.getheader(cdir+'stellar_intens_1.fits', 0)
        self.fobs = head['OBSCURED'] # fractional obscuration
        print('   Fractional obscuration = %.3f' % self.fobs)
        self.bp = (head['MAXLAM']-head['MINLAM'])/head['LAMBDA'] # fractional bandpass
        print('   Fractional bandpass = %.3f' % self.bp)
        self.insttp = 1. # instrument throughput
        print('   Instrument throughput = %.3f' % self.insttp)
        self.corotp = self.get_corotp(plot=False)
        print('   Coronagraph throughput = %.3f' % self.corotp)
        
        pass
    
    def set_pang(self,
                 pang=0.): # deg
        
        # Set position angle.
        self.pang = self.cang-pang # deg
        
        pass
    
    def get_corotp(self,
                   rapt=0.8, # lambda/D
                   oversample=100,
                   plot=True):
        
        # Compute off-axis PSF at half of the maximum separation.
        if (len(self.xx) != 1):
            rr = self.xx[self.xx.shape[0]//2] # lambda/D
        elif (len(self.yy) != 1):
            rr = self.yy[self.yy.shape[0]//2] # lambda/D
        else:
            raise UserWarning('Array offax_psf_offset_list should have more than 1 unique element for at least one axis')
        if ((self.type == '1dx') or (self.type == '1dxo')):
            pos_lod = np.array([rr, self.yy[0]]) # lambda/D
            imgs = np.exp(self.inter_offaxis(rr))
        elif ((self.type == '1dy') or (self.type == '1dyo')):
            pos_lod = np.array([self.xx[0], rr]) # lambda/D
            imgs = np.exp(self.inter_offaxis(rr))
        
        # Compute aperture position and radius on subarray in pixels.
        Npix = int(np.ceil(3*rapt/self.imsc))
        pos_pix = (pos_lod/self.imsc+(imgs.shape[0]-1)/2.).astype(int)
        subarr = imgs[pos_pix[1]-Npix:pos_pix[1]+Npix+1, pos_pix[0]-Npix:pos_pix[0]+Npix+1]
        pos_subarr = (pos_lod/self.imsc+(imgs.shape[0]-1)/2.)-pos_pix+Npix
        rad_subarr = rapt/self.imsc
        
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
        
        # Compute coronagraph throughput.
        corotp = np.sum(subarr_zoom[aptr])
        
        # Plot.
        if (plot == True):
            ext = self.imsz/2.*self.imsc # lambda/D
            f, ax = plt.subplots(1, 4, figsize=(4.8*4, 3.6*1))
            p0 = ax[0].imshow(imgs, origin='lower', extent=(-ext, ext, -ext, ext))
            c0 = plt.colorbar(p0, ax=ax[0])
            c0.set_label('Relative flux', rotation=270, labelpad=20)
            a0 = plt.Circle(pos_lod, rapt, fc='none', ec='red')
            ax[0].add_patch(a0)
            ax[0].set_xlabel('$\Delta$RA [$\lambda/D$]')
            ax[0].set_ylabel('$\Delta$DEC [$\lambda/D$]')
            ax[0].set_title('Coronagraph model')
            p1 = ax[1].imshow(subarr, origin='lower')
            c1 = plt.colorbar(p1, ax=ax[1])
            c1.set_label('Relative flux', rotation=270, labelpad=20)
            a1 = plt.Circle(pos_subarr, rad_subarr, fc='none', ec='red')
            ax[1].add_patch(a1)
            ax[1].set_xlabel('$\Delta$RA [pix]')
            ax[1].set_ylabel('$\Delta$DEC [pix]')
            ax[1].set_title('PSF')
            p2 = ax[2].imshow(subarr_zoom, origin='lower')
            c2 = plt.colorbar(p2, ax=ax[2])
            c2.set_label('Relative flux', rotation=270, labelpad=20)
            a2 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
            ax[2].add_patch(a2)
            ax[2].set_xlabel('$\Delta$RA [pix]')
            ax[2].set_ylabel('$\Delta$DEC [pix]')
            ax[2].set_title('Oversampled PSF')
            p3 = ax[3].imshow(aptr, origin='lower')
            c3 = plt.colorbar(p3, ax=ax[3])
            c3.set_label('Transmission', rotation=270, labelpad=20)
            a3 = plt.Circle(pos_subarr_zoom, rad_subarr_zoom, fc='none', ec='red')
            ax[3].add_patch(a3)
            ax[3].set_xlabel('$\Delta$RA [pix]')
            ax[3].set_ylabel('$\Delta$DEC [pix]')
            ax[3].set_title('Oversampled aperture')
            plt.tight_layout()
            plt.show()
            plt.close()
        
        return corotp
    
    def load_scene(self,
                   path,
                   diam, # m
                   time, # yr
                   wave, # micron
                   odir,
                   haystacks=False):
        
        if (self.verbose == True):
            print('Loading scene...')
        
        # Load scene.
        if (haystacks == False):
            self.scene = load_scene_jens.scene(path, time, wave)
            self.name = path[path.rfind('/')+1:-5]
            self.diam = diam # m
            self.dist = pyfits.getheader(path, 3)['DIST']
            self.vmag = pyfits.getheader(path, 3)['VMAG']
            self.angdiam = pyfits.getheader(path, 3)['ANGDIAM']
        else:
            self.scene = load_scene_haystacks.scene(path, time, wave)
            self.name = path[path.rfind('/')+1:-5]
            self.diam = diam # m
            self.dist = pyfits.getheader(path, 3)['DIST']
            self.vmag = pyfits.getheader(path, 3)['VMAG']
            self.angdiam = pyfits.getheader(path, 3)['ANGDIAM']
        self.odir = odir
        
        # Fix negative flux values.
        mask = self.scene.fstar < 0.
        if (np.sum(mask) > 0.):
            self.scene.fstar[mask] = 0.
            print('   WARNING: fixed negative flux values for star')
        mask = self.scene.fplanet < 0.
        if (np.sum(mask) > 0.):
            self.scene.fplanet[mask] = 0.
            print('   WARNING: fixed negative flux values for planet')
        
        # Unit conversion factor from Jy to ph/s.
        # Flux F_nu is given in Jy = 10^(-26)*W/Hz/m^2.
        # Flux F_lambda = F_nu*c/lambda^2.
        # Photon energy E = h*c/lambda.
        # Count rate in ph/s = 10^(-26)*F_nu*A*dl/h/lambda*T.
        tp = np.array([self.insttp]*self.scene.Nwave)
        area = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2 (effective mirror area)
        dl = self.bp*wave*1e-6 # m
        self.phot = 1e-26*area*dl/6.626e-34/(wave*1e-6)*tp # ph/s for source with 1 Jy
        
        return self.name
    
    def add_star(self,
                 ref=False,
                 tag='sci',
                 save=True):
        
        if (self.verbose == True):
            print('Adding star...')
        
        # Compute star contrast.
        angdiam_lod = self.angdiam*mas2rad/(self.scene.wave*1e-6/self.diam) # lambda/D
        if (ref == False):
            temp = np.exp(self.inter_onaxis_1(angdiam_lod))
        else:
            temp = np.exp(self.inter_onaxis_2(angdiam_lod))
        temp = temp.T
        
        # Compute star flux.
        star = np.zeros((self.scene.Ntime, self.scene.Nwave, self.imsz, self.imsz)) # ph/s
        for i in range(self.scene.Ntime):
            star[i] = np.multiply(temp, self.scene.fstar[i]*self.phot).T # ph/s
        
        # Save star images.
        if (save == True):
            hdu0 = pyfits.PrimaryHDU(star) # ph/s
            hdu0.header['PIXSCALE'] = self.imsc # lambda/D
            hdu0.header['DIAM'] = self.diam # m
            hdu0.header['AREA'] = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2
            hdu0.header['BP'] = self.bp
            hdu0.header['INSTTP'] = self.insttp
            hdu0.header['COROTP'] = self.corotp
            hdu0.header['DIST'] = self.dist
            hdu0.header['VMAG'] = self.vmag
            hdu0.header['ANGDIAM'] = self.angdiam
            hdu0.header['EXTNAME'] = 'STAR'
            hdu1 = pyfits.ImageHDU(self.scene.time)
            hdu1.header['EXTNAME'] = 'TIME'
            hdu2 = pyfits.ImageHDU(self.scene.wave)
            hdu2.header['EXTNAME'] = 'WAVE'
            hdul = pyfits.HDUList([hdu0, hdu1, hdu2])
            path = self.odir+self.name+'/RAW/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            hdul.writeto(path+tag+'_star.fits', overwrite=True)
            hdul.close()
        
        return star
    
    def add_plan(self,
                 tag='sci',
                 save=True):
        
        if (self.verbose == True):
            print('Adding planets...')
        
        # Compute planet separations and position angles.
        plan_offs = (self.scene.xyplanet-self.scene.xystar)*self.scene.pixscale # mas
        plan_seps = np.sqrt(np.sum(plan_offs**2, axis=2)) # mas
        plan_angs = np.rad2deg(np.arctan2(plan_offs[:, :, 0], plan_offs[:, :, 1])) # deg
        if (self.pang != 0.):
            plan_angs += self.pang # deg
            plan_offs[:, :, 0] = np.multiply(plan_seps, np.sin(np.deg2rad(plan_angs))) # mas
            plan_offs[:, :, 1] = np.multiply(plan_seps, np.cos(np.deg2rad(plan_angs))) # mas
        
        # Compute planet flux.
        plan = np.zeros((self.scene.Ntime, self.scene.Nwave, self.imsz, self.imsz)) # ph/s
        for i in range(self.scene.Nplanets):
            sys.stdout.write('\r   Finished %.0f of %.0f planets' % (i, self.scene.Nplanets))
            sys.stdout.flush()
            if ((self.type == '1dx') or (self.type == '1dy')):
                wave_inv = 1./(self.scene.wave*1e-6) # 1/m
                temp = np.dot(plan_seps[i][:, None]*mas2rad, wave_inv[None])*self.diam # lambda/D
                temp = np.exp(self.inter_offaxis(temp))
                for j in range(self.scene.Ntime):
                    temp[j] = np.exp(rotate(np.log(temp[j]), plan_angs[i, j]-90., axes=(1, 2), reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    temp[j] = np.multiply(temp[j].T, self.scene.fplanet[i, j]*self.phot).T # ph/s
            elif (self.type == '1dxo'):
                wave_inv = 1./(self.scene.wave*1e-6) # 1/m
                seps = np.dot(plan_seps[i][:, None]*mas2rad, wave_inv[None])*self.diam # lambda/D
                temp = np.sqrt(seps**2-self.yy[0]**2) # lambda/D
                angs = np.rad2deg(np.arcsin(self.yy[0]/seps)) # deg
                temp = np.exp(self.inter_offaxis(temp))
                for j in range(self.scene.Ntime):
                    for k in range(self.scene.Nwave):
                        temp[j, k] = np.exp(rotate(np.log(temp[j, k]), plan_angs[i, j]-90.+angs[j, k], axes=(0, 1), reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    temp[j] = np.multiply(temp[j].T, self.scene.fplanet[i, j]*self.phot).T # ph/s
            elif (self.type == '1dyo'):
                wave_inv = 1./(self.scene.wave*1e-6) # 1/m
                seps = np.dot(plan_seps[i][:, None]*mas2rad, wave_inv[None])*self.diam # lambda/D
                temp = np.sqrt(seps**2-self.xx[0]**2) # lambda/D
                angs = np.rad2deg(np.arcsin(self.xx[0]/seps)) # deg
                temp = np.exp(self.inter_offaxis(temp))
                for j in range(self.scene.Ntime):
                    for k in range(self.scene.Nwave):
                        temp[j, k] = np.exp(rotate(np.log(temp[j, k]), plan_angs[i, j]-90.+angs[j, k], axes=(0, 1), reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    temp[j] = np.multiply(temp[j].T, self.scene.fplanet[i, j]*self.phot).T # ph/s
            elif (self.type == '2dq'):
                wave_inv = 1./(self.scene.wave*1e-6) # 1/m
                temp = np.dot(plan_offs[i][:, :, None]*mas2rad, wave_inv[None])*self.diam # lambda/D
                temp = np.swapaxes(temp, 1, 2) # lambda/D
                temp = temp[:, :, ::-1] # lambda/D
                offs = temp.copy() # lambda/D
                temp = np.exp(self.inter_offaxis(np.abs(temp)))
                mask = offs[:, :, 0] < 0.
                temp[mask] = temp[mask, ::-1, :]
                mask = offs[:, :, 1] < 0.
                temp[mask] = temp[mask, :, ::-1]
                for j in range(self.scene.Ntime):
                    temp[j] = np.multiply(temp[j].T, self.scene.fplanet[i, j]*self.phot).T # ph/s
            else:
                wave_inv = 1./(self.scene.wave*1e-6) # 1/m
                temp = np.dot(plan_offs[i][:, :, None]*mas2rad, wave_inv[None])*self.diam # lambda/D
                temp = np.swapaxes(temp, 1, 2) # lambda/D
                temp = temp[:, :, ::-1] # lambda/D
                temp = np.exp(self.inter_offaxis(temp))
                for j in range(self.scene.Ntime):
                    temp[j] = np.multiply(temp[j].T, self.scene.fplanet[i, j]*self.phot).T # ph/s
            plan += temp # ph/s
        sys.stdout.write('\r   Finished %.0f of %.0f planets' % (i+1, self.scene.Nplanets))
        sys.stdout.flush()
        print('')
        
        # Save planet images.
        if (save == True):
            hdu0 = pyfits.PrimaryHDU(plan) # ph/s
            hdu0.header['PIXSCALE'] = self.imsc # lambda/D
            hdu0.header['DIAM'] = self.diam # m
            hdu0.header['AREA'] = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2
            hdu0.header['BP'] = self.bp
            hdu0.header['INSTTP'] = self.insttp
            hdu0.header['COROTP'] = self.corotp
            hdu0.header['DIST'] = self.dist
            hdu0.header['VMAG'] = self.vmag
            hdu0.header['ANGDIAM'] = self.angdiam
            hdu0.header['EXTNAME'] = 'STAR'
            hdu1 = pyfits.ImageHDU(self.scene.time)
            hdu1.header['EXTNAME'] = 'TIME'
            hdu2 = pyfits.ImageHDU(self.scene.wave)
            hdu2.header['EXTNAME'] = 'WAVE'
            hdul = pyfits.HDUList([hdu0, hdu1, hdu2])
            path = self.odir+self.name+'/RAW/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            hdul.writeto(path+tag+'_plan.fits', overwrite=True)
            hdul.close()
        
        return plan
    
    def add_disk(self,
                 tag='sci',
                 save=True):
        
        if (self.verbose == True):
            print('Adding disk...')
        
        # Load data cube of spatially dependent PSFs.
        ddir = 'coronagraphs_psflib/'
        if (not os.path.exists(ddir)):
            os.makedirs(ddir)
        path = ddir+self.cdir[self.cdir[:-1].rfind('/')+1:-1]+'.npy'
        try:
            psfs = np.load(path, allow_pickle=True)
            print('   Loaded data cube of spatially dependent PSFs')
        
        # Compute data cube of spatially dependent PSFs.
        except:
            
            # Compute pixel grid.
            ramp = (np.arange(self.imsz)-((self.imsz-1)//2))*self.imsc # lambda/D
            xx, yy = np.meshgrid(ramp, ramp) # lambda/D
            rr = np.sqrt(xx**2+yy**2) # lambda/D
            tt = np.rad2deg(np.arctan2(xx, yy)) # deg
            
            # Compute pixel grid contrast.
            print('   Computing data cube of spatially dependent PSFs')
            psfs = np.zeros((rr.shape[0], rr.shape[1], self.imsz, self.imsz))
            Npsfs = np.prod(rr.shape)
            counter = 0
            for i in range(rr.shape[0]):
                for j in range(rr.shape[1]):
                    if (counter % 100 == 0):
                        sys.stdout.write('\r   Finished %.0f of %.0f PSFs' % (counter, Npsfs))
                        sys.stdout.flush()
                    counter += 1
                    if (self.type == '1dx'):
                        temp = self.inter_offaxis(rr[i, j])
                        temp = np.exp(rotate(temp, tt[i, j]-90., reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    elif (self.type == '1dy'):
                        temp = self.inter_offaxis(rr[i, j])
                        temp = np.exp(rotate(temp, tt[i, j], reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    elif (self.type == '1dxo'):
                        temp = np.sqrt(rr[i, j]**2-self.yy[0]**2) # lambda/D
                        angs = np.rad2deg(np.arcsin(self.yy[0]/rr[i, j])) # deg
                        temp = self.inter_offaxis(temp)
                        temp = np.exp(rotate(temp, tt[i, j]-90.+angs, reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    elif (self.type == '1dyo'):
                        temp = np.sqrt(rr[i, j]**2-self.xx[0]**2) # lambda/D
                        angs = np.rad2deg(np.arcsin(self.xx[0]/rr[i, j])) # deg
                        temp = self.inter_offaxis(temp)
                        temp = np.exp(rotate(temp, tt[i, j]-angs, reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                    elif (self.type == '2dq'):
                        temp = np.array([yy[i, j], xx[i, j]]) # lambda/D
                        temp = np.exp(self.inter_offaxis(np.abs(temp))[0])
                        if (yy[i, j] < 0.):
                            temp = temp[::-1, :] # lambda/D
                        if (xx[i, j] < 0.):
                            temp = temp[:, ::-1] # lambda/D
                    else:
                        temp = np.array([yy[i, j], xx[i, j]]) # lambda/D
                        temp = np.exp(self.inter_offaxis(temp)[0])
                    psfs[i, j] = temp
            sys.stdout.write('\r   Finished %.0f of %.0f PSFs' % (Npsfs, Npsfs))
            sys.stdout.flush()
            print('')
            
            # Save data cube of spatially dependent PSFs.
            np.save(path, psfs, allow_pickle=True)
        
        # Rotate disk so that North is in the direction of the position angle.
        diskimage = self.scene.disk.copy() # Jy
        if (self.pang != 0.):
            diskimage = np.exp(rotate(np.log(diskimage), self.pang, axes=(3, 2), reshape=False, mode='nearest', order=5)) # interpolate in log-space to avoid negative values
        
        # Scale disk to units of lambda/D.
        wave_inv = 1./(self.scene.wave*1e-6) # 1/m
        fact = self.scene.pixscale*mas2rad*wave_inv*self.diam/self.imsc # lambda/D
        disk = np.zeros((self.scene.Ntime, self.scene.Nwave, self.imsz, self.imsz)) # ph/s
        counter_time = 0
        for i in range(self.scene.Ntime):
            counter_wave = 0
            for j in range(self.scene.Nwave):
                sys.stdout.write('\r   Finished %.0f of %.0f times, %.0f of %.0f wavelengths' % (counter_time, self.scene.Ntime, counter_wave, self.scene.Nwave))
                sys.stdout.flush()
                counter_wave += 1
                temp = np.exp(zoom(np.log(diskimage[i, j]), fact[j], mode='nearest', order=5)) # interpolate in log-space to avoid negative values
                temp = temp/fact[j]**2
                
                # Center disk so that (imsz-1)/2 is center.
                if ((temp.shape[0] % 2 == 0) and (self.imsz % 2 != 0)):
                    temp = np.pad(temp, ((0, 1), (0, 1)), mode='edge')
                    temp = np.exp(shift(np.log(temp), (0.5, 0.5), order=5)) # interpolate in log-space to avoid negative values
                    temp = temp[1:-1, 1:-1]
                elif ((temp.shape[0] % 2 != 0) and (self.imsz % 2 == 0)):
                    temp = np.pad(temp, ((0, 1), (0, 1)), mode='edge')
                    temp = np.exp(shift(np.log(temp), (0.5, 0.5), order=5)) # interpolate in log-space to avoid negative values
                    temp = temp[1:-1, 1:-1]
                
                # Crop disk to coronagraph model size.
                if (temp.shape[0] > self.imsz):
                    nn = (temp.shape[0]-self.imsz)//2
                    temp = temp[nn:-nn, nn:-nn]
                else:
                    nn = (self.imsz-temp.shape[0])//2
                    temp = np.pad(temp, ((nn, nn), (nn, nn)), mode='edge')
                disk[i, j] = util.tdot(temp, psfs)*self.phot[j] # ph/s
            counter_time += 1
        sys.stdout.write('\r   Finished %.0f of %.0f times, %.0f of %.0f wavelengths' % (self.scene.Ntime, self.scene.Ntime, self.scene.Nwave, self.scene.Nwave))
        sys.stdout.flush()
        print('')
        
       # Save disk images.
        if (save == True):
            hdu0 = pyfits.PrimaryHDU(disk) # ph/s
            hdu0.header['PIXSCALE'] = self.imsc # lambda/D
            hdu0.header['DIAM'] = self.diam # m
            hdu0.header['AREA'] = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2
            hdu0.header['BP'] = self.bp
            hdu0.header['INSTTP'] = self.insttp
            hdu0.header['COROTP'] = self.corotp
            hdu0.header['DIST'] = self.dist
            hdu0.header['VMAG'] = self.vmag
            hdu0.header['ANGDIAM'] = self.angdiam
            hdu0.header['EXTNAME'] = 'DISK'
            hdu1 = pyfits.ImageHDU(self.scene.time)
            hdu1.header['EXTNAME'] = 'TIME'
            hdu2 = pyfits.ImageHDU(self.scene.wave)
            hdu2.header['EXTNAME'] = 'WAVE'
            hdul = pyfits.HDUList([hdu0, hdu1, hdu2])
            path = self.odir+self.name+'/RAW/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            hdul.writeto(path+tag+'_disk.fits', overwrite=True)
            hdul.close()
        
        return disk 
    
    def sim_sci(self,
                add_star=True,
                add_plan=True,
                add_disk=True,
                tag='sci',
                save_all=True):
        
        if (self.verbose == True):
            print('Simulating science...')
        
        # Initialize science images.
        imgs = np.zeros((self.scene.Ntime, self.scene.Nwave, self.imsz, self.imsz)) # ph/s
        
        # Add star.
        if (add_star == True):
            imgs += self.add_star(ref=False,
                                  tag=tag,
                                  save=save_all)
        
        # Add planets.
        if (add_plan == True):
            imgs += self.add_plan(tag=tag,
                                  save=save_all)
        
        # Add disk.
        if (add_disk == True):
            imgs += self.add_disk(tag=tag,
                                  save=save_all)
        
        # Save science images.
        hdu0 = pyfits.PrimaryHDU(imgs)
        hdu0.header['PIXSCALE'] = self.imsc # lambda/D
        hdu0.header['DIAM'] = self.diam # m
        hdu0.header['AREA'] = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2
        hdu0.header['BP'] = self.bp
        hdu0.header['INSTTP'] = self.insttp
        hdu0.header['COROTP'] = self.corotp
        hdu0.header['DIST'] = self.dist
        hdu0.header['VMAG'] = self.vmag
        hdu0.header['ANGDIAM'] = self.angdiam
        hdu0.header['EXTNAME'] = 'IMGS'
        hdu1 = pyfits.ImageHDU(self.scene.time)
        hdu1.header['EXTNAME'] = 'TIME'
        hdu2 = pyfits.ImageHDU(self.scene.wave)
        hdu2.header['EXTNAME'] = 'WAVE'
        hdul = pyfits.HDUList([hdu0, hdu1, hdu2])
        path = self.odir+self.name+'/RAW/'
        if (not os.path.exists(path)):
            os.makedirs(path)
        hdul.writeto(path+tag+'_imgs.fits', overwrite=True)
        hdul.close()
        
        return imgs
    
    def sim_ref(self,
                rang=0., # deg
                add_star=True,
                add_plan=True,
                add_disk=True,
                tag='ref_psf',
                save_all=False):
        
        if (self.hasref == True):
            if (self.verbose == True):
                print('Simulating reference...')
            
            # Initialize reference images.
            imgs = np.zeros((self.scene.Ntime, self.scene.Nwave, self.imsz, self.imsz)) # ph/s
            
            # Roll telescope.
            pang = self.pang # deg
            self.pang -= rang # deg
            
            # Add star.
            if (add_star == True):
                imgs += self.add_star(ref=True,
                                      tag=tag,
                                      save=save_all)
            
            # Add planets.
            if (add_plan == True):
                imgs += self.add_plan(tag=tag,
                                      save=save_all)
            
            # Add disk.
            if (add_disk == True):
                imgs += self.add_disk(tag=tag,
                                      save=save_all)
            
            # Save reference images.
            hdu0 = pyfits.PrimaryHDU(imgs)
            hdu0.header['PIXSCALE'] = self.imsc # lambda/D
            hdu0.header['DIAM'] = self.diam # m
            hdu0.header['AREA'] = np.pi*self.diam**2/4.*(1.-self.fobs) # m^2
            hdu0.header['BP'] = self.bp
            hdu0.header['INSTTP'] = self.insttp
            hdu0.header['COROTP'] = self.corotp
            hdu0.header['DIST'] = self.dist
            hdu0.header['VMAG'] = self.vmag
            hdu0.header['ANGDIAM'] = self.angdiam
            hdu0.header['EXTNAME'] = 'IMGS'
            hdu1 = pyfits.ImageHDU(self.scene.time)
            hdu1.header['EXTNAME'] = 'TIME'
            hdu2 = pyfits.ImageHDU(self.scene.wave)
            hdu2.header['EXTNAME'] = 'WAVE'
            hdul = pyfits.HDUList([hdu0, hdu1, hdu2])
            path = self.odir+self.name+'/RAW/'
            if (not os.path.exists(path)):
                os.makedirs(path)
            hdul.writeto(path+tag+'_imgs.fits', overwrite=True)
            hdul.close()
            
            # Roll telescope.
            self.pang = pang # deg
        
        else:
            print('   WARNING: coronagraph model does not contain reference PSF')
        
        return imgs
