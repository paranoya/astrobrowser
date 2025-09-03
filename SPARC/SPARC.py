#!/usr/bin/env python
# coding: utf-8

# # Astrobrowser - Spitzer Photometry and Accurate Rotation Curves (SPARC)
# 
# Retrieve flux-calibrated photometric images in SDSS $g$, $r$, and $i$ bands (based on Giordano et al. zero points) and derive surface brightness profiles.

# # 1. Initialisation

# External libraries

# In[1]:


#%matplotlib ipympl
import os
import numpy as np
from scipy import ndimage
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
from astropy.coordinates import SkyCoord, get_icrs_coordinates
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import QTable
from astropy import units as u
from astropy import constants as c
from scripts import astrobrowser



# Utility functions

# In[3]:


show_plots = True
show_plots = False
plt.ioff()


# In[4]:


def new_figure(fig_name, figsize=(12, 8), nrows=1, ncols=1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, suptitle=True):
    plt.close(fig_name)
    fig = plt.figure(fig_name, figsize=figsize, layout="constrained")
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                        sharex=sharex, sharey=sharey,
                        gridspec_kw=gridspec_kw
                       )
    #fig.set_tight_layout(True)
    for ax in axes.flat:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
        ax.tick_params(which='major', direction='inout', length=8, grid_alpha=.3)
        ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
        ax.grid(True, which='both')

    if suptitle is True:
        fig.suptitle(fig_name)
    elif suptitle is not False and suptitle is not None:
        fig.suptitle(suptitle)
    
    return fig, axes


# In[5]:


def test_dir(dir_name):
    if not os.path.isdir(dir_name):
        print(f'>> WARNING: Creating directory "{dir_name}"')
        os.makedirs(dir_name)
    return(dir_name)


# Directories

# In[6]:


input_dir = 'SPARC'
output_dir = test_dir(os.path.join(input_dir, 'output'))
maps_dir = test_dir(os.path.join(output_dir, 'maps'))
plots_dir = test_dir(os.path.join(output_dir, 'plots'))
profiles_dir = test_dir(os.path.join(output_dir, 'profiles'))
params_dir = test_dir(os.path.join(output_dir, 'parameters'))


# # 2. Observations

# ## SPARC catalogues
# 
# I had to edit the main catalogue a bit to conform to the Machine Readable Table (MRT) format.
# 
# It would also be useful to store (ra, dec) instead of fetching them from Sesame.

# In[7]:


SPARC_catalogue = QTable.read(os.path.join(input_dir, 'SPARC_clean.mrt'), format='mrt')


# In[8]:


SPARC_catalogue


# In[9]:


SPARC_models = QTable.read(os.path.join(input_dir, 'MassModels_Lelli2016c.mrt'), format='mrt')


# In[10]:


SPARC_models


# ## SDSS HiPS skymaps

# In[11]:


class HiPS_skymap(object):
    
    def __init__(self, hips_service_url, units, beam=None):
        '''Intensity map in Hierarchical Progressive Survey (HiPS) format'''
        
        print(f'> {hips_service_url}')
        self.url = hips_service_url
        self.properties = astrobrowser.get_hips_proprties(hips_service_url)
        if self.properties is None:
            print('  ERROR: HiPS properties not available!')
            raise -1
        if 'hips_pixel_scale' in self.properties:
            self.hips_pixel = float(self.properties['hips_pixel_scale']) * u.deg
        else:
            print('  ERROR: HiPS pixel size not available!')
            raise -1

        if beam is None:
            if 's_pixel_scale' in self.properties:
                original_pixel = float(self.properties['s_pixel_scale']) * u.deg
                self.beam = original_pixel**2
            else:
                self.beam = self.hips_pixel**2
                print(f'  WARNING: original pixel size not available! using HiPS size = {self.hips_pixel.to_value(u.arcsec)} arcsec')
        else:
            self.beam = beam
            original_pixel_beam = np.sqrt(beam)
            if 's_pixel_scale' in self.properties:
                original_pixel_properties = float(self.properties['s_pixel_scale']) * u.deg
                if not u.isclose(original_pixel_beam, original_pixel_properties):
                    print(f'  WARNING: {original_pixel_beam} is different from {original_pixel_properties} ({original_pixel_properties.to(original_pixel_beam.unit)})')

        self.intensity_units = units
        if u.get_physical_type(units) == 'spectral flux density':
            self.intensity_units = units / self.beam
        
        print(f'  HiPS pixel = {self.hips_pixel.to(u.arcsec):.4f}, original = {np.sqrt(self.beam).to(u.arcsec):.4f}',
              f'; units = {self.intensity_units.to(u.uJy/u.arcsec**2):.2f} = {self.intensity_units.to(u.MJy/u.sr):.4f}')


# These SDSS measurements correpond to surface brightness in nanomaggies per beam (original pixel)

# In[12]:


nanomaggies = 3.631e-6*u.Jy
beam = (0.39564 * u.arcsec)**2
pivot_wavelength = {
    'g': 4702.50 * u.Angstrom,
    'r': 6175.58 * u.Angstrom,
    'i': 7489.98 * u.Angstrom,
}
solar_units_factor = (4*np.pi*u.sr) * c.c  # divide by the pilot wavelength to convert from intensity (e.g. nanomaggies/beam) to luminosity surface brightness (Lsun/pc^2)


# In[13]:


SDSS_skymaps = {
    'g': HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-g', nanomaggies, beam),
    'r': HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-r', nanomaggies, beam),
    'i': HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-i', nanomaggies, beam),
}


# ## Download images

# In[14]:


def fectch_images(galaxy, skymaps, overwrite=False, fig_name=None, max_size=1000):
    """
    Call the AstroBrowser to download HiPS cutouts, or
    read them from disk if they are present.
    """
    cutout_pixel = np.inf
    for band in skymaps:
        if skymaps[band].hips_pixel < cutout_pixel:
            cutout_pixel = skymaps[band].hips_pixel
            
    radius = galaxy['Rdisk'] * np.log(galaxy['SBdisk'].to_value(u.Lsun/u.pc**2)) # estimate 1 Msun/pc^2
    radius *= 1.5 # some buffer
    radius = np.arcsin(radius / galaxy['D']) # angle
    if 2*radius > max_size*cutout_pixel:
        cutout_pixel = 2*radius / max_size
    
    header = {}
    data = {}
    try:
        position = get_icrs_coordinates(galaxy['Galaxy'])
    except:
        print(f"  Name {galaxy['Galaxy']} not found!")
        for band in skymaps:
            header[band] = None
            data[band] = None
        return header, data
    
    for band in skymaps:
        cutout_file = os.path.join(maps_dir, f"{galaxy['Galaxy']}_{band}.fits")
        if overwrite or not os.path.isfile(cutout_file):
            print(f"- Downloading {cutout_file}... (please be patient)")
            header[band], data[band] = astrobrowser.get_cutout(
                skymaps[band].url,
                position.ra.deg, position.dec.deg,
                radius.to_value(u.arcsec), cutout_pixel.to_value(u.arcsec),
                cutout_file, overwrite=True)
            if header[band] is None:
                for bnd in skymaps:
                    header[bnd] = None
                    data[bnd] = None
                return header, data
        else:
            print(f'- Reading "{cutout_file}"')
            with fits.open(cutout_file) as hdu:
                header[band] = hdu[0].header
                data[band] = hdu[0].data

    if fig_name is not None and header[next(iter(header))] is not None:
        plt.close(fig_name)
        n_bands = len(data)
        fig = plt.figure(fig_name, figsize=(6*n_bands, 5))
        for idx, band in enumerate(data):
            wcs = WCS(header[band])
            ax = fig.add_subplot(1, n_bands, idx+1)#, projection=wcs)
            ax.set_title(band)
            #img = data[band] * (nanomaggies/beam).to_value(3631*u.Jy/u.arcsec**2)
            #im = ax.imshow(img, origin='lower', cmap='nipy_spectral', norm=colors.Normalize(-1e-10, 1e-9))
            img = -2.5*np.log10(data[band] * (nanomaggies/beam).to_value(3631*u.Jy/u.arcsec**2))
            im = ax.imshow(img, origin='lower', interpolation='nearest', cmap='nipy_spectral', norm=colors.Normalize(17.5, 26.5))
            cb = plt.colorbar(im, ax=ax, shrink=.9)
        fig.savefig(os.path.join(plots_dir, f'{fig_name}.png'), facecolor='white')
        if not show_plots:
            plt.close(fig_name)

    return header, data

#header, data = fectch_images(galaxy, SDSS_skymaps, fig_name=f"{galaxy['Galaxy']}_cutouts")


# # 3. Analysis

# ## Mass-to-light ratios
# Obtained from Garcia-Benito et al.

# In[15]:


mass_to_light_ratio = {}
mass_to_light_ratio['ggr'] = (-0.88, 1.88)
mass_to_light_ratio['ggi'] = (-0.99, 1.29)
mass_to_light_ratio['gri'] = (-1.08, 3.74)
mass_to_light_ratio['rgr'] = (-0.70, 1.49)
mass_to_light_ratio['rgi'] = (-0.79, 1.03)
mass_to_light_ratio['rri'] = (-0.86, 2.98)
mass_to_light_ratio['igr'] = (-0.69, 1.31)
mass_to_light_ratio['igi'] = (-0.77, 0.90)
mass_to_light_ratio['iri'] = (-0.83, 2.60)


# In[16]:


def estimate_stellar_surface_density(data, fig_name=None, profiles_table=None):
    """
    Estimate stellar surface density from SDSS images,
    according to RGB M/L ratios.
    """
    mass = np.full((len(mass_to_light_ratio),)+data[next(iter(data))].shape, np.nan)
    for idx, mass_map in enumerate(mass_to_light_ratio):
        if mass_map[0] in data and mass_map[1] in data and mass_map[2] in data:
            a, b = mass_to_light_ratio[mass_map]
            mass[idx] = 10**(a - b * 2.5 * np.log10(data[mass_map[1]]/data[mass_map[2]]))
            mass[idx] *= data[mass_map[0]]
            mass[idx] *= (solar_units_factor*nanomaggies/beam/pivot_wavelength[mass_map[0]]).to_value(u.Lsun/u.pc**2)
    
    if profiles_table is not None:
        for idx, mass_map in enumerate(mass_to_light_ratio):
            profiles_table.add_column(mass[idx] << u.Msun/u.pc**2, name=f'{mass_map[0]} and ({mass_map[1]}-{mass_map[2]})')
    
    if fig_name is not None:
        plt.close(fig_name)
        fig = plt.figure(fig_name, figsize=(14, 12))
        #norm = colors.Normalize(1, 5)  # for M/L ratio
        norm = colors.LogNorm(3, 3e3)  # for surface density (Msun/pc^2)
        for idx, mass_map in enumerate(mass_to_light_ratio):
            if mass_map[0] in data and mass_map[1] in data and mass_map[2] in data:
                ax = fig.add_subplot(3, 3, idx+1)
                ax.set_title(f'$\Sigma$ from {mass_map[0]} and ({mass_map[1]}-{mass_map[2]})')
                im = ax.imshow(mass[idx], origin='lower', cmap='nipy_spectral', norm=norm)
                cb = plt.colorbar(im, ax=ax, shrink=.9)
        fig.savefig(os.path.join(plots_dir, f'{fig_name}.png'), facecolor='white')
        if not show_plots:
            plt.close()
    return np.median(mass, axis=0), np.std(np.log10(mass), axis=0)
    #p16, p50, p84 = np.percentile(mass, [16, 50, 84], axis=0)
    #return p50, np.log10((p84-p16)/2/p50)


# ## Fit ellipse

# In[17]:


class Ellipse(object):
    
    def __init__(self,
                 data,
                 center_seed=None, recenter=False, inner_radius=None, max_iter=10,
                 fig_name=None):
        
        # Set nan and negatives to 0
        valid_data = data.copy()
        valid_data = np.where(valid_data > 0, valid_data, 0)

        # Define inner region
        if center_seed is None:
            self.y0, self.x0 = data.shape
            self.x0 /= 2
            self.y0 /= 2
        else:
            self.x0, self.y0 = center_seed
        x = np.arange(valid_data.shape[1]) - self.x0
        y = np.arange(valid_data.shape[0]) - self.y0
        r = np.sqrt((x**2)[np.newaxis, :] + (y**2)[:, np.newaxis])

        if inner_radius is None:
            print('WARNING: inner radius estimation is work in progress (TODO)')
            sorted_by_r = np.argsort(r.ravel())
            cumulative_data = np.cumsum(valid_data.ravel()[sorted_by_r])
            cumulative_data2 = np.cumsum((valid_data**2).ravel()[sorted_by_r])
            n = 1 + np.arange(cumulative_data.size)
            test_stat = cumulative_data2/n - (cumulative_data/n)**2
            fig, axes = new_figure('kkk')
            ax = axes[0, 0]
            ax.plot(r.ravel()[sorted_by_r], cumulative_data2/n, 'r--')
            ax.plot(r.ravel()[sorted_by_r], (cumulative_data/n)**2, 'b--')
            ax.plot(r.ravel()[sorted_by_r], test_stat, 'k-')
            inner_radius = np.sum(r * valid_data) / np.sum(valid_data)
            print('inner radius:', inner_radius, r.ravel()[sorted_by_r], n, cumulative_data)
        self.inner_mask = r < float(inner_radius)

        # Find isophote and recenter, if requested
        iteration = 0
        cm_moved = np.inf
        while cm_moved > 1 and iteration < max_iter:
            iteration += 1
            mask, x_new, y_new = self._find_isophote(valid_data, inner_radius)
            if recenter:
                cm_moved = np.sqrt((x_new-self.x0)**2 + (y_new-self.y0)**2)
                self.x0 = x_new
                self.y0 = y_new
                print(f'  Center at ({self.x0:.2f}, {self.y0:.2f}) moved {cm_moved:.4g} pix')
            else:
                cm_moved = 0
        print(f'> Ellipse centered at ({self.x0:.2f}, {self.y0:.2f})')
        x = np.arange(valid_data.shape[1]) - self.x0
        y = np.arange(valid_data.shape[0]) - self.y0
        r = np.sqrt((x**2)[np.newaxis, :] + (y**2)[:, np.newaxis])

        # Find mean isophote radius as a function of polar angle
        theta_r = np.linspace(0, np.pi, 181)
        mean_r = np.empty_like(theta_r)
        dummy_r = np.arange(1, np.min(valid_data.shape)//2)
        for i, zz in enumerate(theta_r):
            x_i = (self.x0 + dummy_r * np.cos(zz)).astype(int).clip(0, valid_data.shape[1]-1)
            y_i = (self.y0 + dummy_r * np.sin(zz)).astype(int).clip(0, valid_data.shape[0]-1)
            #total_weight = np.sum(mask[y_i, x_i])
            #mean_r_positive = np.sum(mask[y_i, x_i] * r[y_i, x_i]) / total_weight
            cumulative = np.cumsum(mask[y_i, x_i]).astype(float)
            cumulative /= cumulative[-1]
            mean_r_positive = np.interp(.5, cumulative, dummy_r)
            x_i = (self.x0 - dummy_r * np.cos(zz)).astype(int).clip(0, valid_data.shape[1]-1)
            y_i = (self.y0 - dummy_r * np.sin(zz)).astype(int).clip(0, valid_data.shape[0]-1)
            #total_weight = np.sum(mask[y_i, x_i])
            #mean_r_negative = np.sum(mask[y_i, x_i] * r[y_i, x_i]) / total_weight
            cumulative = np.cumsum(mask[y_i, x_i]).astype(float)
            cumulative /= cumulative[-1]
            mean_r_negative = np.interp(.5, cumulative, dummy_r)
            # Geometric mean of both sides
            #mean_r[i] = np.sqrt(mean_r_positive * mean_r_negative)
            # Harmonic mean of both sides
            mean_r[i] = 2 / (1/mean_r_positive + 1/mean_r_negative)
            #mean_r[i] = 1/np.sqrt(1/mean_r_positive**2 + 1/mean_r_negative**2)
        #mean_r = ndimage.median_filter(mean_r, 5)
        valid_r = mean_r > 0
        if np.count_nonzero(valid_r) > 0:
            mean_r = np.interp(theta_r, theta_r[valid_r], mean_r[valid_r])
        else:
            print("WARNING: No valid isophote found!")

        # Fit ellipse
        inv_r2 = 1/mean_r**2
        mean_value = np.mean(inv_r2)
        coeff_cos = np.mean(inv_r2 * np.cos(2*theta_r)) / np.mean(np.cos(2*theta_r)**2)
        coeff_sin = np.mean(inv_r2 * np.sin(2*theta_r)) / np.mean(np.sin(2*theta_r)**2)
        #model = mean_value + coeff_cos*np.cos(2*theta_r) + coeff_sin*np.sin(2*theta_r)
        amplitude = - np.sqrt(coeff_cos**2 + coeff_sin**2)
        coeff_cos /= amplitude
        coeff_sin /= amplitude
        #model = mean_value + amplitude * (coeff_cos*np.cos(2*theta_r) + coeff_sin*np.sin(2*theta_r))
        if coeff_sin < 0:
            self.theta_0 = np.pi - np.arccos(coeff_cos) / 2
        else:
            self.theta_0 = np.arccos(coeff_cos) / 2
        model = mean_value + amplitude * np.cos(2*(theta_r - self.theta_0))
        self.a = 1 / np.sqrt(mean_value + amplitude)
        self.b = 1 / np.sqrt(mean_value - amplitude)
        self.e = 1 - self.b/self.a
        print(f'  (a, b, $\\varphi_0$) = ({self.a:.2f} pix, {self.b:.2f} pix, {self.theta_0*180/np.pi:.2f} deg)')
        
        # Deprojection / profiles:
        r[r <= 0.] = 1e-6
        theta = np.where(y[:, np.newaxis] >= 0, np.arccos(x[np.newaxis, :]/r), 2*np.pi - np.arccos(x[np.newaxis, :]/r))
        theta -= self.theta_0
        theta[theta < 0] += 2*np.pi
        self.r_0 = r * np.sqrt(np.cos(theta)**2 + (np.sin(theta) * self.a/self.b)**2)

        # Plot figure
        if fig_name is not None:
            fig, axes = new_figure(fig_name, nrows=2, ncols=2, sharey=False,)
            
            ax = axes[0, 0]
            im = ax.imshow(mask, origin='lower', interpolation='nearest')
            x_r = self.x0 + mean_r * np.cos(theta_r)
            y_r = self.y0 + mean_r * np.sin(theta_r)
            ax.plot(x_r, y_r, 'k-')
            x_r = self.x0 + mean_r * np.cos(theta_r+np.pi)
            y_r = self.y0 + mean_r * np.sin(theta_r+np.pi)
            ax.plot(x_r, y_r, 'k-')
            self.plot(ax, self.a)
            #self.plot(ax, inner_radius)
            ax.plot(self.x0, self.y0, 'ko')
            ax.contour(self.inner_mask, colors='w', linestyles=':')
            cb = plt.colorbar(im, ax=ax, shrink=.75)
            
            ax = axes[0, 1]
            ax.set_ylabel(r'1 / radius$^2$ [pix$^{-2}$]')
            ax.plot(theta_r * 180/np.pi, inv_r2, 'k-', alpha=.2,
                    label=f'isophote level: {self.isophote_median:.4g} - {self.isophote_mean:.4g} => {np.count_nonzero(mask)} pix')
            ax.plot(theta_r * 180/np.pi, model, 'k--',
                    label=f'(a, b, $\\varphi_0$) = ({self.a:.2f} pix, {self.b:.2f} pix, {self.theta_0*180/np.pi:.2f} deg)')
            ax.axvline(self.theta_0 * 180/np.pi, c='k', ls=':')
            ax.legend()
            
            ax = axes[1, 0]
            im = ax.imshow(valid_data, origin='lower', interpolation='nearest', vmax=2*self.isophote_mean-self.isophote_median, cmap='terrain')
            self.plot(ax, self.a)
            #self.plot(ax, inner_radius)
            ax.plot(self.x0, self.y0, 'ko')
            ax.contour(self.inner_mask, colors='w', linestyles=':')
            cb = plt.colorbar(im, ax=ax, shrink=.75)
            
            ax = axes[1, 1]
            ax.set_ylabel('radius [pix]')
            ax.set_xlabel('azimuthal angle $\\varphi$ [deg]')
            ax.plot(theta_r * 180/np.pi, mean_r, 'k-', alpha=.2,
                    label=f'isophote level: {self.isophote_median:.4g} - {self.isophote_mean:.4g} => {np.count_nonzero(mask)} pix')
            ax.plot(theta_r * 180/np.pi, 1/np.sqrt(model), 'k--',
                    label=f'(a, b, $\\varphi_0$) = ({self.a:.2f} pix, {self.b:.2f} pix, {self.theta_0*180/np.pi:.2f} deg)')
            ax.axvline(self.theta_0 * 180/np.pi, c='k', ls=':')
            ax.legend()
            
            
            fig.savefig(os.path.join(plots_dir, f'{fig_name}.png'), facecolor='white')
            if not show_plots:
                plt.close()

                
    def _find_isophote(self, valid_data, inner_radius, pixel_percentile=25):
        # Maps of polar coordinates (r and theta)
        x = np.arange(valid_data.shape[1]) - self.x0
        y = np.arange(valid_data.shape[0]) - self.y0
        r = np.sqrt((x**2)[np.newaxis, :] + (y**2)[:, np.newaxis])

        mask = r < float(inner_radius)
        self.isophote_mean = np.nanmean(valid_data[mask])
        self.isophote_median = np.nanmedian(valid_data[mask])
        
        # Define isophote
        mask = r < 1.5*float(inner_radius)
        mask &= (valid_data < self.isophote_mean)
        mask &= (valid_data > self.isophote_median)
        mask = ndimage.median_filter(mask, 5)
        print(f'  Isophote level: {self.isophote_median:.4g} - {self.isophote_mean:.4g} => {np.count_nonzero(mask)} pix')
        
        # Recenter
        #x_cm = self.x0 + np.nanmedian((x[np.newaxis, :]*np.ones_like(mask))[mask > 0])
        #y_cm = self.y0 + np.nanmedian((y[:, np.newaxis]*np.ones_like(mask))[mask > 0])
        #x_cm = self.x0 + np.nanmean((x[np.newaxis, :]*np.ones_like(mask))[mask > 0])
        #y_cm = self.y0 + np.nanmean((y[:, np.newaxis]*np.ones_like(mask))[mask > 0])
        weight = self.inner_mask * (r < inner_radius/2) * valid_data
        x_cm = self.x0 + np.sum(x[np.newaxis, :]*weight) / np.sum(weight)
        y_cm = self.y0 + np.sum(y[:, np.newaxis]*weight) / np.sum(weight)

        return mask, float(x_cm), float(y_cm)


    def plot(self, ax, radius, style='k--'):
        theta = np.linspace(0, 2*np.pi, 361)
        for a in np.atleast_1d(radius):
            along_major_axis = a * np.cos(theta)
            along_minor_axis = (a*self.b/self.a) * np.sin(theta)
            ra = self.x0 + along_major_axis * np.cos(self.theta_0) + along_minor_axis * np.cos(self.theta_0 + np.pi/2)
            dec = self.y0 + along_major_axis * np.sin(self.theta_0) + along_minor_axis * np.sin(self.theta_0 + np.pi/2)
            ax.plot(ra, dec, style)
    
    
    def get_profile(self, data, fig_name=None):
        r_0_bins = np.arange(1 + np.sqrt(np.min(data.shape)/2))**2
        r_0_mid = (r_0_bins[:-1] + r_0_bins[1:]) / 2
        median_profile = np.empty(r_0_bins.size-1)
        upper_profile = np.empty_like(median_profile)
        lower_profile = np.empty_like(median_profile)
        for i in range(r_0_bins.size-1):
            r_inner = r_0_bins[i]
            r_outer = r_0_bins[i+1]
            try:
                lower_profile[i], median_profile[i], upper_profile[i] = np.nanpercentile(
                    data[(self.r_0 >= r_inner) & (self.r_0 <= r_outer)], [16, 50, 84])
            except:
                lower_profile[i], median_profile[i], upper_profile[i] = (np.nan, np.nan, np.nan)

        if fig_name is not None:
            fig, axes = new_figure(f'{fig_name}_profile', figsize=(16, 4), ncols=4, sharey=False, sharex=False,
                                   gridspec_kw={'width_ratios': [1, 1, 1, 2]})
            ax = axes[0, 0]
            ax.set_title('data')
            im = ax.imshow(data, origin='lower', interpolation='nearest', cmap='nipy_spectral', norm=colors.LogNorm())
            cb = plt.colorbar(im, ax=ax, shrink=.5)
            self.plot(ax, self.a)
            ax = axes[0, 1]
            ax.set_title('model')
            model = np.interp(self.r_0, r_0_mid, median_profile)
            im = ax.imshow(model, origin='lower', interpolation='nearest', cmap='nipy_spectral', norm=im.norm)
            cb = plt.colorbar(im, ax=ax, shrink=.5)
            self.plot(ax, self.a)
            ax = axes[0, 2]
            ax.set_title('residual')
            residual = data-model
            mad = np.nanmedian(np.fabs(residual))
            im = ax.imshow(residual, origin='lower', interpolation='nearest', cmap='Spectral', norm=colors.Normalize(-5*mad, 5*mad))
            cb = plt.colorbar(im, ax=ax, shrink=.5)
            self.plot(ax, self.a)
            
            ax = axes[0, 3]
            ax.set_title('radial profile')
            ax.plot(self.r_0.ravel(), data.ravel(), 'c.', alpha=.05)
            ax.plot(r_0_mid, median_profile, 'r-+')
            ax.fill_between(r_0_mid, lower_profile, upper_profile, color='k', alpha=.5)
            #ax.set_ylim(-.1, .1)
            ax.set_yscale('log')
            ax.set_ylabel('pixel value')
            ax.set_xlabel('isophotal radius $a$ [pix]')
            fig.savefig(os.path.join(plots_dir, f"{fig_name}_profile.png"), facecolor='white')     
            if not show_plots:
                plt.close()
        
        return r_0_bins, median_profile, lower_profile, upper_profile

#elllipse = Ellipse(surface_density_map, inner_radius=(3*theta_disk/pixscale).to(u.dimensionless_unscaled), fig_name='kk')


# ## Main loop

# In[18]:


for galaxy in SPARC_catalogue[:]:
    if np.ma.is_masked(galaxy['Galaxy']):
        continue

    # Cutouts
    header, data = fectch_images(galaxy, SDSS_skymaps, fig_name=f"{galaxy['Galaxy']}_cutouts")
    if header[next(iter(header))] is None:
        continue

    # Maps
    surface_density_map, surface_density_err = estimate_stellar_surface_density(data, fig_name=f"{galaxy['Galaxy']}_mass-to-light")
    hdr = header[next(iter(header))]
    hdr['BUNIT'] = 'M_sun / pc^2'
    hdr['COMMENT'] = 'M/L based on Garcia-Benito et al. (2019)'
    wcs = WCS(hdr)
    fits.PrimaryHDU(header=hdr, data=surface_density_map).writeto(
        os.path.join(maps_dir, f"{galaxy['Galaxy']}_surface_density.fits"), overwrite=True, output_verify='fix')

    # Profiles
    theta_disk = (galaxy['Rdisk'] / galaxy['D']).to_value(u.dimensionless_unscaled) << u.rad
    pixscale = wcs.proj_plane_pixel_scales()[0]
    ellipse = Ellipse(surface_density_map, inner_radius=(3*theta_disk/pixscale).to_value(u.dimensionless_unscaled),
                      fig_name=f"{galaxy['Galaxy']}_ellipse")
    coord_centre = wcs.pixel_to_world(ellipse.x0, ellipse.y0)
    ra_semi_major_axis = ellipse.x0 + ellipse.a * np.cos(ellipse.theta_0)
    dec_semi_major_axis = ellipse.y0 + ellipse.a * np.sin(ellipse.theta_0)
    coord_a = wcs.pixel_to_world(ra_semi_major_axis, dec_semi_major_axis)
    semi_major_axis = coord_centre.separation(coord_a)
    position_angle = coord_centre.position_angle(coord_a)
    ra_semi_minor_axis = ellipse.x0 + ellipse.b * np.cos(ellipse.theta_0 + np.pi/2)
    dec_semi_minor_axis = ellipse.y0 + ellipse.b * np.sin(ellipse.theta_0 + np.pi/2)
    coord_b = wcs.pixel_to_world(ra_semi_minor_axis, dec_semi_minor_axis)
    semi_minor_axis = coord_centre.separation(coord_b)
    inclination = np.arccos(semi_minor_axis/semi_major_axis)
    np.savetxt(os.path.join(params_dir, f'{galaxy["Galaxy"]}_params.csv'),
               [coord_centre.ra.deg,
                coord_centre.dec.deg,
                inclination.to_value(u.deg),
                position_angle.to_value(u.deg)],
               header='ra, dec, inclination, position_angle (deg)',
               fmt='%.4f')

    r_bins = {}
    median_profile = {}
    for band in data:
        r_bins[band], median_profile[band] = ellipse.get_profile(data[band], fig_name=f"{galaxy['Galaxy']}_{band}")[:2]
    r_mid = r_bins['g']
    r_mid = (r_mid[1:] + r_mid[:-1]) / 2
    
    ra_mid = ellipse.x0 + r_mid * np.cos(ellipse.theta_0)
    dec_mid = ellipse.y0 + r_mid * np.sin(ellipse.theta_0)
    coord_mid = wcs.pixel_to_world(ra_mid, dec_mid)
    theta_mid = coord_centre.separation(coord_mid).to(u.arcsec)

    profiles_table = QTable()
    profiles_table.add_column(theta_mid, name='theta')
    surface_density_profile, surface_density_profile_err = estimate_stellar_surface_density(median_profile, None, profiles_table)
    profiles_table.add_column(surface_density_profile << u.Msun/u.pc**2, name='median')
    profiles_table.add_column(surface_density_profile_err << u.dex, name='std')    
    profiles_table.write(os.path.join(profiles_dir, f"{galaxy['Galaxy']}_surface_density.csv"), overwrite=True)
    
    # Plot median surface density and uncertainty

    fig_name = f"{galaxy['Galaxy']}_surface_density"
    fig = plt.figure(fig_name, figsize=(20, 5))

    ax = fig.add_subplot(141, projection=wcs)
    ax.set_title('$\Sigma_\star$ [M$_\odot$/pc$^2$]')
    im = ax.imshow(surface_density_map, origin='lower', cmap='nipy_spectral', norm=colors.LogNorm(3, 3e3))
    ellipse.plot(ax, ellipse.a)
    cb = plt.colorbar(im, ax=ax, shrink=.5)
    
    ax = fig.add_subplot(142, projection=wcs)
    ax.set_title('uncertainty $\Delta\log_{10}\Sigma_\star$ [dex]')
    im = ax.imshow(surface_density_err, cmap='Spectral_r', vmin=.05, vmax=.35)
    ellipse.plot(ax, ellipse.a)
    cb = plt.colorbar(im, ax=ax, shrink=.5)
    
    ax = fig.add_subplot(143, position=[.55, .23, .4, .55])
    ax.set_ylabel(r'$\Sigma$ [M$_\odot$ / pc$^2$]')
    ax.set_xlabel(r'$R$ [arcsec]')
    ax.set_yscale('log')
    ra_pix = ellipse.x0 + ellipse.r_0.ravel() * np.cos(ellipse.theta_0)
    dec_pix = ellipse.y0 + ellipse.r_0.ravel() * np.sin(ellipse.theta_0)
    coord_pix = wcs.pixel_to_world(ra_pix, dec_pix)
    theta_pix = coord_centre.separation(coord_pix)
    ax.plot(theta_pix.to_value(u.arcsec), surface_density_map.ravel(), 'c.', alpha=.05)
    ax.plot(theta_mid.to_value(u.arcsec), surface_density_profile, 'r-+')
    factor = 10**surface_density_profile_err
    ax.fill_between(theta_mid.to_value(u.arcsec), surface_density_profile/factor, surface_density_profile*factor, color='k', alpha=.5)
    ax.axvline(theta_disk.to_value(u.arcsec), c='k', ls=':',
               label=f'$R_{{disk}}$ = {theta_disk.to(u.arcmin):.2f} ({galaxy["Rdisk"]:.2f} at D={galaxy["D"]:.2f})')
    ax.set_ylim(.3, 3e4)
    ax.set_xlim(-.1*theta_disk.to_value(u.arcsec), 4*theta_disk.to_value(u.arcsec))
    ax.legend(title=f'{galaxy["Galaxy"]} (i={np.arccos(semi_minor_axis/semi_major_axis).to(u.deg):.2f}, pa={position_angle.to(u.deg):.2f})')
    
    fig.savefig(os.path.join(plots_dir, f"{fig_name}.png"), facecolor='white')
    if not show_plots:
        plt.close('all')


# ## Mass model

# In[19]:


for galaxy in SPARC_catalogue[:1]:
    if np.ma.is_masked(galaxy['Galaxy']):
        continue

    params_filename = os.path.join(params_dir, f"{galaxy['Galaxy']}_params.csv")
    if not os.path.isfile(params_filename):
        continue
    ra, dec, inclination, pa = np.loadtxt(params_filename)
    inclination *= u.deg
    
    profiles_filename = os.path.join(profiles_dir, f"{galaxy['Galaxy']}_surface_density.csv")
    if not os.path.isfile(profiles_filename):
        continue
    
    profiles_table = QTable.read(profiles_filename)
    theta = profiles_table['theta'] << u.arcsec
    surface_density = profiles_table['median'] << u.Msun/u.pc**2
    surface_density *= np.cos(inclination)
 
    # Mean surface density
    theta_bins = np.geomspace(theta[0]/2, theta[-1]*2, theta.size*10)
    theta_mid = (theta_bins[1:] + theta_bins[:-1]) / 2
    bin_area = np.pi * (theta_bins[1:]**2 - theta_bins[:-1]**2)
    bin_area[0] = np.pi * theta_bins[1]**2
    mean_surface_density = np.cumsum(bin_area * np.interp(theta_mid, theta, surface_density, right=0))
    mean_surface_density /= np.pi * theta_bins[1:]**2
                                     
    bins = np.where(SPARC_models['ID'] == galaxy['Galaxy'])
    SPARC_D = SPARC_models[bins]['D'][0]
    SPARC_R = SPARC_models[bins]['R']
    SPARC_theta = ((SPARC_models[bins]['R'] / SPARC_D).to_value(u.dimensionless_unscaled) << u.radian).to(u.arcsec)
    SPARC_obs = (SPARC_models[bins]['Vobs']**2 / SPARC_R / np.pi/c.G).to(u.Msun/u.pc**2)
    SPARC_gas = (SPARC_models[bins]['Vgas']**2 / SPARC_R / np.pi/c.G).to(u.Msun/u.pc**2)
    #SPARC_bul = (SPARC_models[bins]['Vbul']**2 / SPARC_R / np.pi/c.G).to(u.Msun/u.pc**2)
    #SPARC_disk = (SPARC_models[bins]['Vdisk']**2 / SPARC_R / np.pi/c.G).to(u.Msun/u.pc**2)

    mass_to_light_disk, mass_to_light_bul, SPARC_fit_D, SPARC_fit_inclination = np.loadtxt(
        os.path.join(input_dir, 'model_fits', 'ByGalaxy', 'Table', f'{galaxy["Galaxy"]}.mrt'),
        usecols=(1, 3, 5, 7), unpack=True)
    '''
    print('disk', np.nanpercentile(mass_to_light_disk, [16, 50, 84]))
    print('bul', np.nanpercentile(mass_to_light_bul, [16, 50, 84]))
    print('D', np.nanpercentile(SPARC_fit_D, [16, 50, 84]))
    print('i', np.nanpercentile(SPARC_fit_inclination, [16, 50, 84]))
    '''
    median_bul = np.nanmedian(mass_to_light_bul)
    median_disk = np.nanmedian(mass_to_light_disk)
    median_D = np.nanmedian(SPARC_fit_D)
    median_i = np.nanmedian(SPARC_fit_inclination)
    distance_rescaling = np.sqrt(median_D / SPARC_D)
    SPARC_bul = SPARC_models[bins]['SBbul'] * median_bul*u.Msun/u.Lsun * distance_rescaling
    SPARC_disk = SPARC_models[bins]['SBdisk'] * median_disk*u.Msun/u.Lsun * distance_rescaling
    #SPARC_stars = (median_bul*SPARC_bul + median_disk*SPARC_disk) * distance_rescaling
    SPARC_stars = SPARC_bul + SPARC_disk
    
    fig_name = f'{galaxy["Galaxy"]}_mass_model'
    fig, axes = new_figure(fig_name)
    ax = axes[0, 0]

    ax.plot(theta, surface_density, 'k-+', label=f'This work (i={inclination:.2f})')
    ax.plot(SPARC_theta, SPARC_stars, 'k:.', label=f'SPARC (i={median_i:.2f}, $\\Upsilon_{{bulge}}\sim${median_bul:.2f}, $\\Upsilon_{{disk}}\sim${median_disk:.2f})')
    '''
    ax.plot(theta, surface_density, 'k-+', label=f'$\\Sigma_\\star$ (i={inclination:.2f})')
    #ax.plot(theta_mid, mean_surface_density, 'k-', label=f'$\\bar\\Sigma_\\star$')
    #ax.plot(SPARC_theta, SPARC_obs*distance_rescaling, 'bo', label=f'SPARC $V^2/\pi GR$ (D={median_D:.2f})')
    ax.plot(SPARC_theta, SPARC_stars, 'b--', label=f'SPARC $\\bar\\Sigma_\\star$ (i={median_i:.2f})')
    ax.plot(SPARC_theta, SPARC_bul, 'r:', alpha=.5, label=f'$\\Upsilon_{{bulge}}\sim${median_bul:.2f}')
    ax.plot(SPARC_theta, SPARC_disk, 'b:', alpha=.5, label=f'$\\Upsilon_{{disk}}\sim${median_disk:.2f}')
    #ax.plot(SPARC_theta, median_bul*SPARC_bul*distance_rescaling, 'r:', alpha=.5, label=f'$\\Upsilon_{{bulge}}\sim${median_bul:.2f}')
    #ax.plot(SPARC_theta, median_disk*SPARC_disk*distance_rescaling, 'b:', alpha=.5, label=f'$\\Upsilon_{{disk}}\sim${median_disk:.2f}')
    #ax.plot(SPARC_theta, SPARC_gas*distance_rescaling, 'c--', alpha=.5, label='gas')
    ax.plot(SPARC_theta, SPARC_stars/distance_rescaling, 'b--', alpha=.2, label=f'SPARC fiducial D={SPARC_D:.2f}')
    '''

    ax.legend(title=f'{galaxy["Galaxy"]} (D={median_D:.2f} Mpc)')
    ax.set_yscale('log')
    ax.set_ylabel(r'surface density $\Sigma$ [M$_\odot$/pc$^2$]')
    ax.set_xlabel(r'galactocentric distance $\theta$ [arcsec]')
    
    fig.savefig(os.path.join(plots_dir, f"{fig_name}.png"), facecolor='white')
    if not show_plots:
        plt.close('all')


# In[20]:


bins = np.where(SPARC_models['ID'] == galaxy['Galaxy'])
SPARC_models[bins]


# In[ ]:





# In[ ]:




