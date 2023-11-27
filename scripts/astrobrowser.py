import numpy as np
from matplotlib import pyplot as plt
from matplotlib import colors

from scipy import ndimage
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table
from astropy.utils.data import conf
from astropy import units as u
from photutils.aperture import SkyEllipticalAperture

from time import time
import requests
import ipywidgets as widgets
from IPython.display import display

#%% -----------------------------------------------------------------------------


def get_hips_proprties(hips_service_url):
    '''Get HiPS properties ;^D'''
    
    with conf.set_temp('remote_timeout', 10):
        try:
            data = requests.get(f"{hips_service_url}/properties")
            properties = {}
            for line in data.content.split(b'\n'):
                if b'= ' in line:
                    key, value = line.split(b'= ', maxsplit=1)
                    properties[key.decode("ISO-8859-1").split(' ')[0]] = value.decode('ISO-8859-1')  # remove spaces form key
            return properties
        except:
            print(f'ERROR: could not get HiPS properties for {hips_service_url}')
            return None  # get_hips_proprties(hips_service_url)  # or try again... Dangerous!

#%% ----------------------------------------------------------------------------


def find_bg(data):
    p16, p50 = np.nanpercentile(data, [16, 50])
    mu0 = p50
    sigma0 = p50 - p16
    weight = np.exp(-.5 * ((data - mu0) / sigma0)**2)
    total_weight = np.nansum(weight)
    mu1 = np.nansum(weight * data) / total_weight
    sigma1 = np.nansum(weight * data**2) / total_weight
    sigma1 = np.sqrt(sigma1 - mu1**2)
    print(mu0, sigma0)
    print(mu1, sigma1)
    #ivar = 1/sigma1**2 - 1/sigma0**2
    #mu = (mu1/sigma1**2 - mu0/sigma0**2) / ivar
    #print(mu, np.sqrt(1/ivar))
    #return mu, np.sqrt(1/ivar)
    return mu1, sigma1


#%% ----------------------------------------------------------------------------


def aperture_photometry(hips_service_url, position, a, b, PA, skymap_units=None, desired_n_beams=100, fig=None):
    """Download a HiPS cutout to compute flux and error for a specified elliptical aperture"""

    hips_properties = get_hips_proprties(hips_service_url)
    if hips_properties is None:
        return np.nan*u.mJy, np.nan*u.mJy

    if 's_pixel_scale' in hips_properties:
        original_pixel = float(hips_properties['s_pixel_scale']) * u.deg
    elif 'hips_pixel_scale' in hips_properties:
        original_pixel = float(hips_properties['hips_pixel_scale']) * u.deg
        print(f'WARNING: original pixel size not available! using HiPS size = {original_pixel.to_value(u.arcsec)} arcsec')
    else:
        print('ERROR: neither original nor HiPS pixel sizes available!')
        return np.nan*u.mJy, np.nan*u.mJy

    beam = original_pixel**2
    if skymap_units is None:
        skymap_units = u.Jy / beam
    total_area = np.pi * a * b
    n_beams = float(total_area / beam)
    if n_beams > desired_n_beams:
        cutout_pixel = np.sqrt(total_area / desired_n_beams) / np.cos(position.dec)  # Mercator projection: pixel_area = cutout_pixel**2 * cos(DEC)
        print('  cutout_pixel =', cutout_pixel)
    else:
        print(f'WARNING: (a, b)=({a.to_value(u.arcsec):.3g}, {b.to_value(u.arcsec):3g}) arcsec',
              f' => {n_beams} beams (original pixel={original_pixel.to_value(u.arcsec):.3g} arcsec) < {desired_n_beams} beams')
        cutout_pixel = float(hips_properties['hips_pixel_scale']) * u.deg
        print(f'          using HiPS pixel scale = {cutout_pixel.to_value(u.arcsec):.3g} arcsec')

    unit_conversion_mJy = (skymap_units * total_area).to_value(u.mJy)

    print(f"> Downloading... (please be patient)")
    header, data = get_cutout(hips_service_url, position.ra.deg, position.dec.deg, 4*a.to_value(u.arcsec), cutout_pixel.to_value(u.arcsec))
    if header is None:
        return np.nan*u.mJy, np.nan*u.mJy


    # Aperture:
    c = np.cos(PA)
    s = np.sin(PA)
    a_deg = a.to_value(u.deg)
    a_y = a_deg * c / header['CDELT2']
    a_x = a_deg * s / header['CDELT1'] * np.cos(position.dec)  # due to Mercator projection
    a_pix2 = a_x**2 + a_y**2
    b_deg = b.to_value(u.deg)
    b_y = b_deg * s / header['CDELT2']
    b_x = - b_deg * c / header['CDELT1'] * np.cos(position.dec)  # due to Mercator projection
    b_pix2 = b_x**2 + b_y**2
    x = np.arange(data.shape[1]) - header['CRPIX1']
    y = np.arange(data.shape[0]) - header['CRPIX2']
    r2 = ((x[np.newaxis, :]*a_x + y[:, np.newaxis]*a_y) / a_pix2)**2
    r2 += ((x[np.newaxis, :]*b_x + y[:, np.newaxis]*b_y) / b_pix2)**2

    pixel_area = cutout_pixel**2  * np.cos(position.dec)  # due to Mercator projection
    n_aperture = int(total_area/pixel_area)
    src_threshold = np.sort(r2.flat)[n_aperture]
    bg_threshold = np.sort(r2.flat)[3*n_aperture]
    aperture = (r2 <= bg_threshold)
    bg_weight = np.where(np.isfinite(data) & ~aperture, 1., 0.)
    p16, p50, p84 = np.nanpercentile(data[~aperture], [16, 50, 84])
    bg_weight[np.abs((data - p50) / (p50-p16)) > 3] = 0
    bg_image = np.where(np.isfinite(data), bg_weight*data, 0.)
    
    smoothing_radius = float(.5*b/cutout_pixel)
    bg_weight = ndimage.gaussian_filter(bg_weight, smoothing_radius)
    bg_image = ndimage.gaussian_filter(bg_image, smoothing_radius) / bg_weight

    aperture = (r2 < src_threshold)
    n_aperture = np.count_nonzero(aperture)
    original_mean = np.nanmean(data[aperture])
    original_std = np.nanstd(data[aperture])
    subtracted_mean = np.nanmean((data-bg_image)[aperture])
    subtracted_std = np.sqrt(np.nanmean((data-bg_image)[aperture & (data > bg_image + subtracted_mean)]**2))
    mean_err = np.nanstd(data[aperture]) / np.sqrt(n_beams)
    bg = np.nanmean(bg_image[aperture])
    bg_var = max(
        np.nanvar(bg_image[aperture]),
        np.nanmean((bg_image - data)[aperture & (bg_image > data)]**2)
    )
    bg_err = np.sqrt(bg_var / np.sum(bg_weight[aperture]) * n_beams/n_aperture)
    subtracted_err = np.sqrt(bg_err**2 + mean_err**2)
    #flux = np.nansum((data - bg_image)[aperture]) * unit_conversion_mJy
    #flux_err = n_aperture * subtracted_err * unit_conversion_mJy
    flux = subtracted_mean * unit_conversion_mJy
    flux_err = subtracted_err * unit_conversion_mJy
    
    bg_std = np.sqrt(bg_var)
    if fig is not None:
        default_cmap = plt.get_cmap('terrain').copy()
        default_cmap.set_bad('gray')
        axes = fig.subplots(nrows=1, ncols=3, squeeze=False)

        ax = axes[0, 0]
        ax.set_title(f'original: {original_mean:.4g} $\pm$ {mean_err:.3g} ({original_std:.3g})')
        im = ax.imshow(data, interpolation='nearest', origin='lower', vmin=bg-3*bg_std, vmax=bg+6*bg_std, cmap=default_cmap)
        #im = ax.imshow(r2, interpolation='nearest', origin='lower', cmap=default_cmap)
        ax.contour(r2, levels=[src_threshold, bg_threshold], colors=['k', 'k'], linestyles=['-', ':'])
        cb = plt.colorbar(im, ax=ax, shrink=.7)
        cb.ax.tick_params(labelsize='small')
        cb.ax.axhline(original_mean+original_std, c='k', ls=':')
        cb.ax.axhline(original_mean, c='k', ls='-')
        cb.ax.axhline(original_mean-original_std, c='k', ls=':')
        
        ax = axes[0, 1]
        #ax.set_title(f'background: [{p16:.4g}, {p50:.4g}, {p84:.4g}]')
        ax.set_title(f'background: {bg:.3g} $\pm$ {bg_err:.3g} ({bg_std:.3g})')
        im = ax.imshow(bg_image, interpolation='nearest', origin='lower', vmin=bg-3*bg_std, vmax=bg+6*bg_std, cmap=default_cmap)
        ax.contour(r2, levels=[src_threshold, bg_threshold], colors=['k', 'k'], linestyles=['-', ':'])
        cb = plt.colorbar(im, ax=ax, shrink=.7)
        cb.ax.tick_params(labelsize='small')
        cb.ax.axhline(bg + bg_std, c='w', ls=':')
        cb.ax.axhline(bg, c='w', ls='-')
        cb.ax.axhline(bg - bg_std, c='w', ls=':')

        ax = axes[0, 2]
        ax.set_title(f'subtracted {subtracted_mean:.4g} $\pm$ {subtracted_err:.3g} ({subtracted_std:.3g})')
        im = ax.imshow(data - bg_image, interpolation='nearest', origin='lower', cmap=default_cmap, vmin=-3*bg_std, vmax=6*bg_std)#, vmin=bg-bg_err, vmax=mean+bg_err)
        ax.contour(r2, levels=[src_threshold, bg_threshold], colors=['k', 'k'], linestyles=['-', ':'])
        cb = plt.colorbar(im, ax=ax, shrink=.7)
        cb.ax.tick_params(labelsize='small')
        cb.ax.axhline(subtracted_std, c='k', ls=':')
        cb.ax.axhline(subtracted_mean, c='k')
        cb.ax.axhline(bg_std, c='w', ls=':')
        
    #return corrected_flux, flux_err
    return flux*u.mJy, flux_err*u.mJy

#%% ----------------------------------------------------------------------------


def get_available_images(ra_deg, dec_deg, radius_deg, max_pixel_deg=np.inf):
    """Query the CDS MOC server to find out what images with the requested resolution are available within the field of view"""
    
    #data = requests.get(f"http://alasky.cds.unistra.fr/MocServer/query?RA={radeg}&DEC={decdeg}&SR={radiusdeg}")
    data = requests.get(f"http://alasky.cds.unistra.fr/MocServer/query?RA={ra_deg}&DEC={dec_deg}&SR={radius_deg}"
                        +"&expr=(hips_frame%3Dequatorial%2Cgalactic%2Cecliptic+||+hips_frame%3D!*)"
                        +"+%26%26+dataproduct_type%3Dimage+%26%26+hips_service_url%3D*&get=record")
    # TODO: process the request, handle errors, and such...

    # or just assume everything's fine ;^D
    all_maps = []
    for line in data.content.split(b'\n'):
        if b'= ' in line:
            key, value = line.split(b'= ', maxsplit=1)
            value = value.decode('ISO-8859-1')
            if key[:2] == b'ID':
                all_maps.append({'ID': value})
                #print('Added map:', value)
            else:
                all_maps[-1][key.decode("ISO-8859-1").split(' ')[0]] = value

    images = {}
    for i, skymap in enumerate(all_maps):
        if 'fits' in skymap['hips_tile_format']:
            if 'client_category' in skymap.keys() and 'em_min' in skymap.keys():
                if 'Image' in skymap['client_category']:
                    if float(skymap['hips_pixel_scale']) <= max_pixel_deg:
                        images[skymap['ID']] = skymap
                        #print(f"{skymap['client_category']}: lambda = [{skymap['em_min']} - {skymap['em_max']}] ({skymap['ID']}): {skymap['obs_title']}")
    return images

#%% ----------------------------------------------------------------------------


def get_cutout(hips_service_url, ra_deg, dec_deg, radius_arcsec, pixel_arcsec):
    """Retrieve a cutout from a public HiPS map"""

    print(f"http://localhost:4000/api/cutout?"
                    +f"radiusasec={radius_arcsec}&pxsizeasec={pixel_arcsec}"
                    +f"&radeg={ra_deg}&decdeg={dec_deg}"
                    +f"&hipsbaseuri={hips_service_url}")
    with conf.set_temp('remote_timeout', 10):
        try:
            hdu = fits.open(f"http://localhost:4000/api/cutout?"
                        +f"radiusasec={radius_arcsec}&pxsizeasec={pixel_arcsec}"
                        +f"&radeg={ra_deg}&decdeg={dec_deg}"
                        +f"&hipsbaseuri={hips_service_url}",
                        ignore_missing_simple=True, mode='readonly')
        except:
            print('ERROR: could not download cutout (most likely, timeout) :^(')
            hdu = None
    if hdu is None:
        return None, None
    else:
        #hdu[0].verify('fix')
        #print('\n---\n', hdu[0].header)
        return hdu[0].header, hdu[0].data

#%% ----------------------------------------------------------------------------


def light_growth_curve(img, max_iter=15, radius=5):
    """Find cumulative growth curve in logarithmic bins (2**n pixels)
    that adapt to the intensity contours and mask adjacent objects"""

    # Start with circular profiles around the image centre
    x0 = .5*img.shape[1]
    y0 = .5*img.shape[0]
    x = np.arange(img.shape[1]) - x0
    y = np.arange(img.shape[0]) - y0
    filtered_img = 1 / (x[np.newaxis, :]**2 + y[:, np.newaxis]**2)
    
    # Iteratively refine curve
    n_iter = 0
    while n_iter < max_iter:
        n_iter += 1
        
        sort_by_area = np.argsort(filtered_img.flat)[::-1]
        area = np.ones_like(img) * img.size

        area.flat[sort_by_area] = np.arange(img.size)
        #index = (np.log(area)/np.log(2)).astype(int) # logarithmic bins (2**n pixels)
        #index = (np.sqrt(area/np.pi)/radius).astype(int)
        index = (np.power(4*area/np.pi, .25).clip(0)).astype(int)
        n_bins = np.max(index) + 1

        # Running median and deviation (from 16/84 percentiles) within equal-area bins
        running_percentiles = np.zeros((n_bins, 3))
        for i in range(n_bins):
            running_percentiles[i] = np.nanpercentile(img[index == i], [16, 50, 84])
        deviation = np.fmin(running_percentiles[:, 2] - running_percentiles[:, 1],  # p84 - p50
                            running_percentiles[:, 1] - running_percentiles[:, 0])  # p50 - p16
        deviation = deviation.clip(np.min(deviation[deviation > 0]))  # no zeros

        # Reconstructed flux and variance within each bin, considering median and deviation
        intensity = np.zeros_like(deviation)
        variance = np.zeros_like(deviation)
        i = 0
        for mu0, var0 in zip(running_percentiles[:, 1], deviation**2):
            data = img[index == i]
            weight = np.exp(-.5 * (data - mu0)**2 / var0)
            total_weight = np.nansum(weight)
            mu1 = np.nansum(weight * data) / total_weight
            var1 = np.nansum(weight * (data - mu1)**2) / total_weight
            if var1 > 0:
                var = 1 / (1/var1 - 1/var0)
                variance[i] = 1 / (1/var1 - 1/var0)
                intensity[i] = variance[i] * (mu1/var1 - mu0/var0)
            i += 1

        # Assign weights to mask nearby sources
        median_img = running_percentiles[:, 1][index]
        weight = (img - median_img) / deviation[index]
        weight = np.exp(-weight**2)

        filtered_img = weight*img + (1-weight)*median_img
        print(np.nansum(filtered_img), img.size, np.nansum(weight), np.sum(intensity[index])/img.size, np.sqrt(np.sum(variance[index]))/img.size)
        filtered_img = np.where(np.isfinite(filtered_img), filtered_img, 0)
        norm = np.sum(filtered_img)
        filtered_img = np.fft.rfftn(filtered_img)
        filtered_img = np.fft.irfftn(filtered_img*np.absolute(filtered_img/norm))
        # make sure the filtered image has the same shape as the original
        filtered_img = np.pad(filtered_img,
                              ((0, img.shape[0]-filtered_img.shape[0]),
                               (0, img.shape[1]-filtered_img.shape[1])), 'edge')
        filtered_img = ndimage.gaussian_filter(filtered_img, radius)
        
    return intensity, variance, index, area, weight, filtered_img

'''
fig_name = 'light_curve'
plt.close(fig_name)
fig = plt.figure(fig_name, figsize=(12, 5))
axes = fig.subplots(nrows=1, ncols=1, squeeze=False)
fig.set_tight_layout(True)
for ax in axes.flat:
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', bottom=True, top=True, left=True, right=True)
    ax.tick_params(which='major', direction='inout', length=8, grid_alpha=.3)
    ax.tick_params(which='minor', direction='in', length=2, grid_alpha=.1)
    ax.grid(True, which='both')

fig.suptitle(fig_name)

ax = axes[0, 0]
ax.scatter(area.flat, img.flat, s=1, c='r', alpha=.1)
ax.scatter(area.flat, intensity[index.flat], s=1, c='k', alpha=.1)

ax.set_ylabel('intensity')
ax.set_xlabel('area [pix]')
'''

#%% ----------------------------------------------------------------------------


class DataExplorer(object):
    
    def __init__(self, catalogue, requested_map, galaxy_index=0):
        """Interactive display"""
        
        self.catalogue = catalogue
        self.requested_map = requested_map
        fig_name = 'AstroBrowser Data Explorer -- Notebook edition'
        plt.close(fig_name)
        self.fig = plt.figure(fig_name, figsize=(12, 6))
        axes = self.fig.subplots(nrows=1, ncols=2, squeeze=False, gridspec_kw={'width_ratios': [1, .05], 'hspace': 0, 'wspace': 0})
        self.ax1 = axes[0, 0]
        self.ax1_cb = axes[0, 1]
        self.fig.set_tight_layout(True)
        
        self.header = None
        self.data = None
        self.galaxy_index = None

        self.widget = widgets.interactive(
            self.update,
            galaxy_index=widgets.BoundedIntText(value=galaxy_index, min=0, max=len(catalogue), continuous_update=False),
            requested_map=widgets.Combobox(
                value=requested_map,
                placeholder='Choose Sky Map',
                #options=['Paul', 'John', 'George', 'Ringo'],
                #description='Combobox:',
                ensure_option=True,
                #disabled=False,
                #continuous_update=False
            ),
        )
        display(self.widget)


    def update(self, galaxy_index, requested_map):
        t0 = time()
        self.galaxy_index = galaxy_index
        galaxy = self.catalogue[galaxy_index]
        self.fig.suptitle(galaxy['ID'])
        print(f"Downloading {requested_map} for {galaxy['ID']}; please be patient ...")
        
        skymaps = get_available_images(galaxy['RA'], galaxy['DEC'], galaxy['RADIUS_ARCSEC']/3600, galaxy['RADIUS_ARCSEC']/3600)
        self.widget.children[1].options = [x for x in skymaps]
        
        self.header, self.data = get_cutout(skymaps[requested_map]['hips_service_url'], galaxy['RA'], galaxy['DEC'], galaxy['RADIUS_ARCSEC'], galaxy['PIXEL_SIZE_ARCSEC'])
        self.wcs = WCS(self.header)
        img = self.data
        linthresh = np.abs(np.nanmedian(img))
        if linthresh > 0:
            norm = colors.SymLogNorm(linthresh=np.abs(np.nanmedian(img)), vmin=np.nanmin(img), vmax=np.nanmax(img))
        else:
            norm = colors.Normalize()

        #intensity, variance, index, area, weight, filtered_img = light_growth_curve(img)
        
        ax = self.ax1
        #ax.clear()
        axis_pars = ax.get_subplotspec()
        ax.remove()
        ax = self.fig.add_subplot(axis_pars, projection=self.wcs)
        self.ax1 = ax
        im = ax.imshow(img, interpolation='nearest', origin='lower', cmap='ocean', norm=norm)
        cb = plt.colorbar(im, cax=self.ax1_cb, shrink=.7, orientation='vertical')
        cb.ax.tick_params(labelsize='small')
        ax.scatter(img.shape[1]/2, img.shape[0]/2, marker='+', c='k')
        ax.coords.grid(color='white', alpha=0.5, linestyle='solid')
        
        self.fig.set_tight_layout(True)
        print(f'Done! ({time()-t0:.3g} s)')

        '''
        ax = self.ax2
        ax.clear()
        #im = ax.imshow(index, interpolation='nearest', origin='lower', cmap='gnuplot2')
        im = ax.imshow(intensity[index], interpolation='nearest', origin='lower', cmap='gnuplot2', norm=norm)
        #im = ax.imshow((img-intensity[index])/(np.abs(img)+np.abs(intensity[index])), interpolation='nearest', origin='lower', cmap='RdBu_r')
        #im = ax.imshow(intensity[index], interpolation='nearest', origin='lower', cmap='gnuplot2', norm=norm)
        #im = ax.imshow(filtered_img, interpolation='nearest', origin='lower', cmap='gnuplot2', norm=norm)
        #im = ax.imshow(weight, interpolation='nearest', origin='lower', cmap='gnuplot2')
        #im = ax.imshow(area, interpolation='nearest', origin='lower', cmap='gnuplot2', norm=colors.LogNorm())
        cb = plt.colorbar(im, cax=self.ax2_cb, shrink=.7, orientation='vertical')
        cb.ax.tick_params(labelsize='small')
        ax.scatter(img.shape[1]/2, img.shape[0]/2, marker='+', c='k')
        ax.contour(weight, levels=[.01], colors=['k'])
        ax.contour(index, levels=np.arange(np.max(index)), colors=['w'], alpha=.5)
        '''

#%% ----------------------------------------------------------------------------
#                                                        ... Paranoy@ Rulz! ;^D