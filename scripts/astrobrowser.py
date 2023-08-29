import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.utils import data
import requests
from scipy import ndimage
from matplotlib import pyplot as plt
from matplotlib import colors
import ipywidgets as widgets
from IPython.display import display

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


def get_cutout(skymap, ra_deg, dec_deg, radius_arcsec, pixel_arcsec):
    """Retrieve a cutout from a public HiPS map"""

    with data.conf.set_temp('remote_timeout', 30):
        hdu = fits.open(f"http://localhost:4000/api/cutout?"
                    +f"radiusasec={radius_arcsec}&pxsizeasec={pixel_arcsec}"
                    +f"&radeg={ra_deg}&decdeg={dec_deg}"
                    +f"&hipsbaseuri={skymap['hips_service_url']}",
                    ignore_missing_simple=True)
    return hdu[0].data

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
        fig_name = 'AstroBrowser Data Explorer'
        plt.close(fig_name)
        self.fig = plt.figure(fig_name, figsize=(12, 6))
        axes = self.fig.subplots(nrows=1, ncols=2, squeeze=False, gridspec_kw={'width_ratios': [1, .05], 'hspace': 0, 'wspace': 0})
        self.ax1 = axes[0, 0]
        self.ax1_cb = axes[0, 1]
        self.fig.set_tight_layout(True)

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
        galaxy = self.catalogue[galaxy_index]
        self.fig.suptitle(galaxy['ID'])
        
        skymaps = get_available_images(galaxy['RA'], galaxy['DEC'], galaxy['RADIUS_ARCSEC']/3600, galaxy['RADIUS_ARCSEC']/3600)
        self.widget.children[1].options = [x for x in skymaps]
        print(requested_map)
        
        img = get_cutout(skymaps[requested_map], galaxy['RA'], galaxy['DEC'], galaxy['RADIUS_ARCSEC'], galaxy['PIXEL_SIZE_ARCSEC'])
        norm = colors.SymLogNorm(linthresh=np.abs(np.nanmedian(img)), vmin=np.nanmin(img), vmax=np.nanmax(img))

        #intensity, variance, index, area, weight, filtered_img = light_growth_curve(img)
        
        ax = self.ax1
        ax.clear()
        im = ax.imshow(img, interpolation='nearest', origin='lower', cmap='ocean', norm=norm)
        cb = plt.colorbar(im, cax=self.ax1_cb, shrink=.7, orientation='vertical')
        cb.ax.tick_params(labelsize='small')
        ax.scatter(img.shape[1]/2, img.shape[0]/2, marker='+', c='k')

        self.fig.set_tight_layout(True)

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