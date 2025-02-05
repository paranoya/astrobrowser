#!/usr/bin/env python
# coding: utf-8

# # Astrobrowser - Herschel Reference Survey
# 
# Explore the HiPS maps available for the galaxies in the HRS and compute aperture photometry.

# # 1. Initialisation

# ## System setup

# External libraries

# In[1]:


#%matplotlib ipympl
import os
import numpy as np
from scipy import special
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.ticker import AutoMinorLocator
from matplotlib.backends.backend_pdf import PdfPages
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.table import Table, QTable, hstack
from astropy import units as u
from astropy import constants as c
from scripts import astrobrowser
from photutils.aperture import SkyEllipticalAperture
import importlib


# Utility functions

# In[3]:


def new_figure(fig_name, figsize=(10, 5), nrows=1, ncols=1, sharex='col', sharey='row', gridspec_kw={'hspace': 0, 'wspace': 0}, suptitle=True):
    plt.close(fig_name)
    fig = plt.figure(fig_name, figsize=figsize)
    axes = fig.subplots(nrows=nrows, ncols=ncols, squeeze=False,
                        sharex=sharex, sharey=sharey,
                        gridspec_kw=gridspec_kw
                       )
    fig.set_tight_layout(True)
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


# In[4]:


def test_dir(dir_name):
    if not os.path.isdir(dir_name):
        print(f'>> WARNING: Creating directory "{dir_name}"')
        os.makedirs(dir_name)
    return(dir_name)


# Directories

# In[5]:


input_dir = 'HRS'
output_dir = test_dir(os.path.join(input_dir, 'output'))
maps_dir = test_dir(os.path.join(output_dir, 'maps'))


# ## Read HRS catalogues

# In[6]:


HRS_optical_UV_catalogue = Table.read(os.path.join(input_dir, 'Optical_UV_cortese_2012', 'table1.dat'), format='ascii.commented_header')
HRS_optical_UV_photometry = Table.read(os.path.join(input_dir, 'Optical_UV_cortese_2012', 'table2.dat'), format='ascii.commented_header')


# In[7]:


HRS_optical_UV_catalogue


# In[8]:


HRS_optical_UV_photometry


# In[9]:


HRS_PACS_catalogue = QTable.read(os.path.join(input_dir, 'PACS_cortese_2014_table2.vot'))
HRS_PACS_catalogue


# In[10]:


HRS_SPIRE_catalogue = QTable.read(os.path.join(input_dir, 'SPIRE_ciesla_2012', 'HRS_PHOTOMETRY_v2.1'), format='ascii.commented_header')
HRS_SPIRE_catalogue['Ra'].unit = u.deg
HRS_SPIRE_catalogue['Dec'].unit = u.deg
HRS_SPIRE_catalogue['a'].unit = u.arcsec
HRS_SPIRE_catalogue['b'].unit = u.arcsec
HRS_SPIRE_catalogue['pa'].unit = u.deg
HRS_SPIRE_catalogue['S250'].unit = u.mJy
HRS_SPIRE_catalogue['S350'].unit = u.mJy
HRS_SPIRE_catalogue['S500'].unit = u.mJy
HRS_SPIRE_catalogue['err_tot250'].unit = u.mJy
HRS_SPIRE_catalogue['err_tot350'].unit = u.mJy
HRS_SPIRE_catalogue['err_tot500'].unit = u.mJy
HRS_SPIRE_catalogue


# # 2. Aperture photometry

# In[26]:


class HiPS_skymap(object):
    
    def __init__(self, hips_service_url, units, beam=None):
        '''Intensity map in Hierarchical Progressive Survey (HiPS) format'''
        
        self.url = hips_service_url
        self.properties = astrobrowser.get_hips_proprties(hips_service_url)
        if self.properties is None:
            print('ERROR: HiPS properties not available!')
            raise -1
        if 'hips_pixel_scale' in self.properties:
            self.hips_pixel = float(self.properties['hips_pixel_scale']) * u.deg
        else:
            print('ERROR: HiPS pixel size not available!')
            raise -1

        if beam is None:
            if 's_pixel_scale' in self.properties:
                original_pixel = float(self.properties['s_pixel_scale']) * u.deg
                self.beam = original_pixel**2
            else:
                self.beam = self.hips_pixel**2
                print(f'WARNING: original pixel size not available! using HiPS size = {self.hips_pixel.to_value(u.arcsec)} arcsec')
        else:
            self.beam = beam
            original_pixel_beam = np.sqrt(beam)
            if 's_pixel_scale' in self.properties:
                original_pixel_properties = float(self.properties['s_pixel_scale']) * u.deg
                if not u.isclose(original_pixel_beam, original_pixel_properties):
                    print(f'WARNING: {original_pixel_beam} is different from {original_pixel_properties}')

        self.intensity_units = units
        if u.get_physical_type(units) == 'spectral flux density':
            self.intensity_units = units / self.beam

            
    def add_band(self, band, catalogue, output_dir='.', overwrite_table=True, overwrite_files=False):
        '''Create a Table with fluxes and errors for the requested band.'''
    
        if f'{band}_flux' in catalogue.colnames:
            if overwrite_table:
                print(f'WARNING: overwriting {band}')
            else:
                print(f'ERROR: cannot overwrite {band}!')
                raise -1  # TODO: raise proper exception
        results = QTable(names=[f'{band}_flux', f'{band}_flux_error'], units=[u.mJy, u.mJy], masked=True)
    
        with PdfPages(os.path.join(output_dir, f'{band}_maps.pdf')) as pdf:
            for galaxy in catalogue:
                position = SkyCoord(galaxy['ra'], galaxy['dec'])
                cutout_file = os.path.join(maps_dir, f"{galaxy['ID']}_{band}.fits")
                fig = plt.figure(figsize=(12, 4))
                flux, flux_err = astrobrowser.aperture_photometry(self, position, galaxy['a'], galaxy['b'], galaxy['pa'],
                                                                  cutout_file=cutout_file, overwrite=overwrite_files, fig=fig)
                results.add_row([flux, flux_err])
                title = f"{galaxy['ID']} {band} flux: ${flux.to_value(u.mJy):.3g} \pm {flux_err.to_value(u.mJy):.3g}$"
                if f'{band}_true_flux' in galaxy.colnames:
                    true_flux = galaxy[f'{band}_true_flux'].to_value(u.mJy)
                    true_err = galaxy[f'{band}_true_err'].to_value(u.mJy)
                    title += f" (${true_flux:.3g} \pm {true_err:.3g}$) mJy"
                print('  galaxy', title)
                fig.suptitle(title)
                plt.tight_layout()
                pdf.savefig()
                plt.close()

        catalogue[f'{band}_flux'] = results[f'{band}_flux']
        catalogue[f'{band}_flux_error'] = results[f'{band}_flux_error']
        catalogue.write(os.path.join(output_dir, f'{band}_photometry.fits'), overwrite=True)

        return catalogue


# ## GALEX

# In[27]:


GALEX_catalogue = QTable()

GALEX_catalogue.add_column([f'HRS-{int(i)}' for i in HRS_optical_UV_catalogue['HRS']], name='ID')

coords = SkyCoord(HRS_optical_UV_catalogue['R.A.'], HRS_optical_UV_catalogue['Dec.'], unit=[u.hourangle, u.deg])
GALEX_catalogue.add_column(coords.ra, name='ra')
GALEX_catalogue.add_column(coords.dec, name='dec')

GALEX_catalogue.add_column(HRS_optical_UV_catalogue['D_25']/2 * u.arcmin, name='a')
GALEX_catalogue.add_column(GALEX_catalogue['a'] * (1 - HRS_optical_UV_photometry['e']), name='b')
GALEX_catalogue.add_column(HRS_optical_UV_photometry['PA'] * u.deg, name='pa')

#FUV

GALEX_catalogue.add_column(3631*u.Jy *
                           10**(-.4 * np.array(np.where(HRS_optical_UV_photometry['FUV_D25'] == '...', 'nan', HRS_optical_UV_photometry['FUV_D25'])).astype(float)),
                           name='FUV_true_flux')
GALEX_catalogue.add_column(GALEX_catalogue['FUV_true_flux'] * (
    1 - 10**(-.4 * np.array(np.where(HRS_optical_UV_photometry['eFUV_D25'] == '...', 'nan', HRS_optical_UV_photometry['eFUV_D25'])).astype(float))), name='FUV_true_err')

#NUV

GALEX_catalogue.add_column(3631*u.Jy *
                           10**(-.4 * np.array(np.where(HRS_optical_UV_photometry['NUV_D25'] == '...', 'nan', HRS_optical_UV_photometry['NUV_D25'])).astype(float)),
                           name='NUV_true_flux')
GALEX_catalogue.add_column(GALEX_catalogue['NUV_true_flux'] * (
    1 - 10**(-.4 * np.array(np.where(HRS_optical_UV_photometry['eNUV_D25'] == '...', 'nan', HRS_optical_UV_photometry['eNUV_D25'])).astype(float))), name='NUV_true_err')

GALEX_catalogue


# In[30]:

units_I_nu = 3631*u.Jy * np.power(10, -0.4*18.82) / (1.5 * u.arcsec)**2
beam = np.pi*(3 * u.arcsec)**2
GALEX_FUV = HiPS_skymap('https://alasky.cds.unistra.fr/GALEX/GALEXGR6_7_FUV', units_I_nu, beam)

units_I_nu = 3631*u.Jy * np.power(10, -0.4*20.08) / (1.5 * u.arcsec)**2
beam = np.pi*(4 * u.arcsec)**2
GALEX_NUV = HiPS_skymap('https://alasky.cds.unistra.fr/GALEX/GALEXGR6_7_NUV', units_I_nu, beam)


# In[ ]:

GALEX_output = GALEX_catalogue
GALEX_output = GALEX_FUV.add_band('FUV', GALEX_output, output_dir)
GALEX_output = GALEX_NUV.add_band('NUV', GALEX_output, output_dir)

'''

# ## SDSS

# In[16]:


SDSS_catalogue = QTable()

SDSS_catalogue.add_column([f'HRS-{int(i)}' for i in HRS_optical_UV_catalogue['HRS']], name='ID')

coords = SkyCoord(HRS_optical_UV_catalogue['R.A.'], HRS_optical_UV_catalogue['Dec.'], unit=[u.hourangle, u.deg])
SDSS_catalogue.add_column(coords.ra, name='ra')
SDSS_catalogue.add_column(coords.dec, name='dec')

SDSS_catalogue.add_column(HRS_optical_UV_catalogue['D_25']/2 * u.arcmin, name='a')
SDSS_catalogue.add_column(SDSS_catalogue['a'] * (1 - HRS_optical_UV_photometry['e']), name='b')
SDSS_catalogue.add_column(HRS_optical_UV_photometry['PA'] * u.deg, name='pa')

for band in ['g', 'r', 'i']:
    SDSS_catalogue.add_column(
        3631*u.Jy * 10**(-.4 * np.array(
            np.where(HRS_optical_UV_photometry[f'{band}_D25'] == '...', 'nan', HRS_optical_UV_photometry[f'{band}_D25'])
        ).astype(float)), name=f'{band}_true_flux')
    SDSS_catalogue.add_column(SDSS_catalogue[f'{band}_true_flux'] * (
        1 - 10**(-.4 * np.array(np.where(HRS_optical_UV_photometry[f'{band}_D25'] == '...', 'nan', HRS_optical_UV_photometry[f'e{band}_D25'])).astype(float))), name=f'{band}_true_err')

# In[17]:

nanomaggies = 3.631e-6*u.Jy
beam = (0.40248 * u.arcsec)**2
#SDSS_u = HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-u', nanomaggies, beam)
SDSS_g = HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-g', nanomaggies, beam)
SDSS_r = HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-r', nanomaggies, beam)
SDSS_i = HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-i', nanomaggies, beam)
#SDSS_z = HiPS_skymap('https://alasky.cds.unistra.fr/SDSS/DR9/band-z', nanomaggies, beam)

# In[20]:

importlib.reload(astrobrowser)
#SDSS_output = SDSS_catalogue[177:179]
SDSS_output = SDSS_catalogue
SDSS_output = SDSS_g.add_band('g', SDSS_output, output_dir)
SDSS_output = SDSS_r.add_band('r', SDSS_output, output_dir)
SDSS_output = SDSS_i.add_band('i', SDSS_output, output_dir)

# ## Herschel
# In[21]:


PACS_catalogue = QTable()#names=['ra', 'dec_deg', 'a', 'b', 'pa'], units=[u.hourangle, u.deg, u.arcsec, u.arcsec, u.deg])

PACS_catalogue.add_column([f'HRS-{i}' for i in HRS_PACS_catalogue['HRS']], name='ID')

coords = SkyCoord(HRS_PACS_catalogue['R.A.__J.2000_'], HRS_PACS_catalogue['Dec__J.2000_'], unit=[u.hourangle, u.deg])
PACS_catalogue.add_column(coords.ra, name='ra')
PACS_catalogue.add_column(coords.dec, name='dec')
PACS_catalogue.add_column(HRS_PACS_catalogue['a'])
PACS_catalogue.add_column(HRS_PACS_catalogue['b'])
PACS_catalogue.add_column(HRS_PACS_catalogue['P.A.'], name='pa')

PACS_catalogue.add_column(HRS_PACS_catalogue['F_100'], name='PACS100_true_flux')
PACS_catalogue.add_column(HRS_PACS_catalogue['sigma_100'], name='PACS100_true_err')
PACS_catalogue.add_column(HRS_PACS_catalogue['F_160'], name='PACS160_true_flux')
PACS_catalogue.add_column(HRS_PACS_catalogue['sigma_160'], name='PACS160_true_err')

PACS_catalogue


# In[22]:


Herschel_PACS100 = HiPS_skymap('http://skies.esac.esa.int/Herschel/PACS100', u.Jy)
Herschel_PACS160 = HiPS_skymap('http://skies.esac.esa.int/Herschel/PACS160', u.Jy)


# In[23]:


importlib.reload(astrobrowser)
PACS_output = PACS_catalogue

PACS_output = Herschel_PACS100.add_band('PACS100', PACS_output, output_dir)
PACS_output = Herschel_PACS160.add_band('PACS160', PACS_output, output_dir)


# In[24]:


SPIRE_catalogue = QTable()#names=['ra', 'dec_deg', 'a', 'b', 'pa'], units=[u.hourangle, u.deg, u.arcsec, u.arcsec, u.deg])

SPIRE_catalogue.add_column([f'HRS-{i}' for i in HRS_SPIRE_catalogue['HRS_1']], name='ID')

coords = SkyCoord(HRS_SPIRE_catalogue['Ra'], HRS_SPIRE_catalogue['Dec'])
SPIRE_catalogue.add_column(coords.ra, name='ra')
SPIRE_catalogue.add_column(coords.dec, name='dec')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['a'])
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['b'])
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['pa'])

SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['S250'], name='SPIRE250_true_flux')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['err_tot250'], name='SPIRE250_true_err')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['S350'], name='SPIRE350_true_flux')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['err_tot350'], name='SPIRE350_true_err')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['S500'], name='SPIRE500_true_flux')
SPIRE_catalogue.add_column(HRS_SPIRE_catalogue['err_tot500'], name='SPIRE500_true_err')

# In[25]:


Herschel_SPIRE250 = HiPS_skymap('http://skies.esac.esa.int/Herschel/SPIRE250', u.MJy/u.sr)
Herschel_SPIRE350 = HiPS_skymap('http://skies.esac.esa.int/Herschel/SPIRE350', u.MJy/u.sr)
Herschel_SPIRE500 = HiPS_skymap('http://skies.esac.esa.int/Herschel/SPIRE500', u.MJy/u.sr)

# In[ ]:

SPIRE_output = SPIRE_catalogue
SPIRE_output = Herschel_SPIRE250.add_band('SPIRE250', SPIRE_output, output_dir)
SPIRE_output = Herschel_SPIRE350.add_band('SPIRE350', SPIRE_output, output_dir)
SPIRE_output = Herschel_SPIRE500.add_band('SPIRE500', SPIRE_output, output_dir)

'''
