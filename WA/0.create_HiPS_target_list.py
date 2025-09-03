#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create target list for the HiPS viewer

Created on August 4, 2023
@author: Yago Ascasibar
"""

from __future__ import print_function, division
from astropy.table import Table
from astropy.coordinates import SkyCoord
import astropy.units as u
import numpy as np

kpc_to_arcsec = 0.048128455  # assuming D=cz/H0 with H0=70 km/s/Mpc
pixel_size = .5 * kpc_to_arcsec
image_radius = 30 * kpc_to_arcsec

with open('HiPS_target_list.csv', 'w') as f:
    f.write('ID, RA, DEC, RADIUS_ARCSEC, PIXEL_SIZE_ARCSEC\n')
    t = Table.read('0.WA_sample_2023A1_40arcsec_2023-01-17_clean.csv')
    for entry in t:
        c = SkyCoord(entry['RA_OPTICAL'], entry['DEC_OPTICAL'], unit=(u.deg, u.deg))
        z = entry['HI_redshift']
        f.write(f"{entry['WA_ID']}, {c.ra.deg:10.6f}, {c.dec.deg:10.6f}, {image_radius/z:5.1f}, {pixel_size/z:4.1f}\n")
