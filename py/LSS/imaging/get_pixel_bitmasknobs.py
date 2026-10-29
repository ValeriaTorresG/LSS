# Get bitmask values from pixel-level per-brick masks for a catalog
# Examples:
# srun -N 1 -C haswell -c 64 -t 04:00:00 -q interactive python read_pixel_bitmaskmasknobs.py 
from __future__ import division, print_function
import sys, os, glob, time, warnings, gc
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, vstack, hstack, join
import fitsio

from astropy.io import fits
from astropy import wcs

from multiprocessing import Pool
import argparse


    

bitmask_dir = '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/'
bricks = Table(fitsio.read(
    '/global/cfs/cdirs/cosmo/data/legacysurvey/dr9/randoms/survey-bricks-dr9-randoms-0.48.0.fits',
    columns=['BRICKID', 'BRICKNAME', 'PHOTSYS'],
))
brick_index_by_id = {}
for brick_index, brickid in enumerate(bricks['BRICKID']):
    # np.where(...)[0][0] used the first row if an ID appeared more than once.
    brick_index_by_id.setdefault(int(brickid), brick_index)

_mask_worker_state = None


def _init_mask_worker(state):
    global _mask_worker_state
    _mask_worker_state = state


def _mask_worker(bid_index):
    return _mask_worker_state.wrapper(bid_index)


def bitmask_radec(brickid, ra, dec):

    brick_index = brick_index_by_id[int(brickid)]

    brickname = str(bricks['BRICKNAME'][brick_index])
    if bricks['PHOTSYS'][brick_index]=='N':
        field = 'north'
    elif bricks['PHOTSYS'][brick_index]=='S':
        field = 'south'
    else:
        # raise ValueError
        # Outside DR9 footprint; assign mask bit 16
        bitmask = np.full(len(ra), 2**16, dtype=np.uint32)
        nobsg = 0
        nobsr = 0
        nobsz = 0
        return bitmask,nobsg,nobsr,nobsz

    bitmask_fn = bitmask_dir+'{}/coadd/{}/{}/legacysurvey-{}-maskbits.fits.fz'.format(field, brickname[:3], brickname, brickname)

    bitmask_img = fitsio.read(bitmask_fn)

    header = fits.getheader(bitmask_fn, 1)
    w = wcs.WCS(header)

    coadd_x, coadd_y = w.wcs_world2pix(ra, dec, 0)
    coadd_x, coadd_y = np.round(coadd_x).astype(int), np.round(coadd_y).astype(int)

    bitmask = bitmask_img[coadd_y, coadd_x]
    nobsl = []
    bl = ['g','r','z']
    for band in bl:
        nobs_fn = bitmask_dir+'{}/coadd/{}/{}/legacysurvey-{}-nexp-{}.fits.fz'.format(field, brickname[:3], brickname, brickname,band)
        if os.path.isfile(nobs_fn): 
            nobs_img = fitsio.read(nobs_fn)
            nobs = nobs_img[coadd_y,coadd_x]
        else:
            nobs = 0
        nobsl.append(nobs)
    return bitmask,nobsl[0],nobsl[1],nobsl[2]

class get_nobsandmask:
    #doesn't seem like a class should be necessary but couldn't figure out multiprocessing otherwise
    def __init__(self,cat,nproc=128):
        #cat should be an astropy table
        #returns table with same ordering and NOBS_{G,R,Z} and MASKBITS columns
        for col in cat.colnames:
            if not col.isupper(): 
                cat.rename_column(col, col.upper())

        if 'TARGETID' not in cat.colnames:
            cat['TARGETID'] = np.arange(len(cat))

        if 'TARGET_RA' in cat.colnames:
            cat.rename_columns(['TARGET_RA', 'TARGET_DEC'], ['RA', 'DEC'])

        if 'INPUT_RA' in cat.colnames:
            cat.rename_columns(['input_ra', 'input_dec'], ['RA', 'DEC'])
        if 'BRICKID' not in cat.colnames:
            from desiutil import brick
            tmp = brick.Bricks(bricksize=0.25)
            cat['BRICKID'] = tmp.brickid(cat['RA'], cat['DEC'])
        # Just some tricks to speed up things up
        index_start = time.perf_counter()
        self.bidorder = np.argsort(cat['BRICKID'])
        sorted_bid = cat['BRICKID'][self.bidorder]
        if len(sorted_bid):
            boundaries = np.flatnonzero(sorted_bid[1:] != sorted_bid[:-1]) + 1
            self.bidcnts = np.concatenate(([0], boundaries, [len(sorted_bid)]))
            self.bid_unique = sorted_bid[self.bidcnts[:-1]]
        else:
            self.bidcnts = np.array([0])
            self.bid_unique = sorted_bid
        self.cat = cat

        print('adding nobs and mask values to '+str(len(cat))+' rows', flush=True)
        print(f'grouped {len(self.bid_unique)} bricks in {time.perf_counter() - index_start:.1f} s', flush=True)
    def wrapper(self,bid_index):

        idx = self.bidorder[self.bidcnts[bid_index]:self.bidcnts[bid_index+1]]
        brickid = self.bid_unique[bid_index]

        ra, dec = self.cat['RA'][idx], self.cat['DEC'][idx]
        tid = self.cat['TARGETID'][idx]
        bitmask,nobsg,nobsr,nobsz = bitmask_radec(brickid, ra, dec)

        data = Table()
        data['idx'] = idx
        data['MASKBITS'] = bitmask
        data['NOBS_G'] = nobsg
        data['NOBS_R'] = nobsr
        data['NOBS_Z'] = nobsz
        data['TARGETID'] = tid

        return data

    # start multiple worker processes
    def get_nobsandmask(self,nproc=128):
        # A bound method makes Pool pickle self (including the full catalog)
        # for every task batch. Give each worker the state once instead.
        workers_start = time.perf_counter()
        print(f'processing {len(self.bid_unique)} bricks with {nproc} workers', flush=True)
        with Pool(processes=nproc, initializer=_init_mask_worker, initargs=(self,)) as pool:
            res = pool.map(_mask_worker, range(len(self.bid_unique)))
        print(f'brick processing finished in {time.perf_counter() - workers_start:.1f} s', flush=True)

        assembly_start = time.perf_counter()
        res = vstack(res)
        # The returned idx values are a permutation of the input row numbers.
        # Scatter each column directly into that order instead of sorting all rows.
        idx = res['idx']
        ordered = Table()
        for col in ('MASKBITS', 'NOBS_G', 'NOBS_R', 'NOBS_Z', 'TARGETID'):
            values = np.empty(len(res), dtype=res[col].dtype)
            values[idx] = res[col]
            ordered[col] = values
        print(f'assembled {len(ordered)} rows in {time.perf_counter() - assembly_start:.1f} s', flush=True)
        return ordered

