'''
Find all potential assignment and counts tiles for DA2 mocks
Use the following environment
source /global/common/software/desi/desi_environment.sh main
'''

import numpy as np
import os
from astropy.table import Table, join, vstack
import argparse
from fiberassign.hardware import load_hardware, get_default_exclusion_margins
from fiberassign._internal import Hardware
from fiberassign.tiles import load_tiles
from fiberassign.targets import Targets, TargetsAvailable, LocationsAvailable, create_tagalong, load_target_file, targets_in_tiles
from fiberassign.assign import Assignment

from fiberassign.utils import Logger

from desitarget.io import read_targets_in_tiles
import desimodel.focalplane
import desimodel.footprint
import desimodel.io as dmio
trad = desimodel.focalplane.get_tile_radius_deg()*1.1 #make 10% greater just in case


import fitsio

import LSS.common_tools as common
#from LSS.imaging import get_nobsandmask
from LSS.main.cattools import count_tiles_better
from LSS.globals import main

import bisect
import time
import shutil
import tempfile
from datetime import datetime, timezone
import multiprocessing

t_start = time.time()

log = Logger.get()

parser = argparse.ArgumentParser()
parser.add_argument("--prog", choices=['DARK','BRIGHT'],default='DARK')
parser.add_argument("--mock", default='generic')
parser.add_argument("--mock_version",default='')
parser.add_argument("--realization")
parser.add_argument("--getcoll",default='y')
parser.add_argument("--base_output", help="base directory for output",default='/global/cfs/cdirs/desi/survey/catalogs/DA2/mocks/')
parser.add_argument("--tracer", help="tracer for CutSky EZ mocks", default=None)
parser.add_argument("--base_input", help="base directory for input for EZ mocks 6Gpc", default = None)
parser.add_argument("--tile_temp_dir", help="Directory for temp tile files, default %(default)s",
                    default=os.path.join(os.environ['SCRATCH'], 'rantiles'))
parser.add_argument("--counttiles", default = 'n')
parser.add_argument("--secgen_ver", default = None)
parser.add_argument("--nprocs", help="Number of multiprocessing processes to use, default %(default)i",
                    default=multiprocessing.cpu_count()//2, type=int)

# On Perlmutter, this read-only access point can be *much* faster thanks to aggressive caching.
#   If you didn't want this for some reason, you could revert '/dvs_ro/cfs/cdirs/desi' to '/global/cfs/cdirs/desi' in the following.
desi_input_dir = os.getenv('DESI_ROOT_READONLY', default='/dvs_ro/cfs/cdirs/desi')

args = parser.parse_args()
print(args)

if args.mock == 'ab2ndgen':
    #infn = args.base_output+'FirstGenMocks/AbacusSummit/forFA'+args.realization+'_matched_input_full_masknobs.fits'
    #infn = args.base_output+'SecondGenMocks/AbacusSummit/forFA'+args.realization+'.fits'
    infn = os.path.join(args.base_output+'SecondGenMocks', 'AbacusSummit'+args.mock_version, 'forFA'+args.realization+'.fits')
    log.info('Reading %s' % infn)
    tars = fitsio.read(infn)
    tarcols = list(tars.dtype.names)
    #tileoutdir = args.base_output+'SecondGenMocks/AbacusSummit/tartiles'+args.realization+'/'
    # tileoutdir = os.path.join(os.getenv('SCRATCH'), 'SecondGenMocks', 'AbacusSummit',args.prog , 'tartiles'+args.realization)
    tileoutdir = os.path.join(args.base_output, 'SecondGenMocks', 'AbacusSummit' + args.mock_version, 'tartiles' + args.realization)
    paoutdir = os.path.join(args.base_output+'SecondGenMocks', 'AbacusSummit'+args.mock_version, 'mock'+args.realization)
elif args.mock == 'ezmocks6':
    # #tr = args.tracer
    # rz = args.realization
    # print("Doing %s"%tr)
    #
    # if  tr == "LRG":
    #     infn1 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/LRG/z0.800/cutsky_LRG_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed%s_NGC.fits"%rz
    #     infn2 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/LRG/z0.800/cutsky_LRG_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed%s_SGC.fits"%rz
    # elif tr == "ELG":
    #     infn1 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/ELG/z1.100/cutsky_ELG_z1.100_EZmock_B6000G1536Z1.1N648012690_b0.345d1.45r40c0.05_seed%s_NGC.fits"%rz
    #     infn2 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/ELG/z1.100/cutsky_ELG_z1.100_EZmock_B6000G1536Z1.1N648012690_b0.345d1.45r40c0.05_seed%s_SGC.fits"%rz
    # elif tr == "QSO":
    #     infn1 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/QSO/z1.400/cutsky_QSO_z1.400_EZmock_B6000G1536Z1.4N27395172_b0.053d1.13r0c0.6_seed%s_NGC.fits"%rz
    #     infn2 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/QSO/z1.400/cutsky_QSO_z1.400_EZmock_B6000G1536Z1.4N27395172_b0.053d1.13r0c0.6_seed%s_SGC.fits"%rz
    ## infn1 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/LRG/z0.800/cutsky_LRG_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed1_NGC.fits"
    ## infn2 = "/global/cfs/cdirs/desi/cosmosim/FirstGenMocks/EZmock/CutSky_6Gpc/LRG/z0.800/cutsky_LRG_z0.800_EZmock_B6000G1536Z0.8N216424548_b0.385d4r169c0.3_seed1_SGC.fits"
    # tars1 = Table.read(infn1)#fitsio.read(infn1)
    # tars2 = Table.read(infn2)#fitsio.read(infn2)
    # tars1["GALCAP"] = "N"
    # tars2["GALCAP"] = "S"
    # tars = vstack([tars1, tars2])
    # tars['TARGETID'] = np.arange(len(tars))

    infn = os.path.join(args.base_input + 'EZMocks_6Gpc', 'EZMocks_6Gpc_' + args.realization + '.fits')
    tars = fitsio.read(infn)
    tarcols = list(tars.dtype.names)#['TARGETID','RA','DEC', 'Z','Z_COSMO','GALCAP', 'NZ', 'RAW_NZ']

    tileoutdir = os.path.join(args.base_output+'EZMocks_6Gpc', 'tartiles'+args.realization)
    paoutdir = os.path.join(args.base_output+'EZMocks_6Gpc', 'EzMocks', 'mock'+args.realization)
    if args.tracer is not None:
        tileoutdir = os.path.join(tileoutdir, args.tracer)
        paoutdir = os.path.join(paoutdir, args.tracer)

else: # args.mock == 'Generic':
    infn = args.base_input
    tars = fitsio.read(infn)
    tarcols = list(tars.dtype.names)
    tileoutdir = os.path.join(args.base_output.replace('global', os.getenv('SCRATCH')), args.mock, 'tartiles'+args.realization)
    paoutdir = os.path.join(args.base_output, 'mock'+args.realization) #os.path.join(args.base_output+'SecondGenMocks', 'AbacusSummit'+args.mock_version, 'mock'+args.realization)

# Ensure that the targets file is sorted by Dec.
t0 = time.time()
is_sorted = np.all(tars['DEC'][:-1] <= tars['DEC'][1:])
if not is_sorted:
    I = np.argsort(tars['DEC'])
    tars = tars[I]
t1 = time.time()
log.info('Sorting/verifying mocks: %.1f' % (t1-t0))

if not os.path.exists(tileoutdir):
    os.makedirs(tileoutdir)
    #print('made '+tileoutdir)
if not os.path.exists(paoutdir):
    os.makedirs(paoutdir)
    #print('made '+paoutdir)

# tiletab = Table.read(os.path.join(desi_input_dir, 'survey', 'catalogs', 'DA2', 'LSS', 'tiles-'+args.prog+'.fits'))
tiletab = Table.read(os.path.join(desi_input_dir, 'survey', 'catalogs', 'DA3', 'LSS', 'tiles-'+args.prog+'.fits'))
log.info('Reading startup globals: %.3f' % (time.time() - t_start))

def get_tile_targ(tile):
    '''
    Creates an astropy Table of (mock) targets within the given `tile`.
    '''
    tdec = tile['DEC']
    decmin = tdec - trad
    decmax = tdec + trad
    dec = tars['DEC']
    # The `tars` global table of targets is sorted by Dec.  We therefore only need to look at
    # indices that can possibly be within range given just the Dec distance (decmin to decmax).
    # "bisect_left" is way faster than "np.searchsorted"!
    #i0,i1 = np.searchsorted(dec, [np.float32(decmin), np.float32(decmax)])
    i0 = bisect.bisect_left(dec, decmin)
    i1 = bisect.bisect_left(dec, decmax, lo=i0)
    Idec = slice(i0, i1+1)
    inds = desimodel.footprint.find_points_radec(tile['RA'], tdec,
                                                 tars['RA'][Idec], tars['DEC'][Idec])
    # dtype=int so that a tile with no targets gives an empty (not float) index array
    rtw = tars[i0 + np.array(inds, dtype=int)]
    rmtl = Table(rtw)
    #print('made table')
    del rtw
    if 'DESI_TARGET' not in tarcols:
        rmtl['DESI_TARGET'] = np.ones(len(rmtl),dtype=int)*2
    if 'NUMOBS_INIT' not in tarcols:
        rmtl['NUMOBS_INIT'] = np.zeros(len(rmtl),dtype=int)
    if 'NUMOBS_MORE' not in tarcols:
        rmtl['NUMOBS_MORE'] = np.ones(len(rmtl),dtype=int)
    if 'PRIORITY' not in tarcols:
        rmtl['PRIORITY'] = np.ones(len(rmtl),dtype=int)*3400
    #if 'OBSCONDITIONS' not in tarcols:
    rmtl['OBSCONDITIONS'] = np.ones(len(rmtl),dtype=int)*516#forcing it to match value assumed below
    if 'SUBPRIORITY' not in tarcols:
        rmtl['SUBPRIORITY'] = np.random.random(len(rmtl))
    return rmtl

def write_tile_targ(ind):
    '''
    Write the targets file for the single tile table index "ind".
    '''
    tile = tiletab[ind]
    fname = os.path.join(tileoutdir, 'tilenofa-'+str(tile['TILEID'])+'.fits')
    log.info('creating %s' % fname)
    rmtl = get_tile_targ(tile)
    #print('added columns for', fname)
    rmtl.write(fname, format='fits', overwrite=True)
    #print('added columns, wrote to', fname)
    return len(rmtl)

margins = get_default_exclusion_margins()
rann = 0
n = 0

def getpa(ind, fbah, hw):
    '''
    `fbah` is the fiberassign header dict from read_fba_header() (read once in main() instead
    of again here) and `hw` is the Hardware matching its RUNDATE.
    '''
    #tile = 1230
    tile = tiletab[ind]['TILEID']

    tilefn = os.path.join(args.tile_temp_dir, str(tile)+'-'+str(rann)+'-tiles.fits')

    pr = args.prog
    if not os.path.isfile(tilefn):
        t = Table(tiletab[ind])
        t['OBSCONDITIONS'] = 516
        t['IN_DESI'] = 1
        t['MTLTIME'] = fbah['MTLTIME']
        t['FA_RUN'] = fbah['FA_RUN']
        t['PROGRAM'] = pr

    obsha = fbah['FA_HA']
    obstheta = fbah['FIELDROT']


    if not os.path.isfile(tilefn):
        t.write(tilefn, overwrite=True)

    tiles = load_tiles(
        tiles_file=tilefn, obsha=obsha, obstheta=obstheta,
        select=[tile])

    tids = tiles.id
    #print('Tile ids:', tids)
    I = np.flatnonzero(np.array(tids) == tile)
    assert(len(I) == 1)
    i = I[0]
    tile_ra  = tiles.ra[i]
    tile_dec = tiles.dec[i]

    # Create empty target list
    tgs = Targets()
    # Create structure for carrying along auxiliary target data not needed by C++.
    plate_radec=True
    tagalong = create_tagalong(plate_radec=plate_radec)
    
    #print(tile)
    # Load target files...
    tilenofafn = os.path.join(tileoutdir, 'tilenofa-%i.fits' % tile)
    load_target_file(tgs, tagalong, tilenofafn)
    #loading it again straight to table format because I can't quickly figure out exactly where targetid,ra,dec gets stored
    tar_tab = fitsio.read(tilenofafn, columns=tarcols)

    # Find targets within tiles, and project their RA,Dec positions
    # into focal-plane coordinates.
    tile_targetids, tile_x, tile_y, tile_xy_cs5 = targets_in_tiles(hw, tgs, tiles, tagalong)
    # Compute the targets available to each fiber for each tile.
    tgsavail = TargetsAvailable(hw, tiles, tile_targetids, tile_x, tile_y)
    # Compute the fibers on all tiles available for each target and sky
    favail = LocationsAvailable(tgsavail)

    # FAKE stucksky (positioners that happen to be stuck on good sky positions)
    stucksky = {}

    # Create assignment object
    asgn = Assignment(tgs, tgsavail, favail, stucksky)
    tgsavail = asgn.targets_avail()
    avail = tgsavail.tile_data(tile)
    fibers = hw.loc_fiber

    # The "FAVAIL" (available targets) HDU is sorted first by LOCATION,
    # then by TARGETID.  Built with whole-array numpy ops instead of filling
    # astropy columns location by location (same values and dtypes).
    lids = sorted(avail.keys())
    ntg = np.array([len(avail[lid]) for lid in lids], dtype=int)
    if np.sum(ntg) == 0:
        # No target reachable by any fiber: nothing to add for this tile.
        return None
    fdata = Table()
    fdata['LOCATION'] = np.repeat(np.array(lids, dtype=int), ntg)
    fdata['FIBER']    = np.repeat(np.array([fibers[lid] for lid in lids], dtype=int), ntg)
    fdata['TARGETID'] = np.concatenate([np.sort(np.array(avail[lid], dtype=int)) for lid in lids])
    fdata = join(fdata,tar_tab,keys=['TARGETID'],join_type='left')
    if args.getcoll == 'y':
        coll = asgn.check_avail_collisions(tile)
        if len(coll) > 0:
            kl = np.array(list(coll.keys())).transpose()
            locs = kl[0]
            ids = kl[1]
            locids = ids*10000+locs
        else:
            locids = np.zeros(0, dtype=int)
        log.info('N collisions: %i' % len(coll))
        locidsin = np.isin(fdata['LOCATION']+10000*fdata['TARGETID'],locids)
        log.info('N collisions original: %i %i' % (np.sum(locidsin),len(fdata)))
        fdata['COLLISION'] = locidsin
    #colltab = Table(forig[locidsin])
    fdata['TILEID'] = tile
    return fdata

def run_one_tile(ind, fbah, hw):
    t0 = time.time()
    ntarg = write_tile_targ(ind)
    res = None
    if ntarg > 0:
        res = getpa(ind, fbah, hw)
    if res is None:
        log.info('Tile %i: no available targets (%i targets within radius), skipping' % (tiletab[ind]['TILEID'], ntarg))
        return None
    res = np.array(res)
    t1 = time.time()
    log.info('Tile %i took %.3f sec' % (tiletab[ind]['TILEID'], t1-t0))
    return res

# Per-process cache of the Hardware currently in use: (index into hardware_times, hw).
_hw_cache = (None, None)

def run_tile_group(task):
    '''
    Run a group of tiles that all share the same hardware configuration.
    task = (ihw, [(ind, fbah), ...]), where hardware_times[ihw] holds the RUNDATE whose
    load_hardware() call defines that configuration (exactly what the serial version
    used to load in the main process).  The Hardware is built here, in the worker, so
    the ~600 configurations are loaded in parallel instead of in series.
    '''
    global _hw_cache
    ihw, tiles = task
    if _hw_cache[0] != ihw:
        _hw_cache = (None, None)
        hw = load_hardware(rundate=hardware_times[ihw][2], add_margins=margins, get_time_range=True)[0]
        _hw_cache = (ihw, hw)
    hw = _hw_cache[1]
    out = []
    for ind, fbah in tiles:
        res = run_one_tile(ind, fbah, hw)
        if res is None:
            continue
        # Encode the FITS table here, in parallel; main() only concatenates the bytes.
        fn = os.path.join(chunk_dir, 'pota-%i.fits' % tiletab[ind]['TILEID'])
        fitsio.write(fn, res, extname='LSS', clobber=True)
        out.append((fn, len(res), np.sum(res['COLLISION'])))
    return out

class FitsTableConcatenator:
    '''
    Concatenate FITS files that each hold one binary table (written by fitsio, with
    identical columns) into a single file.  The result is byte-for-byte what fitsio
    produces with write() of the first table and append() of the others, but the rows
    are copied already encoded (by the workers, in parallel) instead of being encoded
    again in this process, which is what made the serial writing slow.
    '''
    def __init__(self, fn):
        self.out = open(fn, 'wb')
        self.nrows = 0
        self.head = None

    @staticmethod
    def _read(fn):
        with fitsio.FITS(fn) as f:
            hdu = f[1]
            data_start = hdu.get_offsets()['data_start']
            rowlen = hdu.read_header()['NAXIS1']
            nrows = hdu.get_nrows()
        with open(fn, 'rb') as f:
            buf = f.read()
        return buf[:data_start], buf[data_start:data_start + nrows*rowlen], nrows

    @staticmethod
    def _naxis2_offset(head):
        # Offset of the (last, i.e. the table's) NAXIS2 card in the header bytes.
        offs = [i for i in range(0, len(head), 80) if head[i:i+10] == b'NAXIS2  = ']
        return offs[-1]

    def add(self, fn):
        head, data, nrows = self._read(fn)
        if self.head is None:
            self.head = head
            self.inaxis2 = self._naxis2_offset(head)
            self.out.write(head)
        else:
            # Same primary HDU and table header (columns, formats...) except for NAXIS2.
            i = self._naxis2_offset(head)
            blank = b' '*80
            assert (len(head) == len(self.head) and i == self.inaxis2 and
                    head[:i] + blank + head[i+80:] == self.head[:i] + blank + self.head[i+80:]), \
                'FITS header of %s does not match the first table' % fn
        self.out.write(data)
        self.nrows += nrows

    def close(self):
        # Pad the data to a whole FITS block and set the final number of rows.
        pad = (-self.out.tell()) % 2880
        self.out.write(b'\0'*pad)
        card = self.head[self.inaxis2:self.inaxis2+80]
        card = card[:10] + (b'%20d' % self.nrows) + card[30:]
        self.out.seek(self.inaxis2)
        self.out.write(card)
        self.out.close()

# Directory for the per-tile FITS tables written by the workers; set in main().
chunk_dir = None

def read_fba_header(ind):
    '''
    Read the fiberassign header for one tile index.
    '''
    tile = tiletab['TILEID'][ind]
    ts = '%06i' % tile
    fbah = fitsio.read_header(os.path.join(desi_input_dir, 'target', 'fiberassign', 'tiles', 'trunk', ts[:3], 'fiberassign-'+ts+'.fits.gz'))
    return dict([(k, fbah[k]) for k in ['RUNDATE', 'MTLTIME', 'FA_RUN', 'FA_HA', 'FIELDROT']])

def parse_datetime(s):
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        d = datetime.strptime(s, "%Y-%m-%dT%H:%M:%S")
        # msg = "Requested run date '{}' is not timezone-aware.  Assuming UTC.".format(runtime)
        d = d.replace(tzinfo=timezone.utc)

def focalplane_time_range(rundate):
    '''
    Return the (time_lo, time_hi) that load_hardware(rundate=rundate, get_time_range=True)
    would return, without building the (expensive) Hardware object.  The rundate parsing
    and load_focalplane() call are the same ones done inside fiberassign's
    load_hardware_args().
    '''
    try:
        runtime = datetime.strptime(rundate, "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        runtime = datetime.strptime(rundate, "%Y-%m-%dT%H:%M:%S")
        runtime = runtime.replace(tzinfo=timezone.utc)
    fp, exclude, state, create_time, time_lo, time_hi = dmio.load_focalplane(runtime, get_time_range=True)
    return time_lo, time_hi

# List of (time_lo, time_hi, rundate): the hardware configurations, as the time range
# over which each is valid and the RUNDATE to pass to load_hardware() to build it.
hardware_times = []
def get_hardware_for_time(t):
    '''
    Return the index into `hardware_times` of the configuration valid at time `t`.
    '''
    global hardware_times
    for i,(tlo,thi,rundate) in enumerate(hardware_times):
        if (tlo <= t) and (thi is None or thi > t):
            #print('Match to time range', tlo, 'to', thi)
            return i
    return None

def main():
    # Use fork explicitly (Python >= 3.14 defaults to forkserver): the workers need the
    # `tars` and `tiletab` globals without copying/pickling them.
    Pool = multiprocessing.get_context('fork').Pool
    tls = list(tiletab['TILEID'])
    #inds = np.flatnonzero(np.array(tls) == 1230)
    #inds = np.arange(256)
    inds = np.arange(len(tls))

    t0 = time.time()
    # Read all fiberassign headers to get the RUNDATES.
    with Pool(processes=args.nprocs) as pool:
        headers = pool.map(read_fba_header, inds)
    rundates = set([h['RUNDATE'] for h in headers])
    rundates = sorted(list(rundates))
    log.info('Unique rundates: %i of %i' % (len(rundates), len(headers)))
    t1 = time.time()
    log.info('Reading fiberassign headers in parallel: %.3f sec' % (t1-t0))

    # Find the hardware configurations for our RUNDATES.  This reproduces exactly the
    # serial loop
    #     for t in rundates:
    #         if get_hardware_for_time(parse_datetime(t)) is not None: continue
    #         hw,time_lo,time_hi = load_hardware(rundate=t, add_margins=margins, get_time_range=True)
    #         hardware_times.append((time_lo, time_hi, hw))
    # but gets the time ranges (cheap) in parallel and defers building each Hardware
    # (expensive) to the workers.  The first call warms up desimodel's focalplane cache,
    # which the forked workers then inherit.
    global hardware_times, _hw_cache
    focalplane_time_range(rundates[0])
    with Pool(processes=args.nprocs) as pool:
        time_ranges = pool.map(focalplane_time_range, rundates)
    for t,(time_lo,time_hi) in zip(rundates, time_ranges):
        dt = parse_datetime(t)
        cached = get_hardware_for_time(dt)
        if cached is not None:
            continue
        hardware_times.append((time_lo, time_hi, t))
    log.info('Hardware configurations: %i' % len(hardware_times))

    # Group the tiles by hardware configuration, in chunks of at most `group_size` tiles,
    # so that each worker builds a given Hardware once for several tiles.
    group_size = 8
    tiles_by_hw = {}
    for ind,h in zip(inds, headers):
        ihw = get_hardware_for_time(parse_datetime(h['RUNDATE']))
        assert(ihw is not None)
        tiles_by_hw.setdefault(ihw, []).append((ind, h))
    tasks = []
    for ihw in sorted(tiles_by_hw.keys()):
        tl = tiles_by_hw[ihw]
        for i in range(0, len(tl), group_size):
            tasks.append((ihw, tl[i:i+group_size]))
    # Build the first configuration here so that it (and the rest of fiberassign's
    # warm-up) is inherited by the forked workers.
    _hw_cache = (0, load_hardware(rundate=hardware_times[0][2], add_margins=margins, get_time_range=True)[0])

    t2 = time.time()
    log.info('Finding hardware time ranges in parallel: %.3f sec' % (t2-t1))

    # Write output *while* retrieving results in parallel.  Results come back in the
    # (fixed) order of `tasks`, so the row order of the output is reproducible; the
    # serial-loading version wrote tiles in whatever order they finished.
    outfn = os.path.join(paoutdir, 'pota-'+args.prog+'.fits')
    tempout = outfn + '.tmp'
    # Per-tile tables go to node-local memory (/dev/shm) when available; each is
    # deleted as soon as it has been copied into the output.
    global chunk_dir
    chunk_dir = tempfile.mkdtemp(prefix='getpota-', dir='/dev/shm' if os.path.isdir('/dev/shm') else paoutdir)
    try:
        fits = FitsTableConcatenator(tempout)
        ntot = 0
        ncoll = 0
        with Pool(processes=args.nprocs) as pool:
            it = pool.imap(run_tile_group, tasks)
            # fetch results as they complete
            for group in it:
                for fn, nrows, nc in group:
                    ntot += nrows
                    ncoll += nc
                    fits.add(fn)
                    os.remove(fn)
        fits.close()
    finally:
        shutil.rmtree(chunk_dir, ignore_errors=True)
    os.rename(tempout, outfn)
    log.info('Wrote %s' % outfn)
    t3 = time.time()
    if args.getcoll == 'y':
        log.info('%i %i' % (ntot, ncoll))
    log.info('Running tiles and writing results: %.3f sec' % (t3-t2))

if __name__ == '__main__':
    main()
