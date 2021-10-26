#!/usr/bin/env python3
"""
Simple aperture photometry script inspired by Attila Bódi's Python for
Astronomers lectures, especially this presentation:
https://docs.google.com/presentation/d/1OEqOec1ETJmXzkTRLNdwiLwQkULtLzhvYUtBcs_CBsc/present#slide=id.gc6f980f91_0_0
More details are here:
https://sites.google.com/view/pythonforastronomers/home

The working directory must be defined in an argument.
The script assumes that the calibration files are in the same folder as the
object frames. Image selection is done via header informations, so the image
headers should contain the necessary header keywords, e.g. imgtype, exptime,
filter, etc.

The script creates the master bias, then the master dark(s) and master flat(s).
Then, the object frames are processed. For this parallel processing is used.
All the available CPU cores are utilized for this. For machines with smaller
amount of RAM this is not recommended. In this case change
pool = mp.Pool(mp.cpu_count())
to
pool = mp.Pool(4), or a smaller number.

During this process the object frames are trimmed and registered to a reference
image.

Then the FWHM gets determined via the imexam task and ds9.

The next step is source detection with DAOStarFinder. The sources are filtered,
the most faintest and the most brightest are excluded from the list.

Then the script asks for the id of the target and six comparison stars. An
object frame is displayed with the sources and their number.

After this the right aperture will be determined via the display of the curve
of growth.

With this information aperture photometry is performed on the images, also in
parallel mode.

Finally the comp check plots and the differential lightcurve of the target is
displayed and saved to the working directory.

Right now the script can create master flats for multiple filters, but during
the photometry it assumes that all the object frames were taken in the same
filter.
TBD: Implement photometry for multiple filters.

In the current version the script assumes that dark frames ware taken for all
the exposition times and cannot do exptime scaling.
TBD: Implement exptime scaling for dark frames if a master dark does not
exists for the flats or object frames.

Dependencies:
    -astropy
    -imexam
    -ccdproc
    -image_registration
    -photutils
    -matplotlib
    -ds9 (installed in the system)

Parallel processing tutotials:
https://www.machinelearningplus.com/python/parallel-processing-python/
"""
import os
import sys
import getopt
import time
import imexam
import glob
from astropy.table import Table
from astropy.io import fits
import numpy as np
import ccdproc
from astropy.nddata import CCDData
from astropy import units as u
from astropy.stats import sigma_clipped_stats  # Measure bkg
from image_registration import chi2_shift  # Measure shift
from image_registration.fft_tools import shift  # Apply shift
# from image_registration import register_images
from photutils import DAOStarFinder
from photutils import CircularAperture, CircularAnnulus
from photutils import aperture_photometry
from astropy.stats import SigmaClip  # Simple sigma clipping
from photutils import Background2D, MedianBackground  # 2D stat
from photutils.utils import calc_total_error
import matplotlib.pyplot as plt
import multiprocessing as mp
import warnings
warnings.filterwarnings("ignore")


###############################################################################
# Converts flux to magnitude
def flux2mag(flux):
    return -2.5*np.log10(flux)+25.


###############################################################################
# Print iterations progress
# https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in
                                  percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration /
                                                            float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end=printEnd)
    # Print New Line on Complete
    if iteration == total:
        print()
###############################################################################


def process_object_frames(ofile, ref_image_data, mbias, i, ll):
    """

    Parameters
    ----------
    ofile : string
        Input file for processing.
    ref_image_data : CCDData
        Reference image for image registration.
    mbias : hdu.data
        Master bias file for image reduction.
    i : integer
        Loop variable for progress bar.
    l : integer
        Length of object list for progress bar.

    Returns
    -------
    hdu : Fits hdu object.
        Returns the cropped and registred hdu object.

    """
    if os.path.exists('cal/'+str(ofile)) is False:
        # print(f"Processing {ofile}")
        hdu = fits.open(ofile)
        header = hdu[0].header
        exp = hdu[0].header['exptime']
        filt = hdu[0].header['filter']
        hdu_d = fits.open('master/dark'+str(exp)+'.fits')
        mdark = hdu_d[0].data
        hdu_f = fits.open('master/flat'+str(filt)+'.fits')
        mflat = hdu_f[0].data
        data = hdu[0].data
        data_c = (data - mdark - mbias)
        data_c /= (mflat/mflat.max())
        # Trim calibrated images
        data_c = ccdproc.trim_image(CCDData(data_c, unit=u.adu),
                                    fits_section='[1400:2650, 1400:2650 ]')
        # Image registration. First image is the reference.
        mean, median, std = sigma_clipped_stats(data_c, sigma=3.0)
        xoff, yoff, exoff, eyoff = chi2_shift(ref_image_data, data_c,
                                              mean,
                                              return_error=True,
                                              upsample_factor=1)
        # upsample_factor='auto' causes artifacts around bright sources
        data_c = shift.shiftnd(data_c, (-yoff, -xoff))

        hdu = fits.PrimaryHDU(data_c, header=header)
        hdu.writeto('cal/'+str(ofile))
        return hdu
    printProgressBar(i+1, ll, prefix='Progress:', suffix='Complete',
                     length=50)


###############################################################################


def get_aperture(xs, ys, radii, r_in, r_out):
    """

    Parameters
    ----------
    xs : list
        List of x pixel coordinates for sources.
    ys : list
        List of y pixel coordinates for sources.
    radii : list
        List of aperture radii.
    r_in : integer
        Radius of inner background annulus.
    r_out : integer
        Radius of outer background annulus.

    Returns
    -------
    apnum : integer
        Returns the chosen aperture number.

    """
    hdu = fits.open('cal/'+str(objlist['fname'][0]))
    data = hdu[0].data
    header = hdu[0].header
    positions = np.transpose((xs, ys))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    bkg_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    effective_gain = header['EXPTIME']  # exposure time in sec
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg2D = Background2D(data, (25, 25), filter_size=(3, 3),
                         sigma_clip=sigma_clip,
                         bkg_estimator=bkg_estimator)
    error = calc_total_error(data, bkg2D.background, effective_gain)
    phot = aperture_photometry(data, apertures, error=error)
    bkg = aperture_photometry(data, bkg_apertures, error=error)
    bkg_mean = bkg['aperture_sum'] / bkg_apertures.area

    for j in range(len(radii)):
        # Substracting area proportional background from apertures
        bkg_sum = bkg_mean * apertures[j].area
        phot['bkg_sum'] = bkg_sum
        flux_bkgsub = phot['aperture_sum_'+str(j)] - bkg_sum
        phot['aperture_sum_bkgsub_'+str(j)] = flux_bkgsub  # Update table

    radnum = []
    fx = []
    err = []
    for ii in range(len(radii)):
        radnum.append(ii)
        fx.append(phot['aperture_sum_bkgsub_'+str(ii)][ids[0]])
        varerr = phot['aperture_sum_err_'+str(ii)][ids[0]]
        err.append(1.0857*(varerr/phot['aperture_sum_bkgsub_' +
                   str(ii)][ids[0]]))
    # Plotting the CoG.
    plt.errorbar(radnum, flux2mag(fx), yerr=err, fmt='.')
    plt.gca().invert_yaxis()
    plt.xlabel('Aperture number')
    plt.ylabel('Instrumental Magnitude')
    plt.savefig('cal/cog.png', dpi=150)
    plt.show()
    print("Type aperture number, starting from 0!:")
    apnum = int(input())
    return apnum


###############################################################################

def do_phot(ifile, xs, ys, radii, i, ids):
    """

    Parameters
    ----------
    ifile : string
        Input file for processing.
    xs : list
        List of x pixel coordinates for sources.
    ys : list
        List of y pixel coordinates for sources.
    radii : list
        List of aperture radii.
    i : integer
        Loop variable for progress bar.
    ids : list
        List of the source ids, selected for the photometry. The first is the
        target.

    Returns
    -------
    result : array
        Returns the JD and the background substracted fluxes for the selected
        stars and aperture number.

    """
    hdu = fits.open('cal/'+str(ifile))
    data = hdu[0].data
    header = hdu[0].header
    jd = header['jd']  # Getting julian date from header
    positions = np.transpose((xs, ys))
    apertures = [CircularAperture(positions, r=r) for r in radii]
    bkg_apertures = CircularAnnulus(positions, r_in=r_in, r_out=r_out)
    effective_gain = header['EXPTIME']  # exposure time in sec
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = MedianBackground()
    bkg2D = Background2D(data, (25, 25), filter_size=(3, 3),
                         sigma_clip=sigma_clip,
                         bkg_estimator=bkg_estimator)
    error = calc_total_error(data, bkg2D.background, effective_gain)
    phot = aperture_photometry(data, apertures, error=error)
    bkg = aperture_photometry(data, bkg_apertures, error=error)
    bkg_mean = bkg['aperture_sum'] / bkg_apertures.area

    for j in range(len(radii)):
        # Substracting area proportional background from apertures
        bkg_sum = bkg_mean * apertures[j].area
        phot['bkg_sum'] = bkg_sum
        flux_bkgsub = phot['aperture_sum_'+str(j)] - bkg_sum
        phot['aperture_sum_bkgsub_'+str(j)] = flux_bkgsub  # Update table
    result = np.array([jd])  # Convert JD to array
    printProgressBar(i+1, l, prefix='Progress:', suffix='Complete',
                     length=50)
    phot.write('cal/'+str(ifile).replace('.fits', '.phot'),
               format='ascii', overwrite=True)
    for ID in ids:
        # Keeping the data only for the selected ids.
        result = np.hstack((result,
                            phot['aperture_sum_bkgsub_'+str(apnum)][ID],
                            phot['aperture_sum_err_'+str(apnum)][ID]))
    return result


###############################################################################
if __name__ == "__main__":

    wdir = ""
    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]
    unixOptions = "h:wd"
    gnuOptions = ["help", "wdir="]

    try:
        arguments, values = getopt.getopt(
            argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-wd", "--wdir"):
            wdir = currentValue
        elif currentArgument in ("-h", "--help"):
            print("Usage:")
            print("python3 python_example.py --wdir=<workdir>")
            print("To get help: ")
            print("python3 python_example.py --help")
            sys.exit()
    # Set working directory
    if not wdir:
        print("No working directory was given. Exiting.")
        sys.exit()

    os.chdir(wdir)
    if os.path.exists('master') is False:
        os.mkdir('master')
    if os.path.exists('cal') is False:
        os.mkdir('cal')

    itype = []
    filter = []
    fname = []
    exptime = []

    files = glob.glob("*.fits")
    # Collecting data from headers
    for file in files:
        hdul = fits.open(file)
        itype.append(hdul[0].header['imagetyp'])
        filter.append(hdul[0].header['filter'])
        fname.append(str(file))
        exptime.append(hdul[0].header['exptime'])
    # Creating astropy.table with the data
    info = Table([fname, itype, filter, exptime],
                 names=('fname', 'itype', 'filter', 'exptime'),
                 meta={'name': 'first table'})

    # Creating lists
    biaslist = info[np.where(info['itype'] == 'bias')]
    darklist = info[np.where(info['itype'] == 'dark')]
    flatlist = info[np.where(info['itype'] == 'flat')]
    objlist = info[np.where(info['itype'] == 'object')]

    # Master bias
    if os.path.exists('master/mbias.fits') is False:
        print("Creating master bias.")
        biases = []
        for i in range(len(biaslist)):
            hdu = fits.open(biaslist['fname'][i])
            biases.append(hdu[0].data)

        mbias = np.median(biases, axis=0)
        hdu = fits.PrimaryHDU(mbias)
        hdu.writeto('master/mbias.fits')
    else:
        print("Master bias already present!")

    # Getting unique dark expsosure times
    exps = np.unique(darklist['exptime'])

    # Creating master dark(s)
    for exp in exps:
        if os.path.exists('master/dark'+str(exp)+'.fits') is False:
            print(f"Creating master dark with {exp} second exptime.")
            if 'mbias' not in locals():
                hdul = fits.open('master/mbias.fits')
                mbias = hdul[0].data
            darks = []
            dlist = darklist[np.where(darklist['exptime'] == exp)]
            for i in range(len(dlist)):
                hdu = fits.open(dlist['fname'][i])
                darks.append(hdu[0].data - mbias)
            mdark = np.median(darks, axis=0)
            hdu = fits.PrimaryHDU(mdark)
            hdu.writeto('master/dark'+str(exp)+'.fits')
        else:
            print(f"Master dark with {exp} second exptime is already present.")

    # Getting unique flat filters
    filters = np.unique(flatlist['filter'])

    # Master flat(s)
    for filter in filters:
        if os.path.exists('master/flat'+str(filter)+'.fits') is False:
            print(f"Creating master flat for {filter} filter")
            if 'mbias' not in locals():
                hdul = fits.open('master/mbias.fits')
                mbias = hdul[0].data
            flats = []
            flist = flatlist[np.where(flatlist['filter'] == filter)]
            for i in range(len(flist)):
                hdu = fits.open(flist['fname'][i])
                exp = hdu[0].header['exptime']
                hdul_d = fits.open('master/dark'+str(exp)+'.fits')
                mdark = hdul_d[0].data
                flats.append(hdu[0].data - mdark - mbias)
            mflat = np.median(flats, axis=0)
            hdu = fits.PrimaryHDU(mflat)
            hdu.writeto('master/flat'+str(filter)+'.fits')
        else:
            print(f"Master flat for {filter} filter is already present.")

    # Correcting object frames for bias, dark and flat
    print("Processing object frames.")
    ll = len(objlist)
    if 'mbias' not in locals():
        hdul = fits.open('master/mbias.fits')
        mbias = hdul[0].data
    results = []
    hdu_ref = fits.open(objlist['fname'][0])
    ref_image_data = hdu_ref[0].data
    ref_image_data = ccdproc.trim_image(
        CCDData(ref_image_data, unit=u.adu),
        fits_section='[1400:2650, 1400:2650 ]')

    # pool = mp.Pool(4)
    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(process_object_frames, args=(
        ofile, ref_image_data, mbias, i, ll))
        for i, ofile in enumerate(objlist['fname'])]

    results = [r.get() for r in result_objects]
    pool.close()
    pool.join()

    # Determining FWHM from image with imexam
    if os.path.exists('cal/fwhm.dat') is False:
        viewer = imexam.connect()
        time.sleep(3)
        viewer.load_fits("cal/"+objlist['fname'][0])
        viewer.scale()

        viewer.imexam()
        print("Type fwhm size:")
        fwhm = float(input())
        f = open("cal/fwhm.dat", "w")
        f.write(str(fwhm))
        f.close()
    else:
        # If FWHM data file is present it will read the value from it.
        fwhm = float(np.loadtxt('cal/fwhm.dat'))

    # Findign sources on the first image
    hdu = fits.open('cal/'+str(objlist['fname'][0]))
    data = hdu[0].data
    header = hdu[0].header
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=5.0*std)
    sources = daofind(data - median)
    for col in sources.colnames:
        # for consistent table output
        sources[col].info.format = '%.8g'

    # Filtering out too faint and too bright sources
    sources = sources[np.where((sources['peak'] > 1000.0) &
                               (sources['peak'] < 40000.0))]

    # Fixing ids
    for i in range(1, len(sources)):
        sources['id'][i] = i
    # Writing the found and filtered sources to a file
    sources.write('cal/xy', format='ascii', overwrite=True)

    # Creating locations list for imexam to plot
    locations = []
    for point in range(0, len(sources['xcentroid']), 1):
        locations.append((sources['xcentroid'][point],
                          sources['ycentroid'][point], sources['id'][point]))

    # We need the id of the target and 6 comparison stars.
    # If the list does not exist we plot the first frame and the sources on it
    # in ds9.
    if os.path.exists('cal/ids.dat') is False:
        viewer = imexam.connect()
        time.sleep(3)
        viewer.load_fits('cal/'+str(objlist['fname'][0]))
        viewer.scale()
        viewer.mark_region_from_array(locations)
        viewer.imexam()

        ids = []
        x = []
        y = []
        print("Type the number of the target:")
        ids.append(int(input()))
        x.append(sources['xcentroid'][np.where(sources['id'] == ids[0])][0])
        y.append(sources['ycentroid'][np.where(sources['id'] == ids[0])][0])
        for i in range(1, 7):
            print(f"Type the number of the {i}. comparison star:")
            ids.append(int(input()))
            x.append(sources['xcentroid']
                     [np.where(sources['id'] == ids[i])][0])
            y.append(sources['ycentroid']
                     [np.where(sources['id'] == ids[i])][0])
        # ids = [116, 53, 52, 137, 44, 174, 60]  # from phot table later!
        print(ids, x, y)
        ids_data = Table([ids, x, y],
                         names=('ids', 'x', 'y'),
                         meta={'name': 'first table'})
        ids_data.write('cal/ids.dat', format='ascii')
    else:
        # If the data file with the target and comparisons id exists we read it
        # from the disk.
        ids_data = Table.read('cal/ids.dat', format='ascii')
        ids = ids_data['ids']
    print("Starting aperture photometry.")
    # List of aperture radii used for photometry
    radii = [fwhm/2., fwhm, fwhm+2., fwhm+4., fwhm+6., fwhm+8., fwhm+10.]
    # Set the outer annulus size.
    r_in = fwhm+14.
    r_out = fwhm+22.

    apnum = get_aperture(sources['xcentroid'],
                         sources['ycentroid'], radii, r_in, r_out)

    pool = mp.Pool(mp.cpu_count())
    result_objects = [pool.apply_async(do_phot, args=(
        ifile, sources['xcentroid'], sources['ycentroid'], radii, i, ids))
        for i, ifile in enumerate(objlist['fname'])]

    # result_objects is a list of pool.ApplyResult objects
    results = [r.get() for r in result_objects]
    pool.close()
    pool.join()

    for i, result in enumerate(results):
        if i == 0:
            lc = result
            # Create new array
        else:
            lc = np.vstack((lc, result))  # Append to array
    # Writing lightcurve data to file.
    np.savetxt('cal/lc.dat', lc, fmt='%10.6f')
    time = lc[:, 0]
    comp1 = lc[:, 3]
    comp1err = lc[:, 4]
    comp2 = lc[:, 5]
    comp2err = lc[:, 6]
    comp3 = lc[:, 7]
    comp3err = lc[:, 8]
    comp4 = lc[:, 9]
    comp4err = lc[:, 10]
    comp5 = lc[:, 11]
    comp5err = lc[:, 12]
    comp6 = lc[:, 13]
    comp6err = lc[:, 14]
    var = lc[:, 1]
    varerr = lc[:, 2]

    comp = (comp1+comp2+comp3+comp4+comp5+comp6)/6.  # Mean of comp stars
    comp_err = 1.0857*((comp1err+comp2err+comp3err+comp4err +
                        comp5err+comp6err) /
                       (comp1+comp2+comp3+comp4+comp5+comp6))
    var_mag_corr = flux2mag(var)-flux2mag(comp)  # Differential mag
    var_mag_err = 1.0857*(varerr/var)
    comps = [comp1, comp2, comp3, comp4, comp5, comp6]
    # Plotting the comp check plots.
    # If one of the comparison stars is not ~constant we need to select an
    # other one insted of that. We plot the median and write the standard
    # deviation of the points to the title. Smaller std is (usually) better.
    for i in range(len(comps)):
        if i < 5:
            plt.plot(time-2450000,
                     flux2mag(comps[i])-flux2mag(comps[i+1]), '.')
            mean, median, std = sigma_clipped_stats(flux2mag(comps[i]) -
                                                    flux2mag(comps[i+1]),
                                                    sigma=3.0)
            plt.hlines(median, np.min(time-2450000), np.max(time-2450000))
            label = f"comp {i} - comp {i+1}"
            fname = f"compcheck{i}-{i+1}"
        else:
            plt.plot(time-2450000, flux2mag(comps[i])-flux2mag(comps[0]), '.')
            mean, median, std = sigma_clipped_stats(flux2mag(comps[i]) -
                                                    flux2mag(comps[0]),
                                                    sigma=3.0)
            plt.hlines(median, np.min(time-2450000), np.max(time-2450000))
            label = f"comp {i} - comp 0"
            fname = f"compcheck{i}-{0}"
        # format %2.4
        print(f"Standard deviation of points: {label} : {std:.4f}")
        plt.gca().invert_yaxis()
        plt.xlabel('JD')
        plt.title("std: "+str(f"{std:.5f}"))
        plt.ylabel(label)
        plt.savefig(f"cal/{fname}", dpi=150)
        plt.show()

    # Plotting the differential lightcurve of the target. We used an artifical
    # comparison star made from the 6 selected comparison star's flux.
    plt.plot(time-2450000, var_mag_corr, '.')
    plt.errorbar(time-2450000, var_mag_corr,
                 yerr=np.sqrt(var_mag_err**2 + comp_err**2), fmt='.')
    plt.gca().invert_yaxis()
    plt.xlabel('JD')
    plt.ylabel('Differential Instrumental Magnitude')
    plt.title('XX Cyg')
    plt.savefig('cal/XX_Cyg_lc.png', dpi=150)
    plt.show()
