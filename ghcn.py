import numpy as np
import os
import gzip

"""
This script takes the daily data from stations in the GHC network for
a given month and a given element type, and constructs the mean
profile, separately for stations in the northern and southern
hemispheres.

The data are available here:

ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd_gsn.tar.gz

ftp://ftp.ncdc.noaa.gov/pub/data/ghcn/daily/ghcnd-stations.txt

File format information is available here:

http://www1.ncdc.noaa.gov/pub/data/ghcn/daily/readme.txt
"""

# Path to data
bpath = "/nfs/kshedden/GHCN"

# Get the latitude values for each station
fname = os.path.join(bpath, "ghcnd-stations.txt")
lat = {}
fid = open(fname)
for line in fid:
    idx = line[0:11]
    x = float(line[11:20])
    lat[idx] = x
fid.close()

files = os.listdir(os.path.join(bpath, "ghcnd_gsn"))

# Column specs for the daily values in each row
colspecs = []
ii = 21
for k in range(31):
    v = (ii, ii+5)
    colspecs.append(v)
    ii += 8

# Extract only this measurement type
target_element = "TMAX"

# Extract only this month
target_month = "01"

# Convert strings to numeric
def my_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

south = np.zeros(31)
south_n = 0
north = np.zeros(31)
north_n = 0

for fname in files:

    fname = os.path.join(bpath, "ghcnd_gsn", fname)
    gid = gzip.open(fname, "rt")

    for line in gid:

        if line[17:21] != target_element:
            continue

        idx = line[0:11]
        if idx not in lat:
            continue

        latitude = lat[idx]

        # Extract only one month
        month = line[15:17]
        if month != target_month:
            continue

        # Extract the value fields and convert to numeric
        v = [line[p[0]:p[1]] for p in colspecs]
        v = np.asarray(v)

        # Extract the numeric data
        nvals = [my_float(x) for x in v]
        nvals = np.asarray(nvals)
        nvals[nvals == -9999] = np.nan

        # The raw data are in 0.1 degree C, convert to C
        nvals /= 10

        assert(len(nvals) == 31)

        ii = np.isfinite(nvals)
        if latitude > 0:
            north_n += ii
            north[ii] += nvals[ii]
        else:
            south_n += ii
            south[ii] += nvals[ii]

# Convert sums to averages
north /= north_n
south /= south_n
