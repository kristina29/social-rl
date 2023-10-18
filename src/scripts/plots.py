import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

lat = [42.3, 40.86, 41.14, 42.44, 43.10, 43.14, 43.38, 44.86]
lon = [-74.37, -72.57, -73.91, -78.45, -75.33, -73.97, -76.31, -73.61]

fig, ax = plt.subplots(figsize=(12, 15))
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)
map = Basemap(projection='cyl', llcrnrlat=min(lat)-1, urcrnrlat=max(lat)+1,
              llcrnrlon=min(lon)-1.5, urcrnrlon=max(lon)+1, resolution='i')
map.drawcoastlines(linewidth=1)  # draws coastline
map.drawcountries(linewidth=1)  # draws countries
map.drawstates(linewidth=1)
map.fillcontinents(color="#eeeeee", lake_color='#DDEEFF')
map.drawmapboundary(fill_color='#DDEEFF', linewidth=1)
#lons, lats = np.meshgrid(lon, lat)  # 2D lat lon to plot contours
x, y = map(lon, lat)

#csf = map.contourf(x, y, data)  # filled contour
#map.colorbar(csf, "right", extend='both', size="3%", pad="1%")
map.scatter(x, y, s=150, c='#cc0000')

fig.savefig("locations.pdf")