from __future__ import division
from mpl_toolkits.mplot3d import Axes3D
import PIL
import matplotlib.pylab as plt
import matplotlib.colors as clr
import matplotlib.animation as ani
import numpy as np

################################################################################
__author__ = 'Aidan Shafer'
__date__ = '4/22/2017'
'''
Remake of the map module that just had to fucking delete itself after the most
bullshit 420 of my life. Hopefully second times the charm and not once more.
Fuck you AltLatLong_Results.py, fuck you hard.

Includes the worlds first free watch-the-world-drown map. It was truly an honor
to help simulate the demise of mankind. Enjoy!
'''
################################################################################

#Colormap for land, only custom one so-far (Not counting the beautiful maps which are now lost to the abyss)
cdict1 = {'red':   ((0.0, 0.5, 0.5),
                   (0.1, 0.0, 0.0),
                   (0.54, 0.0, 0.0),
                   (0.57, 1.0, 1.0),
                   (0.68, 1.0, 1.0),
                   (0.73, 0.3, 0.3),
                   (0.85, 1.0, 1.0),
                   (1.0, 0.3, 0.3)),

         'green': ((0.0, 0.0, 0.0),
                   (0.35, 0.0, 0.0),
                   (0.5, 1.0, 0.2),
                   (0.55, 1.0, 1.0),
                   (0.57, 1.0, 1.0),
                   (0.65, 0.0, 0.0),
                   (0.73, 0.0, 0.0),
                   (0.8, 1.0, 1.0),
                   (0.85, 1.0, 1.0),
                   (1.0, 0.3, 0.3)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.1, 0.1, 0.1),
                   (0.4, 1.0, 1.0),
                   (0.5, 1.0, 0.0),
                   (0.64, 0.0, 0.0),
                   (0.68, 1.0, 1.0),
                   (0.85, 1.0, 1.0),
                   (1.0, 0.5, 0.5))
        }
# JK I borrowed this one from AltLatLong sorta
cdict2 = {'red':   ((0.0, 0.0, 0.0),
                   (0.7, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.2),
                   (0.7, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (0.7, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }
# I had to try another one, this time for Mars (ooOOOOOOOOOooooooo!!)
cdict3 = {'red':   ((0.0, 0.0, 0.0),
                   (0.125, 0.7, 0.7),
                   (0.25, 0.5, 0.5),
                   (0.35, 0.7, 0.7),
                   (0.45, 0.9, 0.9),
                   (0.5, 0.7, 0.7),
                   (0.75, 0.7, 0.7),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.125, 0.0, 0.0),
                   (0.25, 0.25, 0.25),
                   (0.35, 0.35, 0.35),
                   (0.45, 0.9, 0.9),
                   (0.5, 0.6, 0.6),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (0.4, 0.0, 0.0),
                   (0.5, 0.6, 0.6),
                   (0.65, 1.0, 1.0),
                   (1.0, 1.0, 1.0))
        }
# A little sum'n sum'n for my hood minot. I'm on a roll now.
cdict4 = {'red':   ((0.0, 0.0, 0.0),
                   (0.7, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (0.5, 1.0, 0.2),
                   (0.7, 1.0, 1.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.1),
                   (0.5, 1.0, 0.0),
                   (0.7, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }

# Create the colormap using the dictionary above, and register it so it can be referenced later.
plt.register_cmap( cmap=clr.LinearSegmentedColormap('Map', cdict1) )
plt.register_cmap( cmap=clr.LinearSegmentedColormap('LandSea', cdict2) )
plt.register_cmap( cmap=clr.LinearSegmentedColormap('Martia', cdict3) ) # Not bad for earth's oceans.

#-- Honorable Mentions (native to matplotlib)
# gray
# copper
# seismic
#-- Slightly less honorable mentions
# gist_earth
# cubehelix
# nipy_spectral


################################################################################

class Map(object):
    """
    Class for Maps. Requires either a 2-D numpy array, something which can be
    converted into a 2-D numpy array, or a .npy datafile of a saved 2-D array.
    TopRight_BottomLeft is the Top Right Corner and Bottom Left Corner in
    latitude and longitude (used in bottom left corner of imshow to show
    coordinates, optional). extrema is the bottom and top extrema of elevation
    shown in imshow, warning: default colormap assumes midpoint is sea-level,
    and odd results may arise if not accounted for.

    show() shows the map, and globalwarming() animates the rise of sea level.

    Maps can be added vertically or horizontally with other maps. Can also be
    added to an integer to offset an entire map.

    """
    def __init__(self, datafile, TopRight_BottomLeft=None, extrema=(-8000, 8000), cmap='Map'):
        if isinstance(datafile, str):
            if datafile[:2] != 'C:':
                datafile = "C:/Users/shafe/Desktop/Stuff/Elevation/{} Elevation.npy".format(datafile) # If you're having trouble running the program and retrieving data, it may have something to do with this (aka the datapath hardcoded in).
            self.data = np.load(datafile)
        elif isinstance(datafile, np.ndarray):
            self.data = datafile
        elif isinstance(datafile, list) or isinstance(datafile, tuple):
            self.data = np.array(datafile)
        else:
            raise TypeError("Invalid type {} for map array".format( type(datafile) ))
        if len(np.shape(self.data)) != 2:
            raise TypeError("Invalid shape for map array, input must be 2-Dimensional")

        if not extrema:
            #self.extrema = self.get_stats()
            self.extrema = (self.min(), self.max())
        else:
            self.extrema = extrema

        self.pts = TopRight_BottomLeft
        self.cmap = cmap

#######################*********** PROJECTIONS **********#######################
    def CylindricalProjection(self, scale=1):
        if not self.pts:
            raise ValueError('Latitude and Longitude are needed for a Cylindrical projection.')
        # The x step is left the same to simplify computation and preserve accuracy
        # Both must be in radians for the functions ahead (rather than latitude and longitude in degrees)
        pts = np.deg2rad(self.pts)
        xstep = scale * (pts[1][1] - pts[0][1]) / (self.data.shape[1] - 1)

        # The range of y_values for the map, made so the first and last indices are used with even increments between
        yrange = np.arange(np.tan(pts[1][0]), np.tan(pts[0][0]), xstep)
        # The range of y values in terms of longitude
        theta_range = np.arctan(yrange)
        # Top of the map, used as a reference in array
        top_theta = pts[1][0]

        # For each x, y value on the map, calculate the elevation assuming slope between two points is roughly linear.
        return np.array([ (1 - theta%xstep) * self.data[ int((top_theta - theta) * scale / xstep) - 1, ::scale ] +
                          (theta%xstep) * self.data[ int((top_theta - theta) * scale / xstep), ::scale ] for theta in theta_range[::-1] ])

    def KavrayskiyProjection(self, scale=1):
        # Technically this is the Wagner IV projection. I tried to do Kavrayskiy but when I was simplifying it, I just made Wagner IV by accident.
        # Unfortunatly for Wagner IV, I think his name is stupid so I'm just gonna stick with Kavrayskiy
        if not self.pts:
            raise ValueError('Latitude and Longitude are needed for a Kavrayskiy projection.')
        # The x step is left the same to simplify computation and preserve accuracy
        # Both must be in radians for the functions ahead (rather than latitude and longitude in degrees)
        pts = self.pts
        shape = self.data.shape
        ystep = (pts[1][0] - pts[0][0]) / (shape[0] - 1)
        #ystep = (pts[1][1] - pts[0][1]) / (shape[1] - 1)

        x_range = (np.arange(0, shape[1], scale) - shape[1]/2)
        y_range = np.arange(0, shape[0], scale)

        p1 = x_range[0]
        p2 = x_range[-1]

        newdata = list()
        for y in y_range: # This is horribly innefficcient (calculating each point on the map individually), but I don't know a way around it
            factor = np.sqrt( 1 - 3*((pts[1][0] + (y - shape[0])*ystep) / 180)**2 )
            row = list()
            for x in x_range:
                phi = x / factor
                if p1 <= phi  <= p2:
                    row.append((phi%1) * self.data[y, int(phi-p1)]  +  (1-phi%1) * self.data[y, int(phi-p1)+1])
                else:
                    row.append(float('inf'))
            newdata.append(row)

        return np.array(newdata)

    def MercatorProjection(self, scale=1):
        if not self.pts:
            raise ValueError('Latitude and Longitude are needed for a Mercator projection.')
        # The x step is left the same to simplify computation and preserve accuracy
        # Both must be in radians for the functions ahead (rather than latitude and longitude in degrees)
        pts = np.deg2rad(self.pts)
        xstep = scale * (pts[1][1] - pts[0][1]) / (self.data.shape[1] - 1)

        # The range of y_values for the map, made so the first and last indices are used with even increments between
        yrange = np.arange(np.log(np.tan(pts[1][0]/2 + np.pi/4)), np.log(np.tan(pts[0][0]/2 + np.pi/4)), xstep)
        # The range of y values in terms of longitude
        theta_range = 2 * ( np.arctan(np.exp(yrange)) - np.pi/4 )
        # Top of the map, used as a reference in array
        top_theta = pts[0][0]

        # For each x, y value on the map, calculate the elevation assuming slope between two points is roughly linear.
        return np.array([ (1 - theta%xstep) * self.data[ int((top_theta - theta) * scale / xstep) - 1, ::scale ] +
                          (theta%xstep) * self.data[ int((top_theta - theta) * scale / xstep), ::scale ] for theta in theta_range[::-1] ])


#######################***********  SHOW MAP  **********########################
    def show(self, offset=0, extrema=[], cmap=None, scale=1, view='Default', norm=None, coord_prec=2):
        # Even if you don't pass view='Default', if the string isn't one of these, it's 'default'
        if view == 'Mercator':
            data = self.MercatorProjection(scale)
        elif view == 'Kavrayskiy':
            data = self.KavrayskiyProjection(scale)
        elif view == 'Cylindrical':
            data = self.CylindricalProjection(scale)
        else:
            data = self.data[::scale, ::scale]

        if not cmap:
            cmap = self.cmap

        if not extrema:
            extrema = self.extrema

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.format_coord = lambda x, y: self.format_coord(x, y, data, view, coord_prec=coord_prec) # For some reason I still need this because it all goes to shit once I zoom.

        mouse_move_patch = lambda event: self.mouse_move(fig.canvas.toolbar, event, data, view, coord_prec=coord_prec)        # This gets rid of that gay little braket if you put in a lot of work
        fig.canvas.toolbar._idDrag = fig.canvas.mpl_connect('motion_notify_event', mouse_move_patch)                          # But not perminantly of course 'cause that would be too easy, wouldn't it?

        plt.imshow(data - offset, vmin=extrema[0], vmax=extrema[1], cmap=cmap, norm=norm)
        plt.colorbar()
        plt.show()

    def globalwarming(self, start=0, stop=70, step=1, scale=1, interval=50, extrema=None, cmap=None, pause=500, view='Default', norm=None, coord_prec=2):
        def updatefig(i):
            # Must be defined here, or at least I'm too lazy not to...
            if i <= stop:
                image.set_array(data - i)
                plt.title("{:.1f} meters of sea level rise.".format(i))
            return image,

        if not cmap:
            cmap = self.cmap

        if not extrema:
            extrema = self.extrema

        if view == 'Mercator':
            data = self.MercatorProjection(scale)
        elif view == 'Kavrayskiy':
            data = self.KavrayskiyProjection(scale)
        elif view == 'Cylindrical':
            data = self.CylindricalProjection(scale)
        else:
            data = self.data[::scale, ::scale]

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.format_coord = lambda x, y: self.format_coord(x, y, data, view, coord_prec=coord_prec) # For some reason I still need this because it all goes to shit once I zoom.

        mouse_move_patch = lambda event: self.mouse_move(fig.canvas.toolbar, event, data, view, coord_prec=coord_prec)        # This gets rid of that gay little braket if you put in a lot of work
        fig.canvas.toolbar._idDrag = fig.canvas.mpl_connect('motion_notify_event', mouse_move_patch)

        image = ax.imshow(data, vmin=extrema[0], vmax=extrema[1], cmap=cmap, norm=norm, animated=True)
        anime = ani.FuncAnimation(fig, updatefig, frames=np.arange(start, stop + pause/interval, step), interval=interval)

        fig.colorbar(image)
        plt.show()

#######################************* HELPERS ************#######################
    def mouse_move(self, imager, event, data, view, coord_prec=2): # This is just way too complicated. I don't want to get into why it's here. It's just those stupid brackets annoy the piss outta me.
        if event.inaxes and event.inaxes.get_navigate():
            imager.set_message(self.format_coord(event.xdata, event.ydata, data, view, coord_prec=coord_prec))

    def format_coord(self, x, y, data, view, coord_prec=2):
        # This is defined in self.show() so view data is shown (not map data)

        shape = data.shape      # data, not self.data (takes into account formatting)

        if self.pts:
            if view == 'Mercator':
                pts = np.deg2rad(self.pts)
                lat = np.rad2deg(2*np.arctan(np.exp(  np.log(np.tan(pts[0][0]/2 + np.pi/4)) - ( (pts[1][1] - pts[0][1]) * y / shape[1] )  )) - np.pi/2) # Have fun figurung this out lol
                lon = self.pts[0][1] + (self.pts[1][1] - self.pts[0][1]) * x / shape[1]     # Mercator don't fuck wit x
            elif view == 'Cylindrical':
                pts = np.deg2rad(self.pts)
                lat = np.rad2deg(np.arctan(  np.tan(pts[0][0]) - ( (pts[1][1] - pts[0][1]) * y / shape[1] )  )) # Have fun figurung this out lol
                lon = self.pts[0][1] + (self.pts[1][1] - self.pts[0][1]) * x / shape[1]     # Cylindrical don't fuck wit x
            elif view == 'Kavrayskiy':
                lat = self.pts[0][0] + (self.pts[1][0] - self.pts[0][0]) * y / shape[0]     # But Kavrayskiy don't fuck with y, cuz like, y not?
                lon = (self.pts[0][1] + (self.pts[1][1] - self.pts[0][1]) * x / shape[1]) / np.sqrt(1 - 3*(lat/180)**2)
            else:
                lat = self.pts[0][0] + (self.pts[1][0] - self.pts[0][0]) * y / shape[0]     # Bottom of the map plus the y-value times step size
                lon = self.pts[0][1] + (self.pts[1][1] - self.pts[0][1]) * x / shape[1]     # Dittoish

            if not ((self.pts[0][1] <= lon <= self.pts[1][1]) and (self.pts[1][0] <= lat <= self.pts[0][0])):    # I got sick of latitude and longitude for places off the map.
                lat = float('nan')
                lon = float('nan')

        else:
            lat = float('nan')  # 'Default' Latitude
            lon = float('nan')

        if 0<=y<=shape[0] and 0<=x<=shape[1]:
            alt = data[int(round(y)), int(round(x))]
        else:
            alt = float('nan')  # 'Default' elevation is nonexistant

        return 'Lat={:.{prec}f}, Lon={:.{prec}f}:  Elevation={:.1f} m        '.format(lat, lon, alt, prec=coord_prec)

    def get_stats(self, mode=0): # Depreciated, I would have removed it, but my luck something would actually be using it
        stdev = np.std(self.data)
        return ( -3 * stdev, 3 * stdev )

    def getdata(self): # I know I should, but I never actually use these getters. In fact I think this'll be my only one.
        return self.data

    def min(self):
        return np.amin(self.data)

    def max(self):
        return np.amax(self.data)

    def shape(self):
        return self.data.shape


######################************* OPERATORS ************######################
    def __repr__(self):
        return "Map(self, {}, TopRight_BottomLeft={}, extrema={}, cmap={})".format(self.data, self.pts, self.extrema, self.cmap)

    def __str__(self):
        #return str(self.data)

        return_string = ''
        if len(self.data) < 11:
            datalist = self.data
        else:
            if len(self.data[0]) > 1:
                datalist = list(self.data[:4, :])  +  [ ['.']+['' for val in self.data[0, 1:-1]]+['     .'] for ct in range(3) ]  +  list(self.data[-4:, :])
            else:
                datalist = list(self.data[:4, :])  +  [ ['.'] for ct in range(3) ]  +  list(self.data[-4:, :])
        for row in datalist:
            return_string += '|   '
            if len(row) < 11:
                for val in row:
                    return_string += '{:8.6}   '.format(val)
            else:
                return_string += '{:8.6}   {:8.6}   {:8.6}   {:8.6}    ...    {:8.6}   {:8.6}   {:8.6}   {:8.6}   '.format( *(list(row[:4]) + list(row[-4:])) )
            return_string += '|\n'
        return return_string

    def __add__(self, x):
        if isinstance(x, Map):
            if self.data.shape[1] == x.data.shape[1]:
                newdata = np.vstack((self.data, x.data))
            elif self.data.shape[0] == x.data.shape[0]:
                newdata = np.hstack((self.data, x.data))
            else:
                raise ValueError("Could not broadcast together shapes with {} and {}".format(self.data.shape, x.data.shape))

            if self.pts and x.pts:
                pts = (self.pts[0], x.pts[1])
            else:
                pts = None

            if self.cmap == x.cmap:
                cmap = self.cmap
            else:
                cmap = 'Map'

            extrema = (min(self.extrema[0], x.extrema[0]) , max(self.extrema[1], x.extrema[1]) )

            return Map(newdata, TopRight_BottomLeft=pts, extrema=extrema, cmap=cmap)

        else:
            self.data = self.data + x

    def __radd__(self, x):
        if isinstance(x, Map):
            if self.data.shape[0] == x.data.shape[0]:
                newdata = np.hstack((x.data, self.data))
            elif self.data.shape[1] == x.data.shape[1]:
                newdata = np.vstack((x.data, self.data))
            else:
                raise ValueError("Could not broadcast together shapes with {} and {}".format(self.data.shape, x.data.shape))

            if self.pts and x.pts:
                pts = (x.pts[0], self.pts[1])
            else:
                pts = None

            if self.cmap == x.cmap:
                cmap = self.cmap
            else:
                cmap = 'Map'

            extrema = (min(self.extrema[0], x.extrema[0]) , max(self.extrema[1], x.extrema[1]) )

            return Map(newdata, TopRight_BottomLeft=pts, extrema=extrema, cmap=cmap)

        else:
            self.data = self.data + x

    def __getitem__( self, key ) :
        # As you can see, this got way too fukin complicated
        newdata = self.data[key]
        if len(newdata.shape) == 2:
            pts = self.pts
            ystep = (pts[1][0] - pts[0][0]) / self.data.shape[0]
            xstep = (pts[1][1] - pts[0][1]) / self.data.shape[1]

            if key[0].start:
                ystart = key[0].start
            else:
                ystart = 0
            if key[1].start:
                xstart = key[1].start
            else:
                xstart = 0
            if key[0].stop:
                ystop = key[0].stop
            else:
                ystop = len(self.data)
            if key[1].stop:
                xstop = key[1].stop
            else:
                xstop = len(self.data[0])

            newpts = ((pts[0][0] + ystep*ystart, pts[0][1] + xstep*xstart), (pts[0][0] + ystep*ystop, pts[0][1] + xstep*xstop))
            return Map(newdata, TopRight_BottomLeft=newpts, extrema=self.extrema, cmap=self.cmap)
        else:
            return newdata


####################************ NORMALIZATION ************#####################
class SymPwrNorm(clr.Normalize):
    def __init__(self, vmin=None, vmax=None, power=1.0, midpoint=None, clip=False):
        clr.Normalize.__init__(self, vmin, vmax, clip)

        self.power = power
        self.midpoint = midpoint

    def __call__(self, value, clip=None):
        # Make mid the actual midpoint if one is not specified, if one is, values above and below are scaled linearly.
        if self.midpoint:
            mid = self.midpoint
        else:
            mid = (self.vmax + self.vmin) / 2
        # The x and y values for interpolation i.e. interpolate the y for the given x. (x and y must be the same size so indices match up)
        x, y = [self.vmin, mid, self.vmax], [-1, 0, 1]
        norm_array = np.interp(value, x, y)

        return np.ma.masked_array( ( (norm_array * abs(norm_array)**(self.power-1) ) + 1 ) / 2 )

        #x, y = [self.vmin, mid, self.vmax], [0, 0.5, 1]
        #return np.ma.masked_array( np.interp(value, x, y)**self.power )

        #norm_array = np.interp(value, x, y)
        #return np.ma.masked_array(  ( (norm_array-0.5) * abs(norm_array-0.5)**(self.power-1) ) + 0.5  )


################################################################################
#########################********     MAIN     ********#########################
################################################################################

#fargo = Map('Fargo', TopRight_BottomLeft=((47.33847, -97.029167), (46.78625, -96.616389)), extrema=(260,285), cmap='gray')
#minot = Map('Minot', TopRight_BottomLeft=((48.307667, -101.461111), (48.257539, -101.153111)), extrema=(460, 600), cmap='gray') # [ tiny map - ((48.285, -101.372), (48.19, -101.215)) ]

#world = Map('worldtest', TopRight_BottomLeft=((75, -180), (-75, 180)))                     # True elevation ranges from -10994 (Mariana Trench) to 8848 (Mt. Everest)
#mars = Map('Mars', TopRight_BottomLeft=((89, -180), (-89, 180)), extrema=(-8000, 20000))   # Not the most reliable data, real range is from -8200 to 21200
#moon = Map('Moon', TopRight_BottomLeft=((89, -180), (-89, 180)), extrema=(-9120, 10760))   # Aweful data, real range is from -9120 to 10761

#a = Map('Canada', TopRight_BottomLeft=((75, -129.99), (50, -50)))
b = Map('America', TopRight_BottomLeft=((49.99, -129.99), (24, -50)))
#c = Map('Caribbean', TopRight_BottomLeft=((23.99, -129.99), (14, -50)))
#d = Map('Alaska', TopRight_BottomLeft=((75, -194.99), (50, -130)))
#e = Map('Amazon', TopRight_BottomLeft=((13.99, -94.99), (-20, -30)))
#f = Map('South Cone', TopRight_BottomLeft=((-20.01, -94.99), (-60, -30)))
#g = Map('Scandanavia', TopRight_BottomLeft=((75, -29.99), (55, 60)))
h = Map('Europe', TopRight_BottomLeft=((54.99, -29.99), (35, 60)))
i = Map('Sahara', TopRight_BottomLeft=((34.99, -29.99), (0, 60)))
j = Map('Madagascar', TopRight_BottomLeft=((-0.01, -29.99), (-35, 60)))
#k = Map('Russia', TopRight_BottomLeft=((75, 60.01), (45, 165)))
#l = Map('China', TopRight_BottomLeft=((44.99, 60.01), (25, 165)))
#m = Map('Phillipeans', TopRight_BottomLeft=((24.99, 60.01), (0, 165)))
#n = Map('Indonesia', TopRight_BottomLeft=((-0.01, 90.01), (-25, 180)))
#o = Map('Australia', TopRight_BottomLeft=((-25.01, 90.01), (-50, 180)))

#p = Map('North Pacific', TopRight_BottomLeft=((49.99, -194.99), (30, -130)))
#q = Map('Hawaii', TopRight_BottomLeft=((29.99, -194.99), (14, -130)))
#r = Map('Atlantic', TopRight_BottomLeft=((75, -49.99), (14, -30)))
#s = Map('Cancer', TopRight_BottomLeft=((13.99, -194.99), (0, -95)))
#t = Map('Sopa', TopRight_BottomLeft=((-0.01, -179.99), (-60, -95)))
#u = Map('SubAfrica', TopRight_BottomLeft=((-35.01, -29.99), (-60, 60)))
#v = Map('Indian', TopRight_BottomLeft=((-0.01, 60.01), (-50, 90)))
#w = Map('Antpac', TopRight_BottomLeft=((-60.01, -179.99), (-75, -30)))
#x = Map('Antaf', TopRight_BottomLeft=((-60.01, -29.99), (-75, 60)))
#y = Map('Antind', TopRight_BottomLeft=((-50.01, 60.01), (-75, 180)))

# Not-z, much like Hitler's Germany and Trump's America lol


################################################################################
#print 'a'
if 0:
    skl = 5 # Scale. If this is too small (i.e. 1), it fills up my ram and the computer crashes. Even 5 makkes it run horrendously slowly
    world = (( ( (d+p+q)[::skl, 1500::skl] + (a+b+c)[::skl, ::skl] + r[::skl, ::skl] )  +  ( ((s[::skl, 1500::skl] + t[::skl, ::skl]) + (e+f)[::skl, ::skl]) + w[::skl, ::skl]) )  +
             ( (g+h+i)[::skl, ::skl] + (j+u+x)[::skl, ::skl] ) +
             ( ( (k+l+m)[::skl, ::skl] + ((d+p+q)[::skl, :1500:skl] + s[::skl, :1500:skl]) )  +  ( v[::skl, ::skl] + (n+o)[::skl, ::skl] + y[::skl, ::skl] ) ))
    #print world.pts, world.data.shape

    '''
    # Outdated, all of these

    #na = a + b + c
    #na_fill = ((d + fill_na1) + (a + b + c) + fill_na2)[::4, ::4]

    #sa = e + f
    #sa_fill = (fill_sa1 + (e + f))[::4, ::4]

    #euro = (h + g)[::4, ::4]
    #africa = i + j
    #afro_fill = (i + j + fill_afro)[::4, ::4]

    #new = (na_fill + sa_fill) + (euro + afro_fill)
    '''
#print 'b'
################################################################################


b.show(scale=1, view='Mercator', offset=0)#, extrema=(450, 650), cmap='gray')
#b.show(scale=1, view='Kavrayskiy')#, extrema=(-900, 900))
#(h+g + (k+l)[:4001, :]).show(scale=2)#, view='Mercator')#, extrema=[-2000, 2000])
#(h+i+j).show(scale=1, view='Mercator')#, extrema=[-2000, 2000])

#b.globalwarming(scale=2, view='Mercator', step=2)#, start=-100, stop=100, step=3)
#na.globalwarming(step=1.5, xslice=6, yslice=6)

#minot.show(cmap='gray', coord_prec=4, norm=SymPwrNorm(power=2), extrema=(460, 540))#vmin=440, vmax=600, midpoint=520))
#minot.show(cmap='gray', coord_prec=4)

#(world[::-1, world.data.shape[1]/2 - 500::-1] + world[::-1, :world.data.shape[1]/2 - 500:-1]).show(scale=1, offset=0)#, view='Mercator')#, cmap='Martia')
#world.show(scale=5, view='Kavrayskiy')#, cmap='copper')
#world.globalwarming(view='Mercator', start=-120, stop=60, step=10)

#mars.show(cmap='copper')#, view='Kavrayskiy')#scale=1, extrema=(-9000, 9000))#, cmap='gray')
#moon.show(cmap = 'copper')

#print minot.min(), minot.max(), minot.shape()
#print minot.data.shape

# If you fuck up and overwrite a datafile (*again), use the gray PNY USB with Kanji in the 2nd to front pocket of your backback to retrieve it.
# Hopefully it wasn't something you made since you last backed everything up...
#  #np.save("C:\Users\shafe\Desktop\Stuff\Elevation\Fargo Elevation.npy", new.data)


################################################################################
'''
# load bluemarble with PIL
#bm = PIL.Image.open('C:\Users\shafe\Desktop\Stuff\Elevation\earthmapsample.jpg')
bm = PIL.Image.open('C:\Users\shafe\Desktop\Stuff\Elevation\myworldimg0.jpeg')
#bm = PIL.Image.open('C:\Users\shafe\OneDrive\Pictures\Saved Pictures\Mars_HRSC_MOLA_BlendDEM_Global_200mp_1024.jpg')
# it's big, so I'll rescale it, convert to array, and divide by 256 to get RGB values that matplotlib accept
bm = np.array(bm.resize([int(d/2) for d in bm.size]))/256.

# coordinates of the image - don't know if this is entirely accurate, but probably close
lons = np.linspace(-180, 180, bm.shape[1]) * np.pi/180
lats = np.linspace(-90, 90, bm.shape[0])[::-1] * np.pi/180

# repeat code from one of the examples linked to in the question, except for specifying facecolors:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

x = np.outer(np.cos(lons), np.cos(lats)).T
y = np.outer(np.sin(lons), np.cos(lats)).T
z = np.outer(np.ones(np.size(lons)), np.sin(lats)).T
ax.plot_surface(x, y, z, rstride=4, cstride=4, facecolors = bm)

plt.show()
'''
