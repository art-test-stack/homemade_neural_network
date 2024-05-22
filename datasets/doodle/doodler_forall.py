
#  A class for creating 2-dimensional binary images of simple shapes and figures, with the key feature that
# the sizes and locations of these images can vary (and be determined stochastically).  This allows the
# doodler to produce large data sets for Machine Learning.  The key interface function is 'gen_standard_cases', which
# students should be able to use without digging into many of the details of this file.  However, it is wise to
# experiment with different arguments to that function.  The 8 or 9 shapes that the doodler can produce are
# listed below in the _doodle_image_types_ variables.

# The doodler operates differently than most image generators.  For each image type, the doodler needs a
# predicate function that takes two arguments (row, column) and returns True or False, depending upon whether the
# pixel in that location should be on (i.e. be "foreground" as opposed to "background").  So to add a new image
# type, simply give it a name (e.g. xyz) and then add a method named "gen_xyz" which takes the same arguments
# as 'gen_ball', 'gen_frame', 'gen_polygon', etc. and ends with the call self.doodle(fill_predicate), which will
# then "paint" the canvas with 1's and 0's based on the fill_predicate defined in gen_xyz.

# When calling 'gen_standard_cases' from the python command line, if the "show" option is True, then all
# cases will be displayed.  This will produce many windows, one for each image. These windows will disappear when
# you finish your python session, or you can click on them individually to delete, resize, move or save to
# file.  The Doodler does not provide any simpler means of saving all the images to file.
# However, it can save a list of cases to file using "dump_doodle_cases"; to load cases,
# use "load_doodle_cases".  Once loaded, the cases can be displayed via "show_doodle_cases".  Students can
# make simple modifications to show_doodle_cases in order to only display some of the cases in a collection.
# This can be helpful during debugging and during a demo session, when the instructor only needs to see
# 10 or so cases from a collection.

# In general, students will need to experiment with this code to understand it fully.  It is only sparsely
# commented, although the main interface function, gen_standard_cases, has a significant description.

import random
import math
import numpy as np
import pickle as PK
from inspect import isfunction
import matplotlib.pyplot as PLT
import matplotlib.cm as CMAP  # Color map stuff

# ********************** End of section to remove *************

_doodle_image_types_ = ['ball','ring','frame','box','flower','bar','polygon','triangle','spiral']

class Doodler:

    def __init__(self,rows=10,cols=10,bkg=0,fog=1,nbias=0.0,wrange=[0.1,0.6],hrange=[0.1,0.6],
                 dbias=0.5,cent=False,pet=6,multi=False,gap=1,poly=(3,4,5),bmar=1,spiro=3):
        self.canvas_rows = rows; self.canvas_cols = cols
        self.max_width = cols-1; self.max_height = rows-1
        self.width_range = wrange; self.height_range = hrange  # Ranges for height and width of doodles
        self.background = bkg
        self.foreground = fog
        self.noise_bias = nbias
        self.dot_bias = dbias
        self.centered = cent
        self.num_petals = pet
        self.spiral_whorls = spiro  # Ideal number of whorls in a spiral
        self.multi = multi # True if multiple images per canvas
        self.figure_gap = gap if multi else 0  # Mandatory minimum gap between figures
        self.polygon_sizes = poly # possible sizes (# sides = # angles) for random polygons
        self.bar_margin = bmar # Thickness factor for bars and polygons
        self.gen_canvas_center()
        self.current_canvas = None
        self.gen_cell_params()
        if self.multi: self.gen_cover_map()

    # Returned point = (row,col)
    def gen_random_center(self,width,height):
        w2 = math.floor(width/2); h2 = math.floor(height/2)
        col_range = [w2, self.max_width - w2 ]
        row_range = [h2, self.max_height - h2 ]
        return (random.randint(*row_range),random.randint(*col_range))

    # For some calculations, the real-valued width and height of a cell are needed.
    def gen_cell_params(self):
        self.cell_width = 1 / self.canvas_cols
        self.cell_height = 1 / self.canvas_rows
        self.cell_diagonal = math.sqrt(self.cell_width**2 + self.cell_height**2)

    # Note: This converts row,col into an (x,y) real-valued point
    def get_cell_center(self,row,col):
        return ((col+0.5)*self.cell_width, (row+0.5)*self.cell_height)

    # Create the groups for filled and unfilled images.  These will be used to create "fractional" canvases, where
    # a certain fraction of the images will be filled.

    def gen_image_groups(self,current_images):
        filled = ['ball','box']; unfilled = ['ring','frame','polygon']
        self.filled_images = [f for f in filled if f in current_images]
        self.unfilled_images = [uf for uf in unfilled if uf in current_images]

    def gen_fractional_image_set(self,size,fillfrac=0.5):
        filled_count = size if fillfrac == 'all' else round(size * fillfrac)
        filled_images = np.random.choice(self.filled_images,filled_count,replace=True)
        unfilled_images = np.random.choice(self.unfilled_images,size-filled_count,replace=True)
        return list(filled_images) + list(unfilled_images), filled_count

    # NOTE: the r
    def gen_figure_types(self,all_types,figcount,fillfrac):
        if fillfrac is not None:
            return self.gen_fractional_image_set(figcount,fillfrac=fillfrac)
        else:
            return np.random.choice(all_types,figcount,replace=True), figcount

    def gen_canvas_center(self):
        self.canvas_center = (round(self.max_height/2),round(self.max_width/2))

    # Cover map = 2d array indicating which cells are covered (and thus not available).  This is important
    # anytime multiple shapes are plotted in the same image, since we do not want overlap.
    def gen_cover_map(self):
        self.cover_map = np.zeros((self.canvas_rows,self.canvas_cols))

    def cover_cell(self,r,c):   self.cover_map[r,c] = 1
    def uncover_cell(self,r,c): self.cover_map[r,c] = 0
    def cell_covered(self,r,c): return self.cover_map[r,c] == 1

    def clear_cover_map(self):
        for r in range(self.canvas_rows):
            for c in range(self.canvas_cols):
                self.uncover_cell(r,c)

    # Update the cover map to include the new region subtended by the new image.
    def add_coverage(self,center,half_width,half_height):
        for dc in range(-half_width, half_width + 1):
            for dr in range(-half_height, half_height+1):
                r = center[0] + dr; c = center[1] + dc
                self.cover_cell(r,c)

    def check_availability(self,center,half_width,half_height):
        hw = half_width + self.figure_gap; hh = half_height + self.figure_gap
        for dc in range(-hw, hw + 1):
            for dr in range(-hh, hh+1):
                r = center[0] + dr; c = center[1] + dc
                if not (0 <= c < self.canvas_cols) or not (0 <= r < self.canvas_rows) or self.cell_covered(r,c):
                    return False
        return True  # All cells in the area are available

    def gen_random_width(self,minfrac=0,maxfrac=1):
        range = [max(1,math.floor(minfrac*self.max_width)), math.ceil(maxfrac*self.max_width)]
        return random.randint(*range)

    def gen_random_height(self,minfrac=0,maxfrac=1):
        range = [max(1,math.floor(minfrac*self.max_height)), math.ceil(maxfrac*self.max_height)]
        return random.randint(*range)

    # Boxwidth and boxheight are dims (in # of cells) of the object's bounding box, not of the canvas itself.
    # The returned radius is in the range (0,1)
    def get_normalized_radius(self,boxwidth,boxheight):
        return min(boxwidth * self.cell_width, boxheight * self.cell_height)

    # This returns a center plus two "safe" lengths: a half-width and a half-height
    def gen_image_params(self,wrange=[0,1],hrange=[0,1]):
        image_fits = False
        while not(image_fits):
            w = self.gen_random_width(*wrange); h = self.gen_random_height(*hrange)
            c = self.canvas_center if (self.centered and not(self.multi)) else self.gen_random_center(w,h)
            w2 = max(1,math.floor(w/2)); h2 = max(1,math.floor(h/2))
            image_fits = self.check_availability(c,w2,h2) if self.multi else True
        if self.multi: self.add_coverage(c,w2,h2)
        return c, w2, h2

    def gen_canvas(self,background=0):
        r = self.canvas_rows; c = self.canvas_cols
        return np.array([background] * (r*c)).reshape((r,c))

    # The master doodle method.  All the info is in the fill predicate, which must take the row-col coords
    # of a point and determine whether to fill it or not.
    def doodle(self,fill_pred):
        cas = self.current_canvas if (self.current_canvas is not None) else self.gen_canvas(background=self.background)
        for r in range(self.canvas_rows):
            for c in range(self.canvas_cols):
                cas[r,c] = self.foreground if fill_pred(r,c) else self.background
        return cas

    def gen_frame(self,center,w2,h2):
        w2s = w2*.667; h2s = h2*.667
        def fp(r,c):
            dr = abs(r-center[0]); dc = abs(c-center[1])
            return (h2s <= dr <= h2 and dc <= w2) or (dr <= h2 and w2s <= dc <= w2)
        return self.doodle(fill_pred=fp)

    def gen_box(self, center,w2,h2):
        def fp(r, c):
            dr = abs(r - center[0]) ; dc = abs(c - center[1])
            return (dr <= h2 and dc <= w2)
        return self.doodle(fill_pred=fp)

    def gen_ball(self,center,w2,h2):
        radius = min(w2,h2)
        def fp(r,c):
            return radius >= math.sqrt((r-center[0])**2 + (c-center[1])**2)
        return self.doodle(fill_pred=fp)

    def gen_triangle(self,center,w2,h2):
        return self.gen_polygon(center,w2,h2,n=3,regular=True)

    # This fills a cell iff its center is within a half cell diagonal from the actual real-valued line.  Note:
    # the point 'center' is in (row, col) format, whereas the real-valued points p1,p2 and cp are all in (x,y) format.
    # The angle of the line is derived from the half-width (w2) and half-height(h2).  These form a box, and the bar
    # is one of the two diagonals of that box (chosen randomly).
    def gen_bar(self,center,w2,h2):
        factor = (-1)**random.randint(0,1)
        c1 = round(center[1] - w2); c2 = round(center[1] + w2)
        r1 = round(center[0] + factor*h2); r2 = round(center[0] - factor*h2)
        fp = self.gen_bar_predicate(r1,c1,r2,c2)
        return self.doodle(fill_pred=fp)

    def gen_bar_predicate(self,r1,c1,r2,c2):

        def fp_vertical(r, c):  return abs(c-c1) <= self.bar_margin and min(r1, r2) <= r <= max(r1, r2)
        def fp_horizontal(r, c):    return abs(r-r1) <= self.bar_margin and min(c1, c2) <= c <= max(c1, c2)

        def fp_sloped(r,c):
            # if abs(r-center[0]) <= h2 and abs(c-center[1]) <= w2:
            if min(r1,r2) <= r <= max(r1,r2) and min(c1,c2) <= c <= max(c1,c2):
                cp =self.get_cell_center(r,c)
                dist = abs(a*cp[0]+b*cp[1] +d) / denom
                return dist <= thresh
            else: return False



        if r1 == r2: return fp_horizontal
        elif c1 == c2: return fp_vertical
        else:
            p1 = self.get_cell_center(r1,c1); p2 = self.get_cell_center(r2,c2)  # real-valued coords (x,y)
            dx = p2[0] - p1[0]
            # canvas_rows * 10 effectively makes the slope infinite => a vertical line.
            slope = (p2[1] - p1[1])/dx   # dx will not be zero, since that's covered by the c1 == c2 case
            thresh = self.bar_margin * self.cell_diagonal / 1.414
            a = slope; b = -1; d = p1[1]-slope*p1[0]  # params for the shortest-dist to point equation.
            denom = math.sqrt(a**2 + b**2)
            return fp_sloped


    def gen_flower(self,center,w2,h2):
        radius = min(w2,h2)
        def fp(r,c):
            dr = abs(r - center[0]); dc = abs(c - center[1])
            dist = math.sqrt(dr**2 + dc**2)
            if dist <= 0: return True
            else:
                theta = math.acos(dc/dist)
                return dist <= radius*math.cos(self.num_petals*theta)
        return self.doodle(fill_pred=fp)

    def gen_ring(self,center,w2,h2):
        rad = min(w2, h2) ; rad2 = rad/2
        def fp(r,c):
            return rad >= math.sqrt((r - center[0]) ** 2 + (c - center[1]) ** 2) >= rad2
        return self.doodle(fill_pred=fp)

    def gen_spiral(self,center,w2,h2):
        radius = self.get_normalized_radius(w2,h2)
        delta = radius / self.spiral_whorls  # ideal spacing between whorls
        cp0 = self.get_cell_center(*center)  # gets the (x,y) coords, both in range [0,1]
        def fp(r,c):
            cp1 = self.get_cell_center(r,c)
            d = distance(cp1,cp0)
            if d <= radius:
                theta = angle_2d(cp1,cp0)
                dy = cp1[1] - cp0[1]
                return math.sin((d/delta)*2*math.pi)*math.sin(theta) > 0
            else:   return False
        return self.doodle(fill_pred=fp)

    # This puts random dots inside a rectangular region of the canvas
    def gen_dots(self,center,w2,h2):
        def fp(r,c):
            dr = abs(r - center[0]); dc = abs(c - center[1])
            return (dr <= h2 and dc <= w2 and random.uniform(0,1) <= self.dot_bias)
        return self.doodle(fill_pred=fp)

    def gen_random_polygon_size(self):  return random.choice(self.polygon_sizes)
    def get_polygon_index(self,size):   return self.polygon_sizes.index(size)

    # This creates a random convex polygon, which is typically NOT a regular polygon.  n = # sides/angles.
    def gen_polygon(self,center,w2,h2,n=5,regular=False):
        # Points from gen_convex_polygon are in (x,y) format with x,y in range (0,1), while standard doodler pts
        # are in (row,col) format.
        def scale_pt(pt):
            return [round(rmin + dr*(1 - pt[1])), round(cmin + dc*pt[0])]

        pts = gen_convex_polygon(n,xr=(0,1),yr=(0,1),plot=False,vect=False,regular=regular)  # all coords in range [0,1]
        rmin = center[0] - h2; cmin = center[1] - w2; dr = 2*h2+1; dc = 2*w2+1
        pts2 = [scale_pt(p) for p in pts]  # The pts should now be in doodler coordinates.
        pairs = [[pts2[i],pts2[i+1]] for i in range(len(pts2)-1)]  # Gen pairs of adjacent points
        pairs.append([pts2[-1],pts2[0]])
        # Generate a different predicate for each line (bar) in the polygon.
        preds = [self.gen_bar_predicate(*p[0],*p[1]) for p in pairs]

        def fp(r,c):
            return exists(preds, (lambda pred: pred(r,c)))

        return self.doodle(fill_pred=fp)
    # type = (ball, box,frame, ring, dots, polygon, bar).  If poly is not None, then we're doing polygon
    # classification, so ALL figures will be the same size polygon.

    def gen_image(self,types=['ball'],wr=None,hr=None,figcount=1,fillfrac=None,poly=None):
        if self.multi: self.clear_cover_map()
        wrange = wr if wr != None else self.width_range; hrange = hr if hr != None else self.height_range
        self.current_canvas = self.gen_canvas()
        figtypes, count = self.gen_figure_types(types,figcount,fillfrac=fillfrac)
        for ftype in figtypes:
            method = "Doodler.gen_"+ftype
            center, w2, h2 = self.gen_image_params(wrange, hrange)
            if ftype == 'polygon':
                size = poly if poly else self.gen_random_polygon_size()
                self.current_canvas = eval(method)(self,center,w2,h2,n=size)
            else:
                self.current_canvas = eval(method)(self,center,w2,h2)
        return self.consider_noise(), count

    def consider_noise(self):
        canvas = self.current_canvas
        def fp(r,c):
            base_truth = (canvas[r,c] == self.foreground)
            return not(base_truth) if random.uniform(0,1) <= self.noise_bias else base_truth
        return self.doodle(fill_pred=fp) if self.noise_bias > 0 else canvas

    # When poly is a number, then we're doing polygon-classification, so all figures will be the same type of
    # polygon (i.e. have same # of sides/angles).  Fcount = True => the label is the NUMBER of filled shapes, NOT
    # the fraction.
    def gen_random_case(self,image_types=_doodle_image_types_,wr=None,hr=None,flat=True,label=None,
                        figcount=1,fillfrac=None,fcount=False, tlen=None,poly=None):
        itypes = [random.choice(image_types)] if figcount == 1 else image_types
        a, count = self.gen_image(types=itypes,wr=wr,hr=hr,figcount=figcount,fillfrac=fillfrac,poly=poly)
        if label is None:
            label= self.gen_label(image_types,itypes,figcount=figcount,
                                fillfrac=fillfrac,fillcount=(count if fcount else None),tlen=tlen,poly=poly)
        return list(a.flatten()) + [label] if flat else [a, label]

    # This produces labels for classification of single items, multiple-item counts, fractions or polygons.
    # The arg 'poly' is the size of the current polygon (used only when doing polygon classification tests).
    # For fractions, fillfrac is the fraction of filled objects, while fillcount is the NUMBER of filled
    # objects.  IF fillcount is given (as a number), then use it as the label.  Otherwise, use the fraction converted into a
    # numerator that corresponds to the fixed denominator ( = tlen - 1).  For example, if the fraction is 1/2 and
    # is represented by 2 of 4 objects being filled, then the label is either 2 (for fillcount-based) or 4 (for
    # fraction-based with tlen = 9 (meaning anywhere from 0 to 8 of 8 objects are filled).

    def gen_label(self,all_image_types,curr_image_types,figcount=1,
                  fillfrac=None,fillcount=None,tlen=None,poly=None):
        if poly is not None:    return self.get_polygon_index(poly)  # Polygon testing.
        elif type(fillfrac) in (int, float):  # Doing fraction tests.
            return fillcount if fillcount is not None else round(fillfrac * (tlen - 1))  # label between 0 and tlen-1
        else:
            return figcount if self.multi else all_image_types.index(curr_image_types[0])


    # When fillfrac != None, it can be a scalar or a pair used to determine the number of images that should be
    # filled; the rest will be unfilled.

    def gen_random_cases(self,count,image_types=_doodle_image_types_,flat=True,wr=None,hr=None,
                         figcount=1,fillfrac=None,fcount=False,poly=None,tlen=None):
        imt = image_types if image_types is not None else _doodle_image_types_
        if fillfrac is not None:
            self.gen_image_groups(imt) # Images grouped by whether filled or open.
            if type(fillfrac) == tuple: frange = fillfrac
            elif fillfrac == 'all': frange = (1.0, 1.0)
            else:   frange = (fillfrac, fillfrac)

        if not(tlen):
            tlen = figcount[1] - figcount[0] + 1 if type(figcount) == tuple else 1
        def genfc(): # Produce a random figure count
            return random.randint(*figcount) if type(figcount) == tuple else figcount
        def genff():  # Produce a random fill fraction from within the specified fraction range
            if fillfrac: return random.uniform(*frange)
        def genpoly():
            if poly:    return random.choice(self.polygon_sizes)

        return [self.gen_random_case(image_types=imt, flat=flat,wr=wr,hr=hr,figcount=genfc(),
                                     fillfrac=genff(),fcount=fcount,poly=genpoly(),tlen=tlen) for c in range(count)]

    # This generates cases where ALL shapes/images are the same (though maybe of different size), but there can be a
    #  range of cardinalities
    def gen_random_mono_shape_cases(self,count,image_types=_doodle_image_types_,figcount=(1,5),
                                    wr=[0.1,0.25],hr=[0.1,0.25],tlen=None,flat=True):
        cases = []
        for c in range(count):
            im = random.choice(image_types); fc = random.randint(*figcount)
            lab = self.calc_mono_shape_label(image_types,im,figcount,fc)
            cases.append(self.gen_random_case(image_types=[im],flat=flat,wr=wr,hr=hr,figcount=fc,tlen=tlen,label=lab))
        return cases

    # Labels for mono-shape images are based on location of image in the complete image list and the count.
    def calc_mono_shape_label(self,all_image_types, image, count_range, count):
        i = all_image_types.index(image)
        return i*(count_range[1] - count_range[0] + 1) + (count - count_range[0])

    def show_cases(self,cases):
        show_doodle_cases(cases)

# *************** MAIN *****************

# These are the cases typically used for IT-3030 exercises, since they: a) involve only one object per image, b)
# do not use fill fractions, c) do not do autoencoding cases.
# Arguments:
# count = # of cases to produce; rows = # rows in each image; cols = # columns in each image,
# types = list of types of images (all or a subset of _doodle_image_types_);
# wr, hr = width, height ranges, which are fractions of the total width / height of the canvas -- these control
# the size of each object on the canvas.  E.g. If wr=(0.2,0.4), then each object will have a width of 20-40% of
# the canvas width.
# noise = probability that any given pixel will be flipped from foreground to background or vice versa.
# cent = Center the object on the canvas?
# show = Display each image in its own window?
# flat = Should images be returned as flat vectors or 2-d arrays?
# fc = figure count (should stay at (1,1) unless you are doing counting tasks)
# auto = Are these autoencoder cases?  Probably not
# mono = Does each case include only one TYPE of object (though possibly many of them, depending upon the setting
# of fc.  mono = True for most purposes.
# one_hots = Should the targets be one-hot vectors?  Normally, this is True, unless doing autoencoding.
# multi = Will more than one object be displayed on a canvas at the same time?  Normally no.
# In general, students should play around with the first 10 arguments but leave the others at their defaults.
# NOTE:  The cases are returned as a 5-item tuple: (images, targets, labels, 2d-image-dimensions, flat).  The drawing
# and file dump / load routines assume that same format.

def gen_standard_cases(count=10,rows=50,cols=50,types=_doodle_image_types_,wr=[0.2,0.5],hr=[0.2,0.4],
                       noise=0, cent=False, show=True, flat=False,
                       fc=(1,1),auto=False,mono=True,one_hots=True,multi=False):
    return  gen_doodle_cases(count,rows=rows,cols=cols,imt=types,wr=wr,hr=hr,
                             nbias=noise,cent=cent,show=show,flat=flat,
                             mono=mono,auto=auto,fc=fc,one_hots=one_hots,multi=multi)

# NOTE:  This returns 3 lists, packed in a tuple: features, targets, labels
def gen_doodle_cases(count,rows=30,cols=30,imt=None, hr=[0.15,0.3],wr=[0.15,0.3],
                     nbias=0.0,cent=False, show=True,
                     one_hots=True,auto=False, flat=False,bkg=0, d4=False, fog=1, fillfrac=None, fc=(1,5),
                     gap=1,multi=False,mono=False, dbias=0.7,poly=(4,5,6)):

    d = Doodler(rows=rows, cols=cols, bkg=bkg, fog=fog, nbias=nbias, dbias=dbias, gap=gap, cent=cent,
                multi=multi,poly=poly)
    if mono:
        cases = d.gen_random_mono_shape_cases(count,image_types=imt,figcount=fc,flat=flat,wr=wr,hr=hr)
    else:
        cases = d.gen_random_cases(count, image_types=imt, flat=flat, wr=wr,
                                hr=hr, figcount=fc,fcount=fc,fillfrac=fillfrac)
    inputs = [c[:-1] for c in cases]  if flat else [c[0] for c in cases]
    fulldims = [count, rows*cols] if flat else [count,rows,cols]
    if d4:  fulldims.append(1)  #Specialized for DualNets (see neural/deepauto.py)
    inputs = np.array(inputs).reshape(fulldims)
    target_labels = [c[-1] for c in cases] if flat else [c[1] for c in cases]
    if auto:
        targets = inputs # It's an autoencoder, so targets = inputs
    else:
        targets = target_labels  # targets = labels
        # print("target labels = ", target_labels)
        if one_hots: # return one-hot vectors for the targets
            zfill = (mono or (fc[0] == 0))
            vsize = fc[1] - fc[0] + 1 # vector size = number of different cardinalities
            if mono:    vsize *= len(imt)  # For mono runs, vector size = # shapes x # cardinalities
            targets = np.array([np.array(integer_to_one_hot(targ,vsize,zerofill=zfill)) for targ in targets])
    cases = (inputs.astype(np.float32), targets, target_labels,(rows,cols),flat)
    if show:    d.show_cases(cases)
    return cases

def test_doodler(): return gen_standard_cases(25)

#  **********   Auxiliary Functions *****

def show_doodle_cases(cases):
    images,_,labels,dims,flat = cases  # Unpack the 5-item tuple
    if flat:
        images = [a.reshape(*dims) for a in images]
    graphics_start_interactive_mode()
    for image, label in zip(images, labels):
        quickplot_matrix(image, fs=None, title='Class = {}'.format(label))
    graphics_end_interactive_mode()

# This assumes all cases are packed into a 5-item tuple: (images, targets, labels, 2d-image-dimensions,flat).
def dump_doodle_cases(cases,filepath):
    f = open(filepath, 'wb')  # w => for writing, b => binary
    PK.dump(cases, f)

def load_doodle_cases(filepath):
    f = open(filepath,'rb') # r => read, b => binary
    return PK.load(f)

# Does any item in list L satisfy the boolean predicate?
def exists(L, predicate):
    for item in L:
        if predicate(item): return True
    return False

# Generate a list containing n copies of the item, or n calls to the same function.
def n_of(count, item):
    if isfunction(item):
        return [item() for i in range(count)]
    else:
        return [item for i in range(count)]

# Zerofill => the encoding for zero is [1,0,0...], where not(zerofill) makes it [0,0,0...], i.e. "no hot"
def integer_to_one_hot(int,size,off_val=0,on_val=1,floats=False,zerofill=True):
    if floats:
        off_val = float(off_val); on_val = float(on_val)
    i = int if zerofill else int-1
    if int == 0 and not(zerofill):
        return n_of(size,off_val)
    elif 0 <= i < size:
        v = n_of(size,off_val)
        v[i] = on_val
        return v

# Partition list L into sublists of size n.
def group(L,n):
    if n>=len(L):
        return [[item] for item in L] + [[]]*(n-len(L))
    else:
        groups = []; lmax = len(L); loc = 0
        while loc < lmax:
            groups.append(L[loc:min(loc+n,lmax)])
            loc += n
        return groups

# Randomly partition the elements (L) into n groups (of size len(L) / n).
def random_group(L,n):
    if n >=len(L):  return group(L,n)
    else:
        perm = list(np.random.permutation(L))
        return group(perm,math.ceil(len(L)/n))

def random_pair(L1,L2):
    L1p = list(np.random.permutation(L1))
    L2p = list(np.random.permutation(L2))
    return [[x,y] for x,y in zip(L1p,L2p)]

# Generate the angles and sides for (often random) convex polygons

# n = # sides = # angles ;
# This follows the algorithm presented online: http://cglab.ca/~sander/misc/ConvexGeneration/convex.html
# vect => return

def gen_convex_polygon(n,xr=(0,1),yr=(0,1),plot=True,vect=False,regular=False):
    dx = xr[1] - xr[0]; dy = yr[1] - yr[0]

    def calc_diffs(elems):
        lastval = elems[0]; diffs = []
        for elem in elems[1:]:
            diffs.append(elem-lastval)
            lastval = elem
        return diffs

    def calc_pt_extrema(pts):
        xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
        minx = min(xs); miny = min(ys)
        return minx,miny,max(xs)-minx,max(ys)-miny

    # Scale points into the range given by xr,yr
    def scale_pt(pt):
        return [xr[0] + dx*(pt[0] - minptx)/ptdx, yr[0] + dy*(pt[1] - minpty)/ptdy]

    if regular:
        xvals = np.linspace(0,1,n); yvals = np.linspace(0,1,n)
        groupsize = math.ceil(n-2/2)
        xgroups = group(xvals[1:-1], groupsize); ygroups = group(yvals[1:-1], groupsize)
    else:
        # Generate n random x and y values; sort them.
        xvals = sorted(np.random.uniform(0,1,size=n)); yvals = sorted(np.random.uniform(0,1,size=n))
        # Randomly divide the non-extrema into two subsets.
        xgroups = random_group(xvals[1:-1],2); ygroups = random_group(yvals[1:-1],2)
    # Identify the mins and maxs
    xmin = xvals[0]; xmax=xvals[-1]; ymin=yvals[0]; ymax=yvals[-1]
    # Paste the min and max back onto the ends of the sorted subsets
    xgroups = [[xmin]+sorted(g)+[xmax] for g in xgroups]; ygroups = [[ymin]+sorted(g)+[ymax] for g in ygroups]
    # Compute a sequence of delta-x (and delta-y) that take us from the xmin (ymin) out to xmax (ymax) and then
    # back to xmin (ymin).
    deltaxs = calc_diffs(xgroups[0]) + [-q for q in calc_diffs(xgroups[1])]
    deltays = calc_diffs(ygroups[0]) + [-q for q in calc_diffs(ygroups[1])]
    # Randomly pair the delta-x and delta-y values to produce a set of n vectors.
    vectors = random_pair(deltaxs,deltays)
    # Sort the vectors by angle.  Note that these are angles relative to the origin, not relative to one another, so
    # sorting does not imply that the largest (interior) angles will necessarily be adjacent in the polygon.
    vectors = sorted(vectors,key=(lambda v: math.atan2(v[1],v[0])))
    if plot or not(vect):
        pt = (0.0,0.0) ; pts = []  # If the algorithm works properly, the first pt gets added at the end of the cycle.
        for v in vectors:
            pt = (pt[0] + v[0], pt[1] + v[1])
            pts.append(pt)
        minptx,minpty,ptdx,ptdy = calc_pt_extrema(pts)  # Bind 4 variables used by scale_pt
        pts = [scale_pt(p) for p in pts]
        if plot:
            quickplot_path([pts[-1]]+pts)
    return vectors if vect else pts

# Distance between two points in n-dimensional space ( n > 0).  Each point should be a list or tuple.
def distance(p1,p2):
    return math.sqrt(sum([(a-b)**2 for a,b in zip(p1,p2)]))

# Calculate the angle between 2 points in the Cartesian plane, with due east being 0 and angles
# increasing counterclockwise.
def angle_2d(pt1,pt2,deg=False):
    dx = pt2[0] - pt1[0]; dy = pt2[1] - pt1[1]
    angle = math.atan2(dy,dx)
    return math.degrees(angle) if deg else angle

def graphics_start_interactive_mode(): PLT.ion()
def graphics_end_interactive_mode(): PLT.ioff()

def gen_new_axes():
    ax = PLT.figure().gca()
    ax.clear()
    return ax

# Based on code at: http://matplotlib.sourceforge.net/plot_directive/mpl_examples/api/image_zcoord.py
# This plots the color-coded values of a 2-d array

def quickplot_matrix(X, colormap=CMAP.jet, fs=False, ax=None, trans=False,
                     title="Array", xlabel="X", ylabel="Y"):
    if trans: X = X.transpose()
    # fs,axes = get_default_figspec_and_axes(fs=fs, clear_axes=clear)
    axes = gen_new_axes()
    # axes = fs.instantiate_axes(ax, auto=True, title=title, xlabel=xlabel, ylabel=ylabel)
    # PLT.hold(True)
    axes.set_title(title)
    axes.imshow(X, cmap=colormap, interpolation='nearest', aspect='auto', origin='lower')
    axes.patch.set_facecolor('gray') # Paint the background
    numrows, numcols = X.shape
    # Local func, used by PLT.show() to compute the proper color for each cell.
    def format_coord(x, y):
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < numcols and row >= 0 and row < numrows:
            z = X[row, col]
            return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
        else:
            return 'x=%1.4f, y=%1.4f' % (x, y)
    axes.format_coord = format_coord
    PLT.draw()

# Points is a list of 2-d locations
def quickplot_path(points,color='blue'):
    axes = gen_new_axes() 
    a = np.array(points).transpose()
    axes.plot(a[0], a[1],color=color)
    PLT.draw()