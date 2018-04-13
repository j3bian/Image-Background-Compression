
from math import *
import numpy

def raise_(msg):
    raise msg

def identity(x):
    return x

class wrap:
    
    def __init__(self, o):
        self.__dict__['__object__'] = o
        
    def __getattr__(self, attr):
        d = self.__dict__
        if not d.has_key(attr):
            return getattr(d['__object__'], attr)
        else:
            return d[attr]

    def __setattr__(self, attr, value):
        d = self.__dict__
        o = d['__object__']
        if not hasattr(o, attr):
            d[attr] = value
        else:
            setattr(o, attr, value)


def unwrap(x):
    if isinstance(x, Wrap):
        return x.__object__
    else:
        return x


def cacheresult(fn):
    vname = "_" + fn.__name__
    def new_fn(self, *args):
        v = getattr(self, vname, None)
        if v:
            return v
        else:
            v = fn(self, *args)
            setattr(self, vname, v)
            return v
    return new_fn


class D:
    def __init__(self, **d):
        self.__dict__ = d
    def __repr__(self):
        return repr(self.__dict__)


def scan(a,n):
    for (x, v) in a:
        if n <= x:
            return v
    return None

def find(a, key):
    for (k, v) in a:
        if k == key:
            return v
    return None

def invfind(a, key):
    for (v, k) in a:
        if k == key:
            return v
    return None

def choice_except(a, no):
    # a is assumed to contain more than 1 element
    c = no
    while c == no:
        c = choice(a)
    return c

def cycle(a, o):
    b = a[:]
    l = len(a)
    o = o % l
    for i in xrange(l):
        b[i] = a[(i-o)%l]
    return b

def constrain(a,(low,up)):
    return [min(up,max(low,x)) for x in a]

def merge(s):
    x = []
    for a in s:
       x+=a
    return x   
    
def lookat(x):
    print x
    return x

def onehot(n, i):
    a = [0 for whatever in xrange(n)]
    a[i] = 1
    return a

oh = onehot

def product(a):
    ans = 1
    for x in a:
        ans = ans * x
    return ans


def overlap(bbox1, bbox2):
    ((minx1, miny1), (maxx1, maxy1)) = bbox1
    ((minx2, miny2), (maxx2, maxy2)) = bbox2
    return (((minx1 > minx2) and (minx1 < maxx2)) or ((minx2 > minx1) and (minx2 < maxx1))) and \
           (((miny1 > miny2) and (miny1 < maxy2)) or ((miny2 > miny1) and (miny2 < maxy1)))


def pminus(p1, p2):
    return (p1[0] - p2[0], p1[1] - p2[1])

def pplus(p1, p2):
    return (p1[0] + p2[0], p1[1] + p2[1])

def pnormal(p):
    return (-p[1], p[0])

def pdot(p1, p2):
    return p1[0] * p2[0] + p1[1] * p2[1]

def plen(p):
    return sqrt(p[0] * p[0] + p[1] * p[1])

def pavg(pts):
    return (sum([p[0] for p in pts]) / len(pts), sum([p[1] for p in pts]) / len(pts))

def intersect_factors_helper((p11, p12), (p21, p22)):
    pp = (p22[0] - p21[0], p22[1] - p21[1])
    pp = (-pp[1], pp[0])
    a1x = (p12[0] - p11[0], p12[1] - p11[1])
    a1 = a1x[0] * pp[0] + a1x[1] * pp[1]
    a2x = (p21[0] - p11[0], p21[1] - p11[1])
    a2 = a2x[0] * pp[0] + a2x[1] * pp[1]
    if a1 == 0:
        return None
    else:
        return float(a2) / a1

def intersect_factors((p11, p12), (p21, p22)):
    a = intersect_factors_helper((p11, p12), (p21, p22))

    if a != None:
        b = intersect_factors_helper((p21, p22), (p11, p12))
        return (a,b)
    else:
        return None

def point_in_polygon(p, sides):
    dots = [pdot(pminus(side[1], side[0]), pnormal(pminus(p, side[0]))) for side in sides]
    sign1 = dots[0] < 0
    for dot in dots:
        if (dot < 0) != sign1:
            return False
    return True
    
def polygon_overlap(bpol1, bpol2):
    sides1 = zip(bpol1, cycle(bpol1, 1))
    sides2 = zip(bpol2, cycle(bpol2, 1))

    all = [0]*len(sides1)*len(sides2)
    i = 0
    for side1 in sides1:
        for side2 in sides2:
            all[i] = (side1, side2)
            i = i + 1

    intersect = False
    for tup in all:
        x = intersect_factors(tup[0], tup[1])
        if x:
            a = x[0]
            b = x[1]
            if a >= 0 and a <= 1 and b >= 0 and b <= 1:
                intersect = True
                break

    return intersect or point_in_polygon(bpol1[0], sides2) or point_in_polygon(bpol2[0], sides1)


def rotate_polygon(pol, theta, (cx, cy)):
    if theta:
        ct, st = cos(theta), sin(theta)
        return [(round(ct*(x-cx) - st*(y-cy) + cx, 10), round(st*(x-cx) + ct*(y-cy) + cy, 10)) for (x,y) in pol]
    else:
        return pol

def bpol_to_bbox(bpol):
    xs = [p[0] for p in bpol]
    ys = [p[1] for p in bpol]
    return ((min(xs), min(ys)), (max(xs), max(ys)))

def pull(poly, fix, scale):
    outscale = 1 - scale
    lp = len(poly)
    newpoly = [0] * lp
    offset = (0,0)
    for point, i in zip(poly, xrange(lp)):
        diff = pminus(fix, point)
        v = (diff[0] * outscale, diff[1] * outscale)
        newpoly[i] = pplus(point, v)
        offset = pplus(offset, (v[0] / lp, v[1] / lp))
    return (offset, newpoly)

# def detangle(polys):
#     polys = [x for x in polys]
#     npolys = len(polys)
#     scalings = [1.0 for i in xrange(npolys)]
#     translations = [(0.,0.) for i in xrange(npolys)]
# #    return zip(scalings, translations)
#     queue = [(i, len(poly), zip(cycle(poly, 1), poly)) for poly, i in zip(polys, xrange(npolys))]
#     zzz = 0
#     while queue:
#         i, nsides1, sides1 = queue.pop()
#         for j, nsides2, sides2 in queue:
#             k = 0
#             while k < nsides1:
#                 l = 0
#                 side1 = sides1[k]
#                 while l < nsides2:
#                     side2 = sides2[l]
# #             for side1 in sides1:
# #                 for side2 in sides2:
#                     factors = intersect_factors(side1, side2)
#                     if factors:
#                         a,b = factors
#                         if a > 0 and a < 1 and b > 0 and b < 1:
#                             v1 = pnormal(pminus(side1[1], side1[0]))
#                             v2 = pminus(side2[1], side2[0])
# #                            print pdot(v1, v2), a, b, v1, v2, side1, side2
#                             if pdot(v1, v2) > 0:
#                                 fix = side2[1]
#                                 scale = 1 - b
#                             else:
#                                 fix = side2[0]
#                                 scale = b
#                             offset, newpoly2 = pull(polys[j], fix, scale * 0.95)
#                             polys[j] = newpoly2
# #                            print "?!?", a, repr(b), scale, a > 0 and a < 1 and b > 0 and b < 1
# #                            print fix, offset, polys[j], newpoly2
#                             scalings[j] = scalings[j] * scale
#                             translations[j] = pplus(translations[j], offset)
#                             oldsides2 = sides2
#                             sides2 = zip(cycle(newpoly2, 1), newpoly2)
#                             idx = queue.index((j, nsides2, oldsides2))
# #                            print newpoly2
#                             zzz = zzz + 1
# #                            print zzz
#                             queue[idx] = (j, nsides2, sides2)
#                             k = -1
#                             break
#                         else:
#                             l = l + 1
#                     else:
#                         l = l + 1
#                 k = k + 1
#     return zip(scalings, translations)


def compress_array_1bit(a):
    a2 = numpy.int8([])
    a2.resize(ceil(len(a) / 8.0))
    la = len(a)
    for i in xrange(len(a2)):
        idx = i<<3
        if idx + 7 < la:
            a2[i] = a[idx] | (a[idx+1]<<1) | (a[idx+2]<<2) | (a[idx+3]<<3) | (a[idx+4]<<4) | (a[idx+5]<<5) | (a[idx+6]<<6) | (a[idx+7]<<7)
        else:
            la2 = la - 1
            a2[i] = a[idx] | (a[min(idx+1,la2)]<<1) | (a[min(idx+2,la2)]<<2) | (a[min(idx+3,la2)]<<3) | (a[min(idx+4,la2)]<<4) | (a[min(idx+5,la2)]<<5) | (a[min(idx+6,la2)]<<6) | (a[min(idx+7,la2)]<<7)
    return a2

def compress_array_2bits(a):
    a2 = numpy.int8([])
    a2.resize(ceil(len(a) / 4.0))
    la = len(a)
    for i in xrange(len(a2)):
        idx = i<<2
        if idx + 3 < la:
            a2[i] = a[idx] | (a[idx+1]<<2) | (a[idx+2]<<4) | (a[idx+3]<<6)
        else:
            la2 = la - 1
            a2[i] = a[idx] | (a[min(idx+1,la2)]<<2) | (a[min(idx+2,la2)]<<4) | (a[min(idx+3,la2)]<<6)
    return a2

def compress_array(a, bits_per_entry):
    if not isinstance(a, numpy.ndarray):
        a = numpy.array(a)
#     a2 = numpy.int32([])
#     nbits = 32
    a2 = numpy.uint8([])
    nbits = 8
    a2.resize(ceil(len(a) * bits_per_entry / float(nbits)))
    current = 0
    bits_remaining = nbits
    i = 0
    for v in a:
        bits_written = nbits - bits_remaining
        if not bits_remaining:
            a2[i] = current
            current = v
            bits_remaining = nbits - bits_per_entry
            i = i + 1
        elif bits_remaining >= bits_per_entry:
            current = current | (v << bits_written)
            bits_remaining -= bits_per_entry
        else:
#            a2[i] = current | ((v & ((1 << bits_remaining) - 1)) << bits_written)
            a2[i] = current | (v << bits_written)
            i = i + 1
            current = v >> bits_remaining
            bits_remaining = nbits - bits_per_entry + bits_remaining
    a2[-1] = current
    return a2





