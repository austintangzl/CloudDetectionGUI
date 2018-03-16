import Polygon
from io import StringIO
from xml.dom.minidom import parseString
from zipfile import ZipFile
import math

# How to extract polygons from kml:
# polygons = readPoly(filename) to extract out all the polygons from the kml
#     for p in readPoly(fname):
#         p, desc = p

kmlstr = '''<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://earth.google.com/kml/2.0">
<Document>
<name>Polygon: %s</name>
  <open>1</open>
  %s
</Document>
</kml>'''

polystr = '''     <Placemark>
         <name>%i</name>
         <Polygon>
           <altitudeMode>clampedToGround</altitudeMode>
           <outerBoundaryIs>
           <LinearRing>
             <coordinates>
             %s
             </coordinates>
           </LinearRing>
           </outerBoundaryIs>
         </Polygon>
       </Placemark>'''


def openKMZ(filename):
    zip1 = ZipFile(filename)
    for z in zip1.filelist:
        if z.filename[-4:] == '.kml':
            fstring = zip1.read(z)
            break
    else:
        raise Exception("Could not find kml file in %s" % filename)
    return fstring


# noinspection PyBroadException
def openKML(filename):
    try:
        fstring = openKMZ(filename)
    except Exception:
        fstring = open(filename, 'r').read()
    return parseString(fstring)


# noinspection PyBroadException
def readPoly(filename):
    def parseData(polydata):
        dlines = polydata.split()
        mypoly = []
        for l in dlines:
            l = l.strip()
            if l:
                point = []
                for x in l.split(','):
                    point.append(float(x))
                mypoly.append(point[:2])
        return mypoly

    xml = openKML(filename)
    nodes = xml.getElementsByTagName('Placemark')
    desc = {}
    for n in nodes:
        names = n.getElementsByTagName('name')
        try:
            desc['name'] = names[0].childNodes[0].data.strip()
        except Exception:
            pass

        descriptions = n.getElementsByTagName('description')
        try:
            desc['description'] = names[0].childNodes[0].data.strip()
        except Exception:
            pass

        times = n.getElementsByTagName('TimeSpan')
        try:
            desc['beginTime'] = times[0].getElementsByTagName('begin')[0].childNodes[0].data.strip()
            desc['endTime'] = times[0].getElementsByTagName('end')[0].childNodes[0].data.strip()
        except Exception:
            pass

        times = n.getElementsByTagName('TimeStamp')
        try:
            desc['timeStamp'] = times[0].getElementsByTagName('when')[0].childNodes[0].data.strip()
        except Exception:
            pass

        polys = n.getElementsByTagName('Polygon')
        for poly in polys:
            invalid = False
            c = n.getElementsByTagName('coordinates')
            if len(c) != 1:
                print('invalid polygon found')
                continue
            if not invalid:
                c = c[0]
                d = c.childNodes[0].data.strip()
                data = parseData(d)
                yield (data, desc)


def latlon2meters(p):
    pi2 = 2. * math.pi
    reradius = 1. / 6370000
    alat = 0
    alon = 0
    for i in p:
        alon = alon + i[0]
        alat = alat + i[1]
    lon_ctr = alon / len(p)
    lat_ctr = alat / len(p)
    unit_fxlat = pi2 / (360. * reradius)
    unit_fxlon = math.cos(lat_ctr * pi2 / 360.) * unit_fxlat

    q = []
    olon = p[0][0]
    olat = p[0][1]
    for i in p:
        q.append(((i[0] - olon) * unit_fxlon,
                  (i[1] - olat) * unit_fxlat))
    return q


def polyStats(p):
    pm = Polygon(latlon2meters(p))
    area = pm.area()
    numpts = len(p)
    pl = Polygon(p)
    bbox = pl.boundingBox()
    center = pl.center()

    stat = \
        {'vertices': '%i' % numpts,
         'bounding box': '(%f , %f) - (%f , %f)' % (bbox[0], bbox[2], bbox[1], bbox[3]),
         'center': '(%f , %f)' % (center[0], center[1]),
         'area': '%f m^2' % area}
    return stat


def makepoly(p):
    return Polygon(p)


def intersect(p1, p2):
    q1 = makepoly(p1)
    q2 = makepoly(p2)

    q = q1 & q2

    return q


def get_area(p):
    q = makepoly(p)
    return p.area()


def write_poly(p, fname):
    if isinstance(fname, str):
        f = open(fname, 'w')
    else:
        f = fname
    for i in p:
        f.write('%19.16f,%19.16f,0.\n' % (i[0], i[1]))
    f.flush()


def poly2kmz(pp, fname):
    strs = []
    i = 0
    for p in pp:
        i = i + 1
        f = StringIO()
        write_poly(p, f)
        strs.append(polystr % (i, f.getvalue()))
    s = '\n'.join(strs)
    s = kmlstr % (fname, s)
    open(fname, 'w').write(s)