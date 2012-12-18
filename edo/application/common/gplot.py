#!/usr/bin/env python

"""plot.py -- Plot EDA-SA results file"""

import os, time, math, tempfile
import numpy

try:
    import Gnuplot, Gnuplot.PlotItems, Gnuplot.funcutils
except ImportError:
    # kludge in case Gnuplot hasn't been installed as a module yet:
    import __init__
    Gnuplot = __init__
    import PlotItems
    Gnuplot.PlotItems = PlotItems
    import funcutils
    Gnuplot.funcutils = funcutils

import optparse, logging, sys

LEVELS = {'debug': logging.DEBUG,
          'info': logging.INFO,
          'warning': logging.WARNING,
          'error': logging.ERROR,
          'critical': logging.CRITICAL}

def logger(level_name, filename='plot.log'):
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
        filename=filename, filemode='a'
        )

    console = logging.StreamHandler()
    console.setLevel(LEVELS.get(level_name, logging.NOTSET))
    console.setFormatter(logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s'))
    logging.getLogger('').addHandler(console)

def parser(parser=optparse.OptionParser()):
    parser.add_option('-v', '--verbose', choices=LEVELS.keys(), default='warning', help='set a verbose level')
    parser.add_option('-f', '--files', help='give some input sample files separated by comma (cf. gen1,gen2,...)', default='')
    parser.add_option('-r', '--respop', help='define the population results containing folder', default='./ResPop')
    parser.add_option('-o', '--output', help='give an output filename for logging', default='plot.log')
    parser.add_option('-d', '--dimension', help='give a dimension size', default=2)
    parser.add_option('-m', '--multiplot', action="store_true", help='plot all graphics in one window', dest="multiplot", default=True)
    parser.add_option('-p', '--plot', action="store_false", help='plot graphics separetly, one by window', dest="multiplot")
    parser.add_option('-w', '--windowid', help='give the window id you want to display, 0 means we display all ones, this option should be combined with -p', default=0)
    parser.add_option('-G', '--graphicsdirectory', help='give a directory name for graphics, this option should be combined with -u', default='plot')
    parser.add_option('-g', '--graphicsprefixname', help='give a prefix name for graphics, this option should be combined with -u', default='plot')
    parser.add_option('-t', '--terminal', action="store_true", help='display graphics on gnuplot windows', dest="terminal", default=True)
    parser.add_option('-u', '--png', action="store_false", help='display graphics on png files', dest="terminal")

    options, args = parser.parse_args()

    logger(options.verbose, options.output)

    return options

options = parser()

def wait(str=None, prompt='Press return to show results...\n'):
    if str is not None:
        print str
    raw_input(prompt)

def draw2DRect(min=(0,0), max=(1,1), color='black', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    xmin, ymin = min
    xmax, ymax = max

    cmd = 'set arrow from %s,%s to %s,%s nohead lc rgb "%s"'

    g(cmd % (xmin, ymin, xmin, ymax, color))
    g(cmd % (xmin, ymax, xmax, ymax, color))
    g(cmd % (xmax, ymax, xmax, ymin, color))
    g(cmd % (xmax, ymin, xmin, ymin, color))

    return g

def draw3DRect(min=(0,0,0), max=(1,1,1), state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    # TODO

    return g

def getSortedFiles(path):
    assert path != None

    if options.files == '':

        filelist = os.listdir(path)
        filelist.sort()

    else:

        filelist = options.files.split(',')

    checkFileErrors(path, filelist)

    return filelist

def checkFileErrors(path, filelist):
    for filename in filelist:
        for line in open('%s/%s' % (path, filename)):
            if '-nan' in line:
                logging.warning("checkFileErrors: %s/%s file contains bad value, it is going to be skipped" % (path, filename))
                filelist.remove(filename)
                break

def plotXPointYFitness(path, fields='3:1', state=None, g=None):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'plotXPointYFitness'))

    if state != None: state.append(g)

    g.title('Fitness observation')
    g.xlabel('Coordinates')
    g.ylabel('Fitness (Quality)')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  #title='distribution \'' + filename + '\''
                                  title=""
                                  )
                     )

    if len(files) > 0:
        g.plot(*files)

    return g

def plotXYPointZFitness(path, fields='4:3:1', state=None, g=None):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'plotXYPointZFitness'))

    if state != None: state.append(g)

    g.title('Fitness observation in 3-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')
    g.zlabel('Fitness (Quality)')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  #title='distribution \'' + filename + '\''
                                  title=""
                                  )
                     )

    if len(files) > 0:
        g.splot(*files)

    return g

def plotXYPoint(path, fields='3:4', state=None, g=None):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'plotXYPoint'))

    if state != None: state.append(g)

    g.title('Points observation in 2-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  #title='distribution \'' + filename + '\''
                                  title=""
                                  )
                     )

    if len(files) > 0:
        g.plot(*files)

    return g

def plotXYZPoint(path, fields='3:4:5', state=None, g=None):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'plotXYZPoint'))

    if state != None: state.append(g)

    g.title('Points observation in 3-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')
    g.zlabel('z-axes')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  #title='distribution \'' + filename + '\''
                                  title=""
                                  )
                     )

    if len(files) > 0:
        g.splot(*files)

    return g

def plotParams(path, field='1', state=None, g=None):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'plotXYZPoint'))

    if state != None: state.append(g)

    g.title('Hyper-volume comparaison through all dimensions')
    g.xlabel('Iterations')
    g.ylabel('Hyper-volume')

    g.plot(Gnuplot.File(path, with_='lines', using=field,
                        title='multivariate distribution narrowing'))

    return g

def plot2DRectFromFiles(path, state=None, g=None, plot=True):
    if g == None:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s_%s.png\'' % (options.graphicsprefixname, 'plot2DRectFromFiles'))

    if state != None: state.append(g)

    g.title('Rectangle drawing observation')
    g.xlabel('x-axes')
    g.ylabel('y-axes')

    x1,x2,y1,y2 = 0,0,0,0

    colors = ['red', 'orange', 'blue', 'green', 'gold', 'yellow', 'gray']
    #colors = open('rgb.txt', 'r').readlines()
    colors_size = len(colors)
    i = 0 # for color

    for filename in getSortedFiles(path):
        line = open(path + '/' + filename, 'r').readline()

        fields = line.split(' ')

        if not fields[0] == '2':
            print 'plot2DRectFromFiles: higher than 2 dimensions not possible to draw'
            return

        xmin,ymin,xmax,ymax = fields[1:5]
        #print xmin,ymin,xmax,ymax

        cur_color = colors[i % colors_size]

        draw2DRect((xmin,ymin), (xmax,ymax), cur_color, g=g)

        g('set obj rect from %s,%s to %s,%s back lw 1.0 fc rgb "%s" fillstyle solid 1.00 border -1'
          % (xmin,ymin,xmax,ymax,cur_color)
          )

        if plot:
            if float(xmin) < x1: x1 = float(xmin)
            if float(ymin) < y1: y1 = float(ymin)
            if float(xmax) > x2: x2 = float(xmax)
            if float(ymax) > y2: y2 = float(ymax)

        #print x1,y1,x2,y2

        i += 1

    #print x1,y1,x2,y2

    if plot:
        g.plot('[%s:%s][%s:%s] -9999 notitle' % (x1, x2, y1, y2))

    return g

def main():
    gstate = []

    n = int(options.dimension)
    w = int(options.windowid)
    r = options.respop

    if not options.terminal:
        try:
            os.mkdir(options.graphicsdirectory)
        except OSError:
            pass

    if options.multiplot:
        g = Gnuplot.Gnuplot()

        if not options.terminal:
            g('set terminal png')
            g('set output \'%s/%s_%s.png\'' % (options.graphicsdirectory, options.graphicsprefixname, 'multiplot'))

        g('set parametric')
        g('set nokey')
        g('set noxtic')
        g('set noytic')
        g('set noztic')

        g('set size 1.0, 1.0')
        g('set origin 0.0, 0.0')
        g('set multiplot')

        g('set size 0.5, 0.5')
        g('set origin 0.0, 0.5')

        if n >= 1:
            plotXPointYFitness(r, state=gstate, g=g)

        g('set size 0.5, 0.5')
        g('set origin 0.0, 0.0')

        if n >= 2:
            plotXPointYFitness(r, '4:1', state=gstate, g=g)

        g('set size 0.5, 0.5')
        g('set origin 0.5, 0.5')

        if n >= 2:
            plotXYPointZFitness(r, state=gstate, g=g)

        g('set size 0.5, 0.5')
        g('set origin 0.5, 0.0')

        if n >= 2:
            plotXYPoint(r, state=gstate, g=g)
        elif n >= 3:
            plotXYZPoint(r, state=gstate, g=g)

        g('set nomultiplot')

    else:

        if n >= 1 and w in [0, 1]:
            plotXPointYFitness(r, state=gstate)

        if n >= 2 and w in [0, 2]:
            plotXPointYFitness(r, '4:1', state=gstate)

        if n >= 2 and w in [0, 3]:
            plotXYPointZFitness(r, state=gstate)

        if n >= 3 and w in [0, 4]:
            plotXYZPoint(r, state=gstate)

        if n >= 2 and w in [0, 5]:
            plotXYPoint(r, state=gstate)

    # if n >= 1:
    #     plotParams('./ResParams.txt', state=gstate)

    # if n >= 2:
    #     plot2DRectFromFiles('./ResBounds', state=gstate)
    #     plotXYPoint(r, state=gstate)

    #     g = plot2DRectFromFiles('./ResBounds', state=gstate, plot=False)
    #     plotXYPoint(r, g=g)

    if options.terminal:
        wait(prompt='Press return to end the plot.\n')

# when executed, just run main():
if __name__ == '__main__':
    logging.debug('### plotting started ###')

    main()

    logging.debug('### plotting ended ###')
