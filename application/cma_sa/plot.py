#!/usr/bin/env python

"""plot.py -- Plot CMA-SA results file"""

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

    filelist = os.listdir(path)
    filelist.sort()

    return filelist

def plotXPointYFitness(path, fields='3:1', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    g.title('Fitness observation')
    g.xlabel('Coordinates')
    g.ylabel('Fitness (Quality)')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  title='distribution \'' + filename + '\''))

    g.plot(*files)

    return g

def plotXYPointZFitness(path, fields='4:3:1', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    g.title('Fitness observation in 3-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')
    g.zlabel('Fitness (Quality)')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  title='distribution \'' + filename + '\''))

    g.splot(*files)

    return g

def plotXYPoint(path, fields='3:4', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    g.title('Points observation in 2-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  title='distribution \'' + filename + '\''))

    g.plot(*files)

    return g

def plotXYZPoint(path, fields='3:4:5', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    g.title('Points observation in 3-D')
    g.xlabel('x-axes')
    g.ylabel('y-axes')
    g.zlabel('z-axes')

    files=[]
    for filename in getSortedFiles(path):
        files.append(Gnuplot.File(path + '/' + filename, using=fields,
                                  with_='points',
                                  title='distribution \'' + filename + '\''))

    g.splot(*files)

    return g

def plotParams(path, field='1', state=None, g=None):
    if g == None: g = Gnuplot.Gnuplot()
    if state != None: state.append(g)

    g.title('Hyper-volume comparaison through all dimensions')
    g.xlabel('Iterations')
    g.ylabel('Hyper-volume')

    g.plot(Gnuplot.File(path, with_='lines', using=field,
                        title='multivariate distribution narrowing'))

    return g

def plot2DRectFromFiles(path, state=None, g=None, plot=True):
    if g == None: g = Gnuplot.Gnuplot()
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

def main(n):
    gstate = []

    if n >= 1:
        plotXPointYFitness('./ResPop', state=gstate)

    if n >= 2:
        plotXPointYFitness('./ResPop', '4:1', state=gstate)

    if n >= 2:
        plotXYPointZFitness('./ResPop', state=gstate)

    if n >= 3:
        plotXYZPoint('./ResPop', state=gstate)

    if n >= 1:
        plotParams('./ResParams.txt', state=gstate)

    if n >= 2:
        plot2DRectFromFiles('./ResBounds', state=gstate)
        plotXYPoint('./ResPop', state=gstate)

        g = plot2DRectFromFiles('./ResBounds', state=gstate, plot=False)
        plotXYPoint('./ResPop', g=g)

    wait(prompt='Press return to end the plot.\n')

    pass

# when executed, just run main():
if __name__ == '__main__':
    from sys import argv, exit

    if len(argv) < 2:
        print 'Usage: plot [dimension]'
        exit()

    main(int(argv[1]))
