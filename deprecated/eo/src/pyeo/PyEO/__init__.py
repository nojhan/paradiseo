from PyEO import *

try:
    import Gnuplot
except ImportError:
    print "Python support for Gnuplot not found"
else:

    class eoGnuplot1DMonitor(eoMonitor):
        def __init__(self):
            eoMonitor.__init__(self)
            self.values = []
            self.indices = []
            self.g = Gnuplot.Gnuplot()
            self.g.reset();

        def handleParam(self, i, param):
            param = float(param)

            while len(self.values) <= i:
                self.values.append( [] )

            self.values[i].append(param)

        def __call__(self):
            l = len(self)

            if l > 3 or l == 0:
                print 'Can only handle 1 to 3 params currently'

            i = 0
            for param in self:
                self.handleParam(i,param)
                i += 1

            self.indices.append( len(self.indices) )


            data1 = Gnuplot.Data(self.indices, self.values[0], with = 'lines')

            if l == 1:
                self.g.plot(data1)
            else:
                data2 = Gnuplot.Data(self.indices, self.values[1], with = 'lines')

                if l == 2:
                    self.g.plot(data1, data2)
                else:
                    data3 = Gnuplot.Data(self.indices, self.values[2], with = 'lines')

                    self.g.plot(data1, data2, data3)

def SeperatedVolumeMonitor(eoMonitor):
    def __init__(self, file):
        eoMonitor.__init__(self)
        self.file = file
        self.initialized = None;

    def __call__(self):
        pass

class eoStat(eoStatBase, eoValueParam):
    def __init__(self):
        eoStatBase.__init__(self)
        eoValueParam.__init__(self)

class eoSortedStat(eoSortedStatBase, eoValueParam):
    def __init__(self):
        eoSortedStatBase.__init__(self)
        eoValueParam.__init__(self)

class eoAverageStat(eoStat):
    def __call__(self, pop):
        sum = 0.0;
        for indy in pop:
            sum += indy.fitness

        sum /= len(pop)
        self.object = sum

class eoBestFitnessStat(eoSortedStat):
    def __call__(self, pop):
        self.object = pop[0].fitness
