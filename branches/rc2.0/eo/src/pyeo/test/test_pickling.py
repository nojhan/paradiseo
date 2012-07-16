from maxone import *
import unittest
import cPickle
import tempfile
import os

class TestPickling(unittest.TestCase):
    def do_pickle(self, object):
	filename = tempfile.mktemp()
	file = open(filename, 'wb')
	pickler = cPickle.Pickler(file)

	pickler.dump(object);
	del pickler
	file.close()

	file = open(filename)

	unpickler = cPickle.Unpickler(file)

	object2 = unpickler.load()

	del unpickler
	file.close()
	os.remove(filename)

	return object2

    def testInvalidEO(self):
	eo = EO();
	eo.genome = [1,2,3];

	eo2 = self.do_pickle(eo)
	self.failUnlessEqual( str(eo), str(eo2) )

    def testValidEO(self):
	eo = EO();
	eo.genome = [1,2,3];
	eo.fitness = 10

	eo2 = self.do_pickle(eo)

	self.failUnlessEqual( str(eo), str(eo2) )

    def testPop(self):
	pop = eoPop(40, init)
	for indy in pop:
	    evaluate(indy)

	pop2 = self.do_pickle(pop)

	self.failUnlessEqual( str(pop), str(pop2) )

    def testHowMany(self):
	howmany = eoHowMany(0.8);
	howmany2 = self.do_pickle(howmany)

	self.failUnlessEqual(howmany(10), howmany2(10))
	self.failUnlessEqual( str(howmany), str(howmany) )

    def testRNG(self):
	for i in range(10):
	    rng().rand()

	rng2 = self.do_pickle(rng())

	for i in range(100):
	    a = rng().rand()
	    b = rng2.rand()
	    self.failUnlessEqual(a,b)

    def vParam(self,v):
	v2 = self.do_pickle(v);
	self.failUnlessEqual(v.value, v2.value)

    def testValueParam(self):
	import Numeric

	self.vParam(eoValueParamInt(42,'int'))
	self.vParam(eoValueParamFloat(4.2,'float'))

	v = eoValueParamVec()
	v.value = Numeric.arange(10)
	self.vParam(v)

	v = eoValueParamPair()
	v.value = (0.3,0.5)
	self.vParam(v)

if __name__=='__main__':
    unittest.main()
