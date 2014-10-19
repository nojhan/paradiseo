from maxone import *
import unittest

class Init(eoInit):
    def __call__(self, eo):
        pass

class TestSGA(unittest.TestCase):
    def __init__(self, a):
        unittest.TestCase.__init__(self, a)
        self.pop = eoPop(4, Init())

        for i in range(len(self.pop)):
           self.pop[i].fitness = i;


    def do_test(self, selectOne):
        print '*'*20, "Testing", str(selectOne.__class__), '*'*20
        selection = [0. for i in range(len(self.pop))]

        nTries = 500
        for i in range(nTries):
            selection[ selectOne(self.pop).fitness ] += 1

        for i in range(len(self.pop)):
            print i, selection[i], selection[i] / nTries

        return selection, nTries

    def test_eoDetTournamenSelect(self):
        selectOne = eoDetTournamentSelect(2)
        self.do_test(selectOne)

    def test_eoRandomSelect(self):
        selectOne = eoRandomSelect()
        self.do_test(selectOne)

    def test_eoBestSelect(self):
        selectOne = eoBestSelect()
        (sel, nTries) = self.do_test(selectOne)

        self.failUnlessEqual(sel[0], 0);
        self.failUnlessEqual(sel[-1], nTries);

    def test_eoNoSelect(self):
        selectOne = eoNoSelect()
        self.do_test(selectOne)

    def test_eoStochTournament(self):
        selectOne = eoStochTournamentSelect(0.75)
        self.do_test(selectOne)

    def test_eoSequentialSelect(self):
        selectOne = eoSequentialSelect();
        self.do_test(selectOne)

    def test_eoEliteSequentialSelect(self):
        selectOne = eoEliteSequentialSelect();
        self.do_test(selectOne)

if __name__=='__main__':
    unittest.main()
