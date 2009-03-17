#include <eo>
#include <mo>
#include <moeo>


class ObjectiveVectorTraits : public moeoObjectiveVectorTraits
{
public:
    static bool minimizing (int i)
    {
        return true;
    }
    static bool maximizing (int i)
    {
        return false;
    }
    static unsigned int nObjectives ()
    {
        return 2;
    }
};

typedef moeoRealObjectiveVector < ObjectiveVectorTraits > ObjectiveVector;

typedef MOEO < ObjectiveVector, double, double > Solution;

class testMove : public moMove < Solution >
{
public :

  void operator () (Solution & _solution)
  {
    Solution sol=_solution;
    counter++;
  }

  void setCounter(unsigned int i){
	  counter=i;
  }

  unsigned int getCounter(){
	  return counter;
  }

private:
	unsigned int counter;

} ;

class testMoveInit : public moMoveInit <testMove>
{
public :
  void operator () (testMove & _move, const Solution & _solution)
  {
    _move.setCounter(0);
    const Solution sol(_solution);
  }
} ;

class testMoveNext : public moNextMove <testMove>
{
public :

	testMoveNext(unsigned int _counter=0):counter(_counter){};

  bool operator () (testMove & _move, const Solution & _solution)
  {
    testMove move(_move);
    const Solution sol(_solution);
    return _move.getCounter() < 5;
  }

private:
	unsigned int counter;
} ;

class testMoveIncrEval : public moMoveIncrEval <testMove, ObjectiveVector>
{
public :
  ObjectiveVector operator () (const testMove & _move, const Solution & _solution)
  {
    ObjectiveVector objVec= _solution.objectiveVector();
    objVec[0]+=2;
    objVec[1]+=2;
    return objVec;
  }
} ;
