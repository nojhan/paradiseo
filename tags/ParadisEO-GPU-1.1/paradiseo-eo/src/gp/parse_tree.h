#ifndef PARSE_TREE_HH
#define PARSE_TREE_HH

/**

 *	Parse_tree and subtree classes
 *  (c) copyright Maarten Keijzer 1999, 2000

 * Permission to copy, use,  modify, sell and distribute this software is granted provided
 * this copyright notice appears in all copies. This software is provided "as is" without
 * express or implied warranty, and with no claim as to its suitability for
 * any purpose.

 * Permission to modify the code and to distribute modified code is granted,
 * provided the above notices as well as this one are retained, and a notice that the code was
 * modified is included with the above copyright notice.



  Usage information.

  class Node (your node in the tree) must have the following implemented:

  ******      Arity      ******
    \code
        int arity(void) const
    \endcode

        Note:	the default constructor of a Node should provide a
                        Node with arity 0!

  ******    Evaluation   ******

  A parse_tree is evaluated through one of it's apply() members:

        1) parse_tree::apply(RetVal)

           is the simplest evaluation, it will call

    \code
           RetVal Node::operator()(RetVal, subtree<Node, RetVal>::const_iterator)
    \endcode

       (Unfortunately the first RetVal argument is mandatory (although you
       might not need it. This is because MSVC does not support member template
       functions properly. If it cannot deduce the template arguments (as is
       the case in templatizing over return value) you are not allowed to
       specify them. calling tree.apply<double>() would result in a syntax
       error. That is why you have to call tree.apply(double()) instead.)


        2) parse_tree::apply(RetVal v, It values)

       will call:

    \code
       RetVal Node::operator()(RetVal, subtree<... , It values)
    \endcode

       where It is whatever type you desire (most of the time
           this will be a std::vector containing the values of your
           variables);

        3) parse_tree::apply(RetVal, It values, It2 moreValues)

       will call:

    \code
       RetVal Node::operator()(RetVal, subtree<... , It values, It2 moreValues)
    \endcode

           although I do not see the immediate use of this, however...

        4) parse_tree::apply(RetVal, It values, It2 args, It3 adfs)

                that calls:

    \code
       RetVal Node::operator()(subtree<... , It values, It2 args, It3 adfs)
    \endcode

                can be useful for implementing adfs.


        In general it is a good idea to leave the specifics of the
        arguments open so that different ways of evaluation remain
        possible. Implement the simplest eval as:

    \code
        template <class It>
                RetVal operator()(RetVal dummy, It begin) const
    \endcode

  ******	Internal Structure    ******

  A parse_tree has two template arguments: the Node and the ReturnValue
  produced by evaluating the node. The structure of the tree is defined
  through a subtree class that has the same two template arguments.

  The nodes are stored in a tree like :

                        node4
                        /  \
                 node3 node2
                        / \
            node1 node0

  where nodes 2 and 4 have arity 2 and nodes 0,1 and 3 arity 0 (terminals)

  The nodes are subtrees, containing the structure of the tree, together
  with its size and depth. They contain a Node, the user defined template
  argument. To access these nodes from a subtree, use operator-> or operator*.

  The numbers behind the nodes define a reverse-polish or postfix
  traversel through the tree. The parse_tree defines iterators
  on the tree such that

  tree.begin() points at the subtree at node0 and
  tree.back()  returns the subtree at node4, the complete tree

  Likewise operator[] is defined on the tree, such that:

  tree[0] will return the subtree at node0, while
  tree[2] will return the subtree at node2

  Assigments of subtrees is protected so that the code:

  tree[2] = tree[0];

  will not crash and result in a tree structured as:

                node4
                /  \
         node3 node0

  Note that the rank numbers no longer specify their place in the tree:

  tree[0] still points at node0, but
  tree[1] now points to node3 and
  tree[2] points at the root node4

  Embedded iterators are implemented to iterate over nodes rather
  than subtrees. So an easy way to copy your tree to a std::vector is:

  std::vector<Node> vec(tree.size());
  copy(tree.ebegin(), tree.eend(), vec.begin());

  You can also copy it to an std::ostream_iterator with this
  technique, given that your Node implements an appropriate
  operator<<. Reinitializing a tree with the std::vector is also
  simple:

  tree.clear();
  copy(vec.begin(), vec.end(), back_inserter(tree));

  or from an std::istream:

\code
    copy(std::istream_iterator<T>(my_stream), std::istream_iterator<T>(), back_inserter(tree));
\endcode

  Note that the back_inserter must be used as there is no
  resize member in the parse_tree. back_inserter will use
  the push_back member from the parse_tree

*/

#include <vector>
#include <utility> // for swap

#ifdef _MSC_VER
#pragma warning(disable : 4786) // disable this nagging warning about the limitations of the mirkosoft debugger
#endif

namespace gp_parse_tree
{

#include "node_pool.h"

/// This ones defined because gcc does not always implement namespaces
template <class T>
inline void do_the_swap(T& a, T& b)
{
  T tmp = a;
  a = b;
  b = tmp;
}

template <class T> class parse_tree
{
  public :


class subtree
{

/**
    a bit nasty way to use a pool allocator (which would otherwise use slooow new and delete)
    @todo use the std::allocator interface
*/

#if (defined(__GNUC__) || defined(_MSC_VER)) && !(defined(_MT) || defined(MACOSX) || defined(__APPLE__)) // not multithreaded (or MACOSX - J. Eggermont)
    Node_alloc<T>         node_allocator;
    Tree_alloc<subtree>   tree_allocator;
#else
    Standard_Node_alloc<T>    node_allocator;
    Standard_alloc<subtree>   tree_allocator;
#endif

public :

    typedef subtree* iterator;
    typedef const subtree* const_iterator;

        /* Constructors, assignments */

    subtree(void) : content(node_allocator.allocate()), args(0), parent(0), _cumulative_size(0), _depth(0), _size(1)
                        {}
    subtree(const subtree& s)
        : content(node_allocator.allocate()),
          args(0),
          parent(0),
          _cumulative_size(1),
          _depth(1),
          _size(1)
    {
        copy(s);
    }

    subtree(const T& t) : content(node_allocator.allocate()), args(0), parent(0), _cumulative_size(0), _depth(0), _size(1)
                        { copy(t); }

    template <class It>
        subtree(It b, It e) : content(node_allocator.allocate()), args(0), parent(0), _cumulative_size(0), _depth(0), _size(1)
    { // initialize in prefix order for efficiency reasons
        init(b, --e);
    }

    virtual ~subtree(void)    { tree_allocator.deallocate(args, arity()); node_allocator.deallocate(content); }

    subtree& operator=(const subtree& s)
        {
            if (s.get_root() == get_root())
        { // from the same tree, maybe a child. Don't take any chances
            subtree anotherS = s;
            return copy(anotherS);
        }

                copy(s);
        updateAfterInsert();
        return *this;
        }

    subtree& operator=(const T& t)       { copy(t); updateAfterInsert(); return *this; }

        /* Access to the nodes */

    T&       operator*(void)             { return *content; }
    const T& operator*(void) const       { return *content; }
    T*       operator->(void)            { return content; }
    const T* operator->(void) const      { return content; }

        /* Equality, inequality check, Node needs to implement operator== */

        bool operator==(const subtree& other) const
        {
                if (! (*content == *other.content))
                        return false;

                for (int i = 0; i < arity(); i++)
                {
                        if (!(args[i] == other.args[i]))
                                return false;
                }

                return true;
        }

        bool operator !=(const subtree& other) const
        {
                return !operator==(other);
        }

        /* Arity */
    int arity(void) const                { return content->arity(); }

        /* Evaluation with an increasing amount of user defined arguments */
        template <class RetVal>
        void apply(RetVal& v) const                              { (*content)(v, begin()); }

        template <class RetVal, class It>
        void apply(RetVal& v, It values) const
    {
        (*content)(v, begin(), values);
    }

    template <class RetVal, class It>
    void apply_mem_func(RetVal& v, It misc, void (T::* f)(RetVal&, typename subtree::iterator, It))
    {
        (content->*f)(v, begin(), misc);
    }


/*	template <class RetVal, class It, class It2>
        void apply(RetVal& v, It values, It2 moreValues) const
                { (*content)(v, begin(), values, moreValues); }

        template <class RetVal, class It, class It2, class It3>
        void apply(RetVal& v, It values, It2 moreValues, It3 evenMoreValues) const
                { (*content)(v, begin(), values, moreValues, evenMoreValues); }
*/

    template <class Pred>
    void find_nodes(std::vector<subtree*>& result, Pred& p)
    {
        if (p(*content))
        {
            result.push_back(this);
        }

        for (int i = 0; i < arity(); ++i)
        {
            args[i].find_nodes(result, p);
        }
    }

    template <class Pred>
    void find_nodes(std::vector<const subtree*>& result, Pred& p) const
    {
        if (p(*content))
        {
            result.push_back(this);
        }

        for (int i = 0; i < arity(); ++i)
        {
            args[i].find_nodes(result, p);
        }
    }

        /* Iterators */

    iterator begin(void)              { return args; }
    const_iterator begin(void) const  { return args; }

    iterator end(void)                { return args + arity(); }
    const_iterator end(void) const    { return args + arity(); }

        subtree& operator[](int i)                { return *(begin() + i); }
        const subtree& operator[](int i) const { return *(begin() + i); }

        /* Some statistics */

    size_t size(void) const { return _size; }

    size_t cumulative_size(void) const { return _cumulative_size; }
    size_t depth(void) const           { return _depth; }

    const subtree& select_cumulative(size_t which) const
    { return imp_select_cumulative(which); }

    subtree& select_cumulative(size_t which)
    { return const_cast<subtree&>(imp_select_cumulative(which)); }

    subtree& get_node(size_t which)
    { return const_cast<subtree&>(imp_get_node(which));}
    const subtree& get_node(size_t which) const
    { return imp_get_node(which); }

    subtree*       get_parent(void)        { return parent; }
    const subtree* get_parent(void) const  { return parent; }

        void clear(void)
        { tree_allocator.deallocate(args, arity()); args = 0; *content = T(); parent = 0; _cumulative_size = 0; _depth = 0; _size = 0; }

        void swap(subtree& y)
        {
        do_the_swap(content, y.content);
        do_the_swap(args, y.args);

        adopt();
        y.adopt();

        do_the_swap(parent, y.parent);

        do_the_swap(_cumulative_size, y._cumulative_size);
        do_the_swap(_depth, y._depth);
        do_the_swap(_size, y._size);
                updateAfterInsert();
        }

protected :

        virtual void updateAfterInsert(void)
    {
        _depth = 0;
        _size = 1;
        _cumulative_size = 0;

        for (iterator it = begin(); it != end(); ++it)
        {
            _size += it->size();
            _cumulative_size += it->_cumulative_size;
            _depth = it->_depth > _depth? it->_depth: _depth;
        }
        _cumulative_size += _size;
        _depth++;

        if (parent)
            parent->updateAfterInsert();
    }

private :

    const subtree& imp_select_cumulative(size_t which) const
    {
        if (which >= (_cumulative_size - size()))
            return *this;
        // else

        for (int i = arity() - 1; i >= 0; --i)
                {
            if (which < args[i]._cumulative_size)
                return args[i].imp_select_cumulative(which);
            which -= args[i]._cumulative_size;
        }

        return *this; // error!
    }

    const subtree& imp_get_node(size_t which) const
    {
        if (which == size() - 1)
            return *this;

        for (int i = arity() - 1; i >= 0; --i)
                {
            unsigned c_size = args[i].size();
            if (which < c_size)
                return args[i].imp_get_node(which);
            which -= c_size;
        }

        return *this; // error!
    }

    const subtree* get_root(void) const
    {
        if (parent == 0)
            return this;
        // else

        return parent->get_root();
    }
    subtree& copy(const subtree& s)
    {
                int old_arity = arity();

                int new_arity = s.arity();

                if (new_arity != old_arity)
                {
                        tree_allocator.deallocate(args, old_arity);

            args = tree_allocator.allocate(new_arity);
                }

                switch(new_arity)
                {
                case 3 : args[2].copy(s.args[2]); args[2].parent = this; // no break!
                case 2 : args[1].copy(s.args[1]); args[1].parent = this;
                case 1 : args[0].copy(s.args[0]); args[0].parent = this;
                case 0 : break;
                default :
                        {
                                for (int i = 0; i < new_arity; ++i)
                                {
                                        args[i].copy(s.args[i]);
                    args[i].parent = this;
                                }
                        }
                }

                *content = *s.content;
        _size = s._size;
        _depth = s._depth;
        _cumulative_size = s._cumulative_size;

        return *this;
    }

    subtree& copy(const T& t)
    {
        int oldArity = arity();

                if (content != &t)
            *content = t;
                else
                        oldArity = -1;

                int ar = arity();

                if (ar != oldArity)
                {
            if (oldArity != -1)
                            tree_allocator.deallocate(args, oldArity);

            args = tree_allocator.allocate(ar);

            //if (ar > 0)
                        //	args = new subtree [ar];
                        //else
                        //	args = 0;
                }

        adopt();
        updateAfterInsert();
                return *this;
    }

        void disown(void)
        {
                switch(arity())
                {
                case 3 : args[2].parent = 0; // no break!
                case 2 : args[1].parent = 0;
                case 1 : args[0].parent = 0; break;
                case 0 : break;
                default :
                        {
                                for (iterator it = begin(); it != end(); ++it)
                                {
                                        it->parent = 0;
                                }
                        }
                }

        }

    void adopt(void)
    {
                switch(arity())
                {
                case 3 : args[2].parent = this; // no break!
                case 2 : args[1].parent = this;
                case 1 : args[0].parent = this; break;
                case 0 : break;
                default :
                        {
                                for (iterator it = begin(); it != end(); ++it)
                                {
                                        it->parent = this;
                                }
                        }
                }
    }

    template <class It>
    void init(It b, It& last)
    {
        *this = *last;

#ifndef NDEBUG
        if (last == b && arity() > 0)
        {
            throw "subtree::init()";
        }
#endif

        for (int i = 0; i < arity(); ++i)
        {
            args[i].parent = 0;
            args[i].init(b, --last);
            args[i].parent = this;
        }

        updateAfterInsert();
    }

    T* content;
    subtree*			args;
    subtree*			parent;

        size_t _cumulative_size;
    size_t _depth;
    size_t _size;
};

// Continuing with parse_tree

    typedef T value_type;

        /* Constructors and Assignments */

    parse_tree(void) : _root(), pushed() {}
    parse_tree(const parse_tree& org) : _root(org._root), pushed(org.pushed) { }
    parse_tree(const subtree& sub) : _root(sub), pushed() { }

    template <class It>
        parse_tree(It b, It e) : _root(b, e), pushed() {}

    virtual ~parse_tree(void) {}

    parse_tree& operator=(const parse_tree& org) { return copy(org); }
        parse_tree& operator=(const subtree& sub)
                { return copy(sub); }


        /* Equality and inequality */

        bool operator==(const parse_tree& other) const
        { return _root == other._root; }

        bool operator !=(const parse_tree& other) const
        { return !operator==(other); }

        /* Simple tree statistics */

    size_t size(void) const  { return _root.size(); }
        size_t depth(void) const { return _root.depth(); }
    void clear(void)        { _root.clear(); pushed.resize(0); }

        /* Evaluation (application), with an increasing number of user defined arguments */

        template <class RetVal>
        void apply(RetVal& v) const
                { _root.apply(v); }

        template <class RetVal, class It>
                void apply(RetVal& v, It varValues) const
                { _root.apply(v, varValues); }

    template <class RetVal, class It>
        void apply_mem_func(RetVal& v, It misc, void (T::* f)(RetVal&, typename subtree::iterator, It))
    {
        _root.apply_mem_func(v, misc, f);
    }

        //template <class RetVal, class It, class It2>
        //	void apply(RetVal& v, It varValues, It2 moreValues) const
        //	{ _root.apply(v, varValues, moreValues); }

        //template <class RetVal, class It, class It2, class It3>
        //	void apply(RetVal& v, It varValues, It2 moreValues, It3 evenMoreValues) const
        //	{ _root.apply(v, varValues, moreValues, evenMoreValues); }

    template <class Pred>
    void find_nodes(std::vector<subtree*>& result, Pred& p)
    {
        _root.find_nodes(result, p);
    }

    template <class Pred>
    void find_nodes(std::vector<const subtree*>& result, Pred& p) const
    {
        _root.find_nodes(p);
    }

        /* Customized Swap */
        void swap(parse_tree<T>& other)
        {
                do_the_swap(pushed, other.pushed);
                _root.swap(other._root);
        }

        /* Definitions of the iterators */

    class base_iterator
    {
    public :

        base_iterator() {}
        base_iterator(subtree* n)  { node = n; }

        base_iterator& operator=(const base_iterator& org)
        { node = org.node; return *this; }

        bool operator==(const base_iterator& org) const
        { return node == org.node; }
        bool operator!=(const base_iterator& org) const
        { return !operator==(org); }

                base_iterator operator+(size_t n) const
                {
                        base_iterator tmp = *this;

                        for(;n != 0; --n)
                        {
                                ++tmp;
                        }

                        return tmp;
                }

                base_iterator& operator++(void)
        {
            subtree* parent = node->get_parent();

            if (parent == 0)
            {
                node = 0;
                return *this;
            }
            // else
            typename subtree::iterator it;
            for (it = parent->begin(); it != parent->end(); ++it)
            {
                if (node == &(*it))
                    break;
            }

            if (it == parent->begin())
                node = parent;
            else
            {
                node = &(--it)->get_node(0);
            }

            return *this;
        }

        base_iterator operator++(int)
        {
            base_iterator tmp = *this;
            operator++();
            return tmp;
        }

    protected :
        subtree* node;
    };

    class iterator : public base_iterator
    {
    public:

        using base_iterator::node;

        typedef std::forward_iterator_tag iterator_category;
        typedef subtree value_type;
        typedef size_t distance_type;
        typedef size_t difference_type;
        typedef subtree* pointer;
        typedef subtree& reference;

        iterator() : base_iterator() {}
        iterator(subtree* n): base_iterator(n)  {}
        iterator& operator=(const iterator& org)
        { base_iterator::operator=(org); return *this; }

        subtree& operator*(void)  { return *node; }
        subtree* operator->(void) { return node; }
    };



    class embedded_iterator : public base_iterator
    {
    public:

        using base_iterator::node;

        typedef std::forward_iterator_tag iterator_category;
        typedef T value_type;
        typedef size_t distance_type;
        typedef size_t difference_type;
        typedef T* pointer;
        typedef T& reference;

        embedded_iterator() : base_iterator() {}
        embedded_iterator(subtree* n): base_iterator(n)  {}
        embedded_iterator& operator=(const embedded_iterator& org)
        { base_iterator::operator=(org); return *this; }

        T&  operator*(void)  { return **node; }
        T* operator->(void) { return &**node; }
    };

    class base_const_iterator
    {
    public:

        base_const_iterator() {}
        base_const_iterator(const subtree* n)  { node = n; }

        base_const_iterator& operator=(const base_const_iterator& org)
        { node = org.node; return *this; }

        bool operator==(const base_const_iterator& org) const
        { return node == org.node; }
        bool operator!=(const base_const_iterator& org) const
        { return !operator==(org); }

        base_const_iterator& operator++(void)
        {
            const subtree* parent = node->get_parent();

            if (parent == 0)
            {
                node = 0;
                return *this;
            }
            // else
            typename subtree::const_iterator it;

            for (it = parent->begin(); it != parent->end(); ++it)
            {
                if (node == &(*it))
                    break;
            }

            if (it == parent->begin())
                node = parent;
            else
                node = &(--it)->get_node(0);
            return *this;
        }

        base_const_iterator operator++(int)
        {
            base_const_iterator tmp = *this;
            operator++();
            return tmp;
        }

    protected :

        const subtree* node;
    };

    class const_iterator : public base_const_iterator
    {
    public:

        using base_const_iterator::node;

        typedef std::forward_iterator_tag iterator_category;
        typedef const subtree value_type;
        typedef size_t distance_type;
        typedef size_t difference_type;
        typedef const subtree* pointer;
        typedef const subtree& reference;

        const_iterator() : base_const_iterator() {}
        const_iterator(const subtree* n): base_const_iterator(n)  {}
        const_iterator& operator=(const const_iterator& org)
        { base_const_iterator::operator=(org); return *this; }

        const subtree& operator*(void)  { return *node; }
        const subtree* operator->(void) { return node; }
    };

    class embedded_const_iterator : public base_const_iterator
    {
    public:

        using base_const_iterator::node;

                typedef std::forward_iterator_tag iterator_category;
                typedef const T value_type;
                typedef size_t distance_type;
                typedef size_t difference_type;
                typedef const T* pointer;
                typedef const T& reference;

        embedded_const_iterator() : base_const_iterator() {}
        embedded_const_iterator(const subtree* n): base_const_iterator(n)  {}
        embedded_const_iterator& operator=(const embedded_const_iterator& org)
        { base_const_iterator::operator=(org); return *this; }

                embedded_const_iterator operator+(size_t n) const
                {
                        embedded_const_iterator tmp = *this;

                        for(;n != 0; --n)
                        {
                                ++tmp;
                        }

                        return tmp;
                }

        const T&  operator*(void)  const { return **node; }
        const T*  operator->(void) const { return node->operator->(); }
    };

        /* Iterator access */

    iterator begin(void)            { return iterator(&operator[](0)); }
    const_iterator begin(void) const { return const_iterator(&operator[](0)); }
    iterator end(void)               { return iterator(0); }
    const_iterator end(void) const   { return const_iterator(0);}

    embedded_iterator       ebegin(void)       { return embedded_iterator(&operator[](0)); }
    embedded_const_iterator ebegin(void) const { return embedded_const_iterator(&operator[](0)); }
    embedded_iterator       eend(void)         { return embedded_iterator(0); }
    embedded_const_iterator eend(void) const   { return embedded_const_iterator(0);}

    bool empty(void) const { return size() == 0; }
    bool valid(void) const { return pushed.empty(); }

        /* push_back */

        void push_back(const parse_tree<T>& tree)
        {
                if (!empty())
                        pushed.push_back(_root);

                _root = tree.back();
        }

    void push_back(const T& t)
    {
        if (!empty())
            pushed.push_back(_root);

                _root = t;

        for (typename subtree::iterator it = _root.begin(); it != _root.end(); it++)
        {
                        *it = pushed.back();
            pushed.pop_back();
        }

    }

        /* Access to subtrees */

    subtree& back(void)              { return _root; }
    const subtree& back(void) const  { return _root; }
    subtree& root(void)              { return _root; }
    const subtree& root(void) const  { return _root; }

    subtree& front(void)              { return _root[0]; }
    const subtree& front(void) const  { return _root[0]; }

    subtree& operator[](size_t i)
    { return const_cast<subtree&>(_root.get_node(i)); }
    const subtree& operator[](size_t i) const
    { return _root.get_node(i); }

    subtree& get_cumulative(size_t i)
    { return const_cast<subtree&>(_root.get_cumulative(i)); }
    const subtree& get_cumulative(size_t i) const
    { return get_cumulative(i); }

    private :

    parse_tree& copy(const parse_tree& org)
    {
        _root = org._root;
        pushed = org.pushed;

        return *this;
    }

        parse_tree& copy(const subtree& sub)
    { _root = sub; pushed.resize(0); return *this; }

    subtree _root;
    std::vector<subtree > pushed;
}; // end class parse_tree


} // end namespace gp_parse_tree

namespace std
{ // for use with stlport on MSVC

template <class T> inline
std::forward_iterator_tag iterator_category(typename gp_parse_tree::parse_tree<T>::embedded_iterator)
{
        return std::forward_iterator_tag();
}

template <class T> inline
ptrdiff_t*  distance_type(typename gp_parse_tree::parse_tree<T>::embedded_iterator)
{
        return 0;
}

template <class T> inline
std::forward_iterator_tag iterator_category(typename gp_parse_tree::parse_tree<T>::iterator)
{
        return std::forward_iterator_tag();
}

template <class T> inline
ptrdiff_t*  distance_type(typename gp_parse_tree::parse_tree<T>::iterator)
{
        return 0;
}

/* Put customized swaps also in std...

template<class T> inline
void swap(gp_parse_tree::parse_tree<T>& a, gp_parse_tree::parse_tree<T>& b)
{
    a.swap(b);
}

template<class T> inline
void iter_swap(std::vector<gp_parse_tree::parse_tree<T> >::iterator a, std::vector<gp_parse_tree::parse_tree<T> > b)
{
    a->swap(*b);
}*/


} // namespace std


#endif
