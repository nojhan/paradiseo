
template< class EOT>
class eoEvalPrint: public eoEvalFunc<EOT>
{
    protected:
        std::ostream& _out;
        eoEvalFunc<EOT>& _eval;
        std::string _sep;

    public:

        eoEvalPrint(eoEvalFunc<EOT>& eval, std::ostream& out=std::cout, std::string sep="\n") :
            _out(out),
            _eval(eval),
            _sep(sep)
        {}

        void operator()( EOT& sol )
        {
            _eval(sol);
            _out << sol << _sep;
        }
};
