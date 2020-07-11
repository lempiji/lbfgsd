module lbfgsd.linesearch;

private import lbfgsd.cost;

enum LineSearchType
{
    Armijo,
    Wolfe,
    StrongWolfe
}

struct LineSearchOptions(T)
{
    LineSearchType type = LineSearchType.Armijo;
    bool bulkEvaluate = true; //only use by wolfe or strong wolfe

    size_t maxIterations = 20;

    T initialStepSize = 1.0;

    T wolfe = 0.9;
    T armijo = 1e-4;
}

struct LineSearchResult(T)
{
    bool success;
    size_t numIterations;
    T stepSize;
    T cost;
}

class BackTrackLineSearcher(T, size_t NInput)
{
    alias Cost = CostFunction!(T, NInput);
    alias Result = LineSearchResult!T;
    alias Options = LineSearchOptions!T;

public:
    ref Options options() @safe @nogc pure nothrow
    {
        return _options;
    }

public:
    void setCostFunction(Cost cost) @safe @nogc pure nothrow
    {
        _cost = cost;
    }

public:
    Result search(in T[] x, in T[] g, in T[] d, in T f0, T[] xn, T[] gn)
    {
        return search(x, g, d, f0, xn, gn, _options.initialStepSize);
    }

    Result search(in T[] x, in T[] g, in T[] d, in T f0, T[] xn, T[] gn, T step)
    {
        Result result = void;
        result.numIterations = 0;
        result.success = false;
        result.stepSize = step;

        import std.numeric : dotProduct;
        immutable ginit = dotProduct(g, d);
        immutable c_armijo = _options.armijo * ginit;
        immutable c_wolfe = _options.wolfe * ginit;
        immutable type = _options.type;
        immutable bulk = _options.bulkEvaluate;

        enum inc = 2.1;
        enum dec = 0.5;
        T fx;
        foreach (i; 0 .. _options.maxIterations)
        {
            ++result.numIterations;

            xn[] = x[] + step * d[];
            fx = bulk
                ? _cost.evaluate(xn, gn)
                : _cost.evaluate(xn, null);

            //Armijo
            if (fx > f0 + step * c_armijo)
            {
                step *= dec;
                continue;
            }

            if (type == LineSearchType.Armijo)
            {
                if (!bulk) _cost.evaluate(xn, gn); //calc gradient when the Armijo method
                result.success = true;
                break;
            }

            //Wolfe
            if (!bulk) _cost.evaluate(xn, gn);

            const dg = dotProduct(gn, d);
            if (dg < c_wolfe)
            {
                step *= inc;
                continue;
            }
            if (type == LineSearchType.Wolfe)
            {
                result.success = true;
                break;
            }

            //Strong Wolfe
            if (dg > -c_wolfe)
            {
                step *= dec;
                continue;
            }
            if (type == LineSearchType.StrongWolfe)
            {
                result.success = true;
                break;
            }
        }
        //iteration is over
        result.stepSize = step;
        result.cost = fx;
        return result;
    }

private:
    Cost _cost;
    Options _options;
}
unittest
{
    static struct Func
    {
        T opCall(T)(in T[] x) @safe @nogc pure nothrow
        {
            import lbfgsd.math;
            auto t1 = x[0] - 1;
            auto t2 = x[1] + 10;
            return t1 * t1 + t2 * t2 + exp(x[0] + x[1]);
        }
    }

    Func fn;
    auto cost = new AutoDiffCostFunction!(Func, double, 2)(fn);

    auto searcher = new BackTrackLineSearcher!(double, 2);
    foreach (t; [LineSearchType.Armijo, LineSearchType.Wolfe, LineSearchType.StrongWolfe])
    {
        searcher.options.type = t;
        searcher.options.maxIterations = 5;
        searcher.setCostFunction(cost);

        auto x = new double[2];
        auto g = new double[2];
        auto xn = new double[2];
        auto gn = new double[2];
        auto d = new double[2];

        x[] = 0;
        auto f = cost.evaluate(x, g);
        d[] = -g[];

        auto result = searcher.search(x, g, d, f, xn, gn);

        assert(result.success);
        assert(result.numIterations <= 5);
        assert(result.stepSize > 0);
    }
}
unittest
{
    import lbfgsd.functions;
    RosenBrockFunction fn;
    auto cost = new AutoDiffCostFunction!(RosenBrockFunction, double, 2)(fn);

    auto searcher = new BackTrackLineSearcher!(double, 2);
    foreach (t; [LineSearchType.Armijo, LineSearchType.Wolfe, LineSearchType.StrongWolfe])
    {
        searcher.options.type = t;
        searcher.options.maxIterations = 5;
        searcher.options.bulkEvaluate = false;
        searcher.setCostFunction(cost);

        auto x = new double[2];
        auto g = new double[2];
        auto xn = new double[2];
        auto gn = new double[2];
        auto d = new double[2];

        x[] = 0;
        auto f = cost.evaluate(x, g);
        d[] = -g[];

        auto result = searcher.search(x, g, d, f, xn, gn);

        assert(result.success);
        assert(result.numIterations <= 5);
        assert(result.stepSize > 0);
    }
}
