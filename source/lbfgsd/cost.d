module lbfgsd.cost;

private import lbfgsd.autodiff;

/**
 *Basic Cost Function
 */
abstract class CostFunction(T, size_t NInput)
{
    abstract T evaluate(const scope T[] x, scope T[] gradient)
    in
    {
        assert(x.length == NInput);
        assert(gradient.length == 0 || gradient.length == NInput);
    }
    do
    {
        return T.init;
    }
}

/**
 *Automatic Differentation
 */
final class AutoDiffCostFunction(TFunc, T, size_t NInput) : CostFunction!(T, NInput)
{
public:
    this(TFunc func)
    {
        _func = func;
        _x.length = NInput;
    }

public:
    override T evaluate(const scope T[] x, scope T[] gradient)
    {
        if (gradient == null || gradient.length == 0)
            return _func(x);

        assert(gradient.length == NInput);
        foreach (i; 0 .. NInput)
        {
            _x[i].a = x[i];
            _x[i].d[] = T(0);
            _x[i].d[i] = T(1);
        }

        auto y = _func(_x);
        gradient[] = y.d[];
        return y.a;
    }

private:
    TFunc _func;
    Variable!(T, NInput)[] _x;
}

unittest
{
    import lbfgsd.functions;

    RosenBrockFunction fn;
    CostFunction!(double, 2) cost = new AutoDiffCostFunction!(RosenBrockFunction, double, 2)(fn);

    auto x = new double[2];
    x[0] = 1;
    x[1] = 1;
    auto y = cost.evaluate(x, null);
    assert(y == 0);
}
unittest
{
    import lbfgsd.functions;

    RosenBrockFunction fn;
    CostFunction!(double, 2) cost = new AutoDiffCostFunction!(RosenBrockFunction, double, 2)(fn);

    auto x = new double[2];
    x[0] = 1;
    x[1] = 1;
    auto g = new double[2];
    g[] = 0;
    auto y = cost.evaluate(x, g);
    assert(y == 0);
    assert(g[0] == 0);
    assert(g[1] == 0);
}

/**
 *Numeric Differentation
 */
final class NumericDiffCostFunction(TFunc, T, size_t NInput) : CostFunction!(T, NInput)
{
public:
    this(TFunc func)
    {
        _func = func;
        _x.length = NInput;
    }

public:
    override T evaluate(const scope T[] x, scope T[] gradient)
    {
        if (gradient == null || gradient.length == 0)
            return _func(x);

        assert(gradient.length == NInput);
        enum eps = 1e-8;
        enum neps = 0.5 / eps;
        _x[] = x[];
        foreach (i; 0 .. NInput)
        {
            auto org = _x[i];
            _x[i] = org + eps;
            auto f1 = _func(_x);
            _x[i] = org - eps;
            auto f2 = _func(_x);
            _x[i] = org;
            gradient[i] = (f1 - f2) * neps;
        }
        return _func(x);
    }

private:
    TFunc _func;
    T[] _x;
}

unittest
{
    import lbfgsd.functions;

    RosenBrockFunction fn;
    CostFunction!(double, 2) cost = new NumericDiffCostFunction!(RosenBrockFunction, double, 2)(fn);

    auto x = new double[2];
    x[0] = 1;
    x[1] = 1;
    auto y = cost.evaluate(x, null);
    assert(y == 0);
}

unittest
{
    import lbfgsd.functions;
    import std.math: approxEqual;

    RosenBrockFunction fn;
    CostFunction!(double, 2) cost = new NumericDiffCostFunction!(RosenBrockFunction, double, 2)(fn);

    auto x = new double[2];
    x[0] = 1;
    x[1] = 1;
    auto g = new double[2];
    g[] = 0;
    auto y = cost.evaluate(x, g);
    assert(y == 0);
    assert(approxEqual(g[0], 0));
    assert(approxEqual(g[1], 0));
}
