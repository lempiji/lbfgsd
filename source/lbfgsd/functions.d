module lbfgsd.functions;

private import lbfgsd.math;

struct RosenBrockFunction
{
    T opCall(T)(in T[] x) @safe @nogc const pure nothrow
    {
        auto y = T(0);
        foreach (i; 0 .. x.length - 1)
        {
            y += square(1 - x[i]) + 100 * square(x[i + 1] - square(x[i]));
        }
        return y;
    }
}
@safe pure nothrow unittest
{
    RosenBrockFunction fn;

    auto y = fn([1.0, 1.0, 1.0]);
    assert(y == 0);
}
