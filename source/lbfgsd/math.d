module lbfgsd.math;

private static import std.math;

private static import std.numeric;

private import lbfgsd.autodiff;

/// optimized (x * x)
auto square(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = (x.a + x.a) * x.d[];
        y.a = x.a * x.a;
        return y;
    }
    else
        return x * x;
}

unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    auto y = Var(2.0, 1);

    auto sx = square(x);
    assert(sx.a == 1);
    assert(sx.d[0] == 2);
    assert(sx.d[1] == 0);

    auto sy = square(y);
    assert(sy.a == 4);
    assert(sy.d[0] == 0);
    assert(sy.d[1] == 4);

    auto z = 3.0;
    auto sz = square(z);
    assert(sz == 9);
}

auto dotProduct(T, U)(const scope T[] a, const scope U[] b)
{
    static if (is(T : Variable!(S, N), S, size_t N) || is(U : Variable!(W, M), W, size_t M))
    {
        assert(a.length == b.length);

        import std.traits;

        alias Q = Unqual!(typeof(T.init * U.init));
        auto sum0 = Q(0), sum1 = Q(0);

        const all_endp = a.length;
        const smallblock_endp = all_endp & ~3;
        const bigblock_endp = all_endp & ~15;

        size_t i = 0;
        auto ap = a[];
        auto bp = b[];
        for (; i != bigblock_endp; i += 16, ap = ap[16 .. $], bp = bp[16 .. $])
        {
            sum0 += ap[0] * bp[0];
            sum1 += ap[1] * bp[1];
            sum0 += ap[2] * bp[2];
            sum1 += ap[3] * bp[3];
            sum0 += ap[4] * bp[4];
            sum1 += ap[5] * bp[5];
            sum0 += ap[6] * bp[6];
            sum1 += ap[7] * bp[7];
            sum0 += ap[8] * bp[8];
            sum1 += ap[9] * bp[9];
            sum0 += ap[10] * bp[10];
            sum1 += ap[11] * bp[11];
            sum0 += ap[12] * bp[12];
            sum1 += ap[13] * bp[13];
            sum0 += ap[14] * bp[14];
            sum1 += ap[15] * bp[15];
        }

        for (; i != smallblock_endp; i += 4, ap = ap[4 .. $], bp = bp[4 .. $])
        {
            sum0 += ap[0] * bp[0];
            sum1 += ap[1] * bp[1];
            sum0 += ap[2] * bp[2];
            sum1 += ap[3] * bp[3];
        }

        for (; i != all_endp; ++i, ap = ap[1 .. $], bp = bp[1 .. $])
        {
            sum0 += ap[0] * bp[0];
        }

        return sum0 + sum1;
    }
    else
        return std.numeric.dotProduct(a, b);
}

unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    auto ys = new Var[3];
    foreach (i; 0 .. xs.length)
    {
        ys[i] = xs[i] = Var(i, i);
    }
    auto z = dotProduct(xs, ys);
    assert(z.a == 5);
    assert(z.d[0] == 0);
    assert(z.d[1] == 2);
    assert(z.d[2] == 4);
}

unittest
{
    auto xs = new double[3];
    auto ys = new double[3];
    foreach (i; 0 .. xs.length)
    {
        ys[i] = xs[i] = i;
    }
    auto z = dotProduct(xs, ys);
    assert(z == 5);
}

unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    auto ys = new double[3];
    foreach (i; 0 .. xs.length)
    {
        xs[i] = Var(i, i);
        ys[i] = i;
    }
    auto z = dotProduct(xs, ys);
    assert(z.a == 5);
    assert(z.d[0] == 0);
    assert(z.d[1] == 1);
    assert(z.d[2] == 2);

    auto w = dotProduct(ys, xs);
    assert(z.a == w.a);
    assert(z.d[0] == w.d[0]);
    assert(z.d[1] == w.d[1]);
    assert(z.d[2] == w.d[2]);
}

unittest
{
    alias Var = Variable!(double, 1000);
    auto xs = new Var[1000];
    foreach (i; 0 .. xs.length)
        xs[i] = Var(i, i);

    auto y = dotProduct(xs, xs);
    assert(y.a == 332833500);
    foreach (i; 0 .. xs.length)
        assert(y.d[i] == 2 * i);
}

T sum(T)(const scope T[] xs) @safe @nogc pure nothrow
{
    assert(xs.length > 0);
    T y = xs[0];
    foreach (i; 1 .. xs.length)
        y += xs[i];
    return y;
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    xs[0] = Var(0.2, 0);
    xs[1] = Var(0.4, 1);
    xs[2] = Var(0.6, 2);

    auto sx = sum(xs);
    assert(std.math.approxEqual(sx.a, 1.2));
    assert(sx.d[0] == 1);
    assert(sx.d[1] == 1);
    assert(sx.d[2] == 1);

    auto ys = new double[3];
    ys[0] = 0.2;
    ys[1] = 0.4;
    ys[2] = 0.6;
    auto sy = sum(ys);
    assert(std.math.approxEqual(sy, 1.2));
}

T sumsq(T)(const scope T[] xs) @safe @nogc pure nothrow
{
    assert(xs.length > 0);
    T y = square(xs[0]);
    foreach (i; 1 .. xs.length)
        y += square(xs[i]);
    return y;
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    xs[0] = Var(0.2, 0);
    xs[1] = Var(0.4, 1);
    xs[2] = Var(0.6, 2);

    auto sx = sumsq(xs);
    assert(sx.a == 0.56);
    assert(sx.d[0] == 0.4);
    assert(sx.d[1] == 0.8);
    assert(sx.d[2] == 1.2);

    auto ys = new double[3];
    ys[0] = 0.2;
    ys[1] = 0.4;
    ys[2] = 0.6;

    auto sy = sumsq(ys);
    assert(sy == 0.56);
}

T sumxmy2(T, U)(const scope T[] xs, const scope U[] ys) @safe @nogc pure nothrow
{
    assert(xs.length > 0);
    assert(xs.length == ys.length);

    T sum = square(xs[0] - ys[0]);
    foreach (i; 1 .. xs.length)
        sum += square(xs[i] - ys[i]);
    return sum;
}

unittest
{
    alias Var = Variable!(double, 3);
    auto xs = new Var[3];
    xs[0] = Var(0.2, 0);
    xs[1] = Var(0.4, 1);
    xs[2] = Var(0.6, 2);
    auto ys = new Var[3];
    ys[0] = Var(0.1, 0);
    ys[1] = Var(0.2, 1);
    ys[2] = Var(0.3, 2);

    auto s = sumxmy2(xs, ys);
    assert(s == square(xs[0] - ys[0]) + square(xs[1] - ys[1]) + square(xs[2] - ys[2]));
}

T sqrt(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.sqrt(x.a);
        T y = void;
        y.d[] = x.d[] * (0.5 / t);
        y.a = t;
        return y;
    }
    else
        return std.math.sqrt(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = sqrt(x);
    auto w = sqrt(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1 / (2 * w)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T exp(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.exp(x.a);
        T y = void;
        y.d[] = t * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.exp(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = exp(x);
    auto w = exp(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], w));
    assert(std.math.approxEqual(z.d[1], 0));
}

T log(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / x.a;
        y.a = std.math.log(x.a);
        return y;
    }
    else
        return std.math.log(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = log(x);
    auto w = log(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.5));
    assert(std.math.approxEqual(z.d[1], 0));
}

T sin(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.cos(x.a) * x.d[];
        y.a = std.math.sin(x.a);
        return y;
    }
    else
        return std.math.sin(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = sin(x);
    auto w = sin(y);

    assert(std.math.approxEqual(z.a, 0.909297427));
    assert(std.math.approxEqual(z.d[0], -0.416146837));
    assert(std.math.approxEqual(z.d[1], 0));
    assert(std.math.approxEqual(w, 0.909297427));
}

T cos(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = -std.math.sin(x.a) * x.d[];
        y.a = std.math.cos(x.a);
        return y;
    }
    else
        return std.math.cos(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = cos(x);
    auto w = cos(y);

    assert(std.math.approxEqual(z.a, -0.416146837));
    assert(std.math.approxEqual(z.d[0], -0.909297427));
    assert(std.math.approxEqual(z.d[1], 0));
    assert(std.math.approxEqual(w, -0.416146837));
}

T tan(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        auto t = std.math.tan(x.a);
        y.d[] = (1 + t * t) * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.tan(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(2.0, 0);
    double y = 2.0;

    auto z = tan(x);
    auto w = tan(y);

    assert(std.math.approxEqual(z.a, w));
    assert(std.math.approxEqual(z.d[0], 1 + w * w));
    assert(std.math.approxEqual(z.d[1], 0));
}

T sinh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.cosh(x.a) * x.d[];
        y.a = std.math.sinh(x.a);
        return y;
    }
    else
        return std.math.sinh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = sinh(x);
    auto w = sinh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], std.math.cosh(1.0)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T cosh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = std.math.sinh(x.a) * x.d[];
        y.a = std.math.cosh(x.a);
        return y;
    }
    else
        return std.math.cosh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = cosh(x);
    auto w = cosh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], std.math.sinh(1.0)));
    assert(std.math.approxEqual(z.d[1], 0));
}

T tanh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        const t = std.math.tanh(x.a);
        T y = void;
        y.d[] = (1 - t * t) * x.d[];
        y.a = t;
        return y;
    }
    else
        return std.math.tanh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.0, 0);
    double y = 1.0;

    auto z = tanh(x);
    auto w = tanh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1 - w * w));
    assert(std.math.approxEqual(z.d[1], 0));
}

T asinh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / std.math.sqrt(x.a * x.a + 1);
        y.a = std.math.asinh(x.a);
        return y;
    }
    else
        return std.math.asinh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(0.5, 0);
    double y = 0.5;

    auto z = asinh(x);
    auto w = asinh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.894427));
    assert(std.math.approxEqual(z.d[1], 0));
}

T acosh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / std.math.sqrt(x.a * x.a - 1);
        y.a = std.math.acosh(x.a);
        return y;
    }
    else
        return std.math.acosh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1.5, 0);
    double y = 1.5;

    auto z = acosh(x);
    auto w = acosh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 0.894427));
    assert(std.math.approxEqual(z.d[1], 0));
}

T atanh(T)(const scope T x) @safe @nogc pure nothrow
{
    static if (is(T : Variable!(U, N), U, size_t N))
    {
        T y = void;
        y.d[] = x.d[] / (1 - x.a * x.a);
        y.a = std.math.atanh(x.a);
        return y;
    }
    else
        return std.math.atanh(x);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(0.5, 0);
    double y = 0.5;

    auto z = atanh(x);
    auto w = atanh(y);

    assert(z.a == w);
    assert(std.math.approxEqual(z.d[0], 1.33333));
    assert(std.math.approxEqual(z.d[1], 0));
}

/// correlation coefficient
auto correl(T, U)(const scope T[] xs, const scope U[] ys)
{
    assert(xs.length == ys.length);
    import std.traits;

    alias Q = Unqual!(typeof(T.init * U.init));

    immutable N = xs.length;
    auto xa = sum(xs) / N;
    auto ya = sum(ys) / N;

    return correl(xs, ys, xa, ya);
}

@safe unittest
{
    alias Var = Variable!(double, 2);

    auto xa = Var(1, 0);
    auto ya = Var(1, 1);

    auto xs = new Var[3];
    auto ys = new Var[3];
    foreach (i, ref x; xs)
        x = xa * i;
    foreach (i, ref y; ys)
        y = ya * i;

    auto c = correl(xs, ys);
    import std.algorithm, std.math;

    assert(approxEqual(c.a, 1.0));
    assert(equal!approxEqual(c.d[], [0.0, 0.0]));
}

auto correl(T, U)(const scope T[] xs, const scope U[] ys, const scope T xa, const scope T ya)
{
    assert(xs.length == ys.length);
    import std.traits;

    alias Q = Unqual!(typeof(T.init * U.init));

    auto xv = T(0);
    auto yv = U(0);
    auto cv = Q(0);

    foreach (i; 0 .. xs.length)
    {
        auto tx = xs[i] - xa;
        auto ty = ys[i] - ya;
        cv += tx * ty;
        xv += square(tx);
        yv += square(ty);
    }
    return cv / (sqrt(xv) * sqrt(yv));
}

T sigmoid(T)(const scope T x)
{
    return T(1) / (1 + exp(-x));
}

T swish(T)(const scope T x)
{
	return x * sigmoid(x);
}

T softplus(T)(const scope T x)
{
	return log(1 + exp(x));
}

T mish(T)(const scope T x)
{
	return x * tanh(softplus(x));
}
