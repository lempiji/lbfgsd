module lbfgsd.autodiff;

enum isVariable(T) = is(T : Variable!(U, N), U, size_t N);

unittest
{
    static assert(isVariable!(Variable!(float, 3)));
    static assert(isVariable!(Variable!(double, 2)));
    static assert(!isVariable!float);
    static assert(!isVariable!double);
    static assert(!isVariable!string);
}

struct Variable(T, size_t N)
{
public:
    this(T val) @safe @nogc pure nothrow
    {
        d[] = T(0);
        a = val;
    }

    this(T val, size_t n) @safe @nogc pure nothrow
    {
        assert(n < N);

        d[] = T(0);
        d[n] = T(1);
        a = val;
    }

public:
    ref Variable opAssign(const scope T val) @safe @nogc pure nothrow
    {
        d[] = T(0);
        a = val;
        return this;
    }

    Variable opUnary(string op)() @safe @nogc const pure nothrow
    {
        static if (op == "-")
        {
            Variable t;
            t.d[] = -d[];
            t.a = -a;
            return t;
        }
        else
            static assert(false);
    }

    Variable opBinary(string op)(const scope Variable r) @safe @nogc const pure nothrow
    {
        Variable t;
        static if (op == "+")
        {
            t.d[] = d[] + r.d[];
            t.a = a + r.a;
        }
        else static if (op == "-")
        {
            t.d[] = d[] - r.d[];
            t.a = a - r.a;
        }
        else static if (op == "*")
        {
            t.d[] = d[] * r.a + a * r.d[];
            t.a = a * r.a;
        }
        else static if (op == "/")
        {
            const u = T(1) / r.a;
            t.d[] = (d[] - a * u * r.d[]) * u;
            t.a = a * u;
        }
        else
            static assert(false);
        return t;
    }

    Variable opBinary(string op)(const scope T r) @safe @nogc const pure nothrow
    {
        Variable t;
        static if (op == "+")
        {
            t.d[] = d[];
            t.a = a + r;
        }
        else static if (op == "-")
        {
            t.d[] = d[];
            t.a = a - r;
        }
        else static if (op == "*")
        {
            t.d[] = d[] * r;
            t.a = a * r;
        }
        else static if (op == "/")
        {
            t.d[] = d[] / r;
            t.a = a / r;
        }
        else
            static assert(false);
        return t;
    }

    Variable opBinaryRight(string op)(const scope T l) @safe @nogc const pure nothrow
    {
        Variable t;
        static if (op == "+")
        {
            t.d[] = d[];
            t.a = l + a;
        }
        else static if (op == "-")
        {
            t.d[] = -d[];
            t.a = l - a;
        }
        else static if (op == "*")
        {
            t.d[] = l * d[];
            t.a = l * a;
        }
        else static if (op == "/")
        {
            t.d[] = -l / d[];
            t.a = l / a;
        }
        return t;
    }

    ref Variable opOpAssign(string op)(const scope Variable r) @safe @nogc pure nothrow
    {
        static if (op == "+")
            return this = this + r;
        else static if (op == "-")
            return this = this - r;
        else static if (op == "*")
            return this = this * r;
        else static if (op == "/")
            return this = this / r;
        else
            static assert(false);
    }

    ref Variable opOpAssign(string op)(const scope T r) @safe @nogc pure nothrow
    {
        static if (op == "+")
            return this = this + r;
        else static if (op == "-")
            return this = this - r;
        else static if (op == "*")
            return this = this * r;
        else static if (op == "/")
            return this = this / r;
        else
            static assert(false);
    }

    bool opEquals(const scope T rhs) const
    {
        return a == rhs;
    }

    bool opEquals(const scope Variable rhs) const
    {
        return a == rhs.a;
    }

    int opCmp(const scope T rhs) const
    {
        const t = a - rhs;
        if (t == 0)
            return 0;
        if (t < 0)
            return -1;
        return 1;
    }

    int opCmp(const scope Variable rhs) const
    {
        const t = a - rhs.a;
        if (t == 0)
            return 0;
        if (t < 0)
            return -1;
        return 1;
    }

public:
    T[N] d;
    T a;
}

@safe pure nothrow unittest
{
    import std.algorithm;

    alias Var = Variable!(double, 3);
    auto x = Var(1);
    assert(x.a == 1);
    assert(equal(x.d[], [0.0, 0.0, 0.0]));
    x = 2;
    assert(x.a == 2);
    assert(equal(x.d[], [0.0, 0.0, 0.0]));
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1, 0);
    auto y = Var(1, 1);
    assert(x.d.length == 2);
    assert(x.d[0] == 1);
    assert(x.d[1] == 0);
    assert(y.d.length == 2);
    assert(y.d[0] == 0);
    assert(y.d[1] == 1);

    y = -x;
    assert(y.d[0] == -1);
    assert(y.d[1] == 0);
    assert(y.a == -1);

    x = 0;
    assert(x.d[0] == 0);
    assert(x.d[1] == 0);
    assert(x.a == 0);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1, 0);
    auto y = Var(2, 1);
    Var z;

    z = x + y;
    assert(z.d[0] == 1);
    assert(z.d[1] == 1);
    assert(z.a == x.a + y.a);

    z = x - y;
    assert(z.d[0] == 1);
    assert(z.d[1] == -1);
    assert(z.a == x.a - y.a);

    z = x * y;
    assert(z.d[0] == y.a);
    assert(z.d[1] == x.a);
    assert(z.a == x.a * y.a);

    z = x / y;
    assert(z.d[0] == 0.5);
    assert(z.d[1] == -0.25);
    assert(z.a == x.a / y.a);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 1);
    auto x = Var(1, 0);
    auto y = 2.0;
    Var z;

    z = x + y;
    assert(z.d[0] == 1);
    assert(z.a == x.a + y);

    z = x - y;
    assert(z.d[0] == 1);
    assert(z.a == x.a - y);

    z = x * y;
    assert(z.d[0] == y);
    assert(z.a == x.a * y);

    z = x / y;
    assert(z.d[0] == x.d[0] / y);
    assert(z.a == x.a / y);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 1);
    auto x = Var(1, 0);
    Var y;

    y = 1 + x;
    assert(y.d[0] == 1);
    assert(y.a == 1 + x.a);

    y = 1 - x;
    assert(y.d[0] == -1);
    assert(y.a == 1 - x.a);

    y = 2 * x;
    assert(y.d[0] == 2);
    assert(y.a == 2 * x.a);

    y = 2 / x;
    assert(y.d[0] == -2);
    assert(y.a == 2 / x.a);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1, 0);
    auto y = Var(2, 1);

    x += y;
    assert(x.d[0] == 1);
    assert(x.d[1] == 1);
    assert(x.a == 3);

    x -= y;
    assert(x.d[0] == 1);
    assert(x.d[1] == 0);
    assert(x.a == 1);

    x *= y;
    assert(x.d[0] == 2);
    assert(x.d[1] == 1);
    assert(x.a == 2);

    x /= y;
    assert(x.d[0] == 1);
    assert(x.d[1] == 0);
    assert(x.a == 1);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1, 0);

    x += 2;
    assert(x.d[0] == 1);
    assert(x.d[1] == 0);
    assert(x.a == 3);

    x -= 2;
    assert(x.d[0] == 1);
    assert(x.d[1] == 0);
    assert(x.a == 1);

    x *= 2;
    assert(x.d[0] == 2);
    assert(x.d[1] == 0);
    assert(x.a == 2);

    x /= 4;
    assert(x.d[0] == 0.5);
    assert(x.d[1] == 0);
    assert(x.a == 0.5);
}

@safe pure nothrow unittest
{
    alias Var = Variable!(double, 2);
    auto x = Var(1, 0);

    assert(x > 0);
    assert(x == 1);
    assert(x < 2);

    auto y = Var(2, 0);
    assert(x < y);

    assert(x > Var(0));
    assert(x >= Var(0));
    assert(x == Var(1));
    assert(x <= Var(2));
    assert(x < Var(2));
}
