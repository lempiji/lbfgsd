# numeric
Automatic differentiation and L-BFGS implementation in the D Programming Language.

```D
auto solver = new SimpleSolver!(double, 2);

struct Func
{
    T opCall(T)(in T[] x)
    {
        return x[0] * x[0] + x[1] * x[1];
    }
}

Func f;
solver.setAutoDiffCost(f);

auto x = new double[2];
x[] = 1;
solver.solve(x); //solve by L-BFGS
assert(equals(x, [0.0, 0.0]));
```
