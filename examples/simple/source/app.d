module app;

import std.stdio;
import std.typecons;
import lbfgsd;

void main()
{
    auto solver = new SimpleSolver!(float, 2);
    solver.options.linesearch.maxIterations = 1000;
    solver.options.maxIterations = 2000;
	//solver.options.estimateStepSize = false;

    LeastSquareLossFunc fn;
	foreach (i; 0 .. 20)
	{
		import std.random : uniform;
		// (x, 3x - 1 + random)
		fn.data ~= tuple(float(i), float(3f * i - 1 + uniform(-0.1f, 0.1f)));
	}
    solver.setAutoDiffCost(fn);

    auto x = new float[2];
	x[] = 0;
    auto result = solver.solve(x);

	writeln("iterations : ", result.iterations.length);
	writeln("last 10 steps :");
	import std.range : tail;
	foreach (i, iter; result.iterations.tail(10))
	{
		writeln(i, " : ", iter);
	}
	writeln("-----");
	writeln(result.status);
	writeln(x);
	writeln(fn(x));
	//writeln(result.lastCost);
}

struct LeastSquareLossFunc
{
	Tuple!(float, float)[] data;

	auto opCall(T)(const scope T[] ab)
	{
		const A = ab[0];
		const B = ab[1];
		auto result = T(0);
		foreach (d; data)
		{
			const t = A * d[0] + B;
			result += square(d[1] - t);
		}
		return result;
	}
}
