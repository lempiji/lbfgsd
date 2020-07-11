module app;

import std.stdio;
import std.typecons;
import lbfgsd;

void main()
{
	auto solver = new SimpleSolver!(float, 4);
    solver.options.linesearch.maxIterations = 1000;
	solver.options.maxIterations = 10_000;

	import std.random : uniform;

	LogLikehoodCostFunc fn;
	foreach (i; 0 .. 20)
		fn.data ~= uniform(-3.0f, 4.0f);
	foreach (i; 0 .. 10)
		fn.data ~= uniform(-1.0f, 1.0f);

	solver.setAutoDiffCost(fn);

	auto x = new float[4];
	x[] = 0;
	auto result = solver.solve(x);

	writeln("iterations : ", result.iterations.length);
	writeln("-----");
	const gamma = x[0];
	const delta = exp(x[1]);
	const lambda = exp(x[2]);
	const xi = x[3];
	writeln(result.status);
	writefln!"%.2f, %.2f, %.2f, %.2f"(gamma, delta, lambda, xi);
	writeln(result.lastCost);

	import lbfgsd.distribution;

	auto dist = johnsonSUDistribution(gamma, delta, lambda, xi);
	writefln!" 5%% : %.4f"(dist.cdfInverse(0.05f));
	writefln!"25%% : %.4f"(dist.cdfInverse(0.25f));
	writefln!"50%% : %.4f"(dist.cdfInverse(0.50f));
	writefln!"75%% : %.4f"(dist.cdfInverse(0.75f));
	writefln!"95%% : %.4f"(dist.cdfInverse(0.95f));

	writeln("-----");
	import lantern;
	import std.algorithm;

	static struct Val
	{
		float data;
	}

	fn.data.map!(val => Val(val)).describe().printTable();
}

struct LogLikehoodCostFunc
{
	float[] data;

	auto opCall(T)(const scope T[] params)
	{
		// https://en.wikipedia.org/wiki/Johnson's_SU-distribution
		import lbfgsd.distribution : johnsonSUDistribution;

		const gamma = params[0];
		const delta = exp(params[1]); // delta > 0
		const lambda = exp(params[2]); // lambda > 0
		const xi = params[3];

		auto dist = johnsonSUDistribution(gamma, delta, lambda, xi);

		auto result = T(0);
		foreach (d; data)
		{
			result += dist.logLikehood(d);
		}
		return -result;
	}
}
