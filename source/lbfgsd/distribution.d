module lbfgsd.distribution;

import std.math : PI;
import lbfgsd.math;

struct NormalDistribution(T)
{
public:
    this(T mu,  T sigma)
    {
        this.mu = mu;
        this.sigma = sigma;
    }

public:
    auto pdf(U)(U x)
    {
        return exp(-square(x - mu) / square(sigma)) / sqrt(2 * PI * square(sigma));
    }

    auto cdf(U)(U x)
    {
        import std.mathspecial: normalDistribution;
        return normalDistribution(x);
    }

    auto cdfInverse(U)(U p)
    {
        import std.mathspecial: normalDistributionInverse;
        return normalDistributionInverse(x);
    }

    auto logLikehood(U)(U x)
    {
        return -square(x - mu) / square(sigma) - log(sigma) - (log(PI) + log(2)) / 2;
    }
public:
    T mu; //average
    T sigma; //standard devision
}

struct JohnsonSUDistribution(T)
{
public:
    this(T gamma, T delta, T lambda, T xi)
    {
        this.gamma = gamma;
        this.delta = delta;
        this.lambda = lambda;
        this.xi = xi;
    }

public:
    auto pdf(U)(U x)
    {
        auto z = (x - xi) / lambda;
        return delta * exp(-0.5 * square(gamma + delta * asinh(z))) / (sqrt(2 * PI) * (lambda * sqrt(square(z) + 1)));
    }

    auto cdf(U)(U x)
    {
        import std.mathspecial: normalDistribution;
        return normalDistribution(gamma + delta * asinh((x - xi) / lambda));
    }

	auto cdfInverse(U)(U p)
	{
		import std.mathspecial : normalDistributionInverse;
		auto x = normalDistributionInverse(p);
		return xi + lambda * sinh((x - gamma) / delta);
	}

    auto logLikehood(U)(U x)
    {
        auto xs = xi - x;
        return -0.5 * (log((2 * PI * (square(lambda) + square(xs))) / square(delta)) + square(delta * asinh(xs / lambda) - gamma));
    }

public:
    T gamma;
    T delta;
    T lambda;
    T xi;
}

JohnsonSUDistribution!T johnsonSUDistribution(T)(T gamma, T delta, T lambda, T xi)
{
    return typeof(return)(gamma, delta, lambda, xi);
}

unittest
{
    auto dist = johnsonSUDistribution(1.0, 1.0, 1.0, 1.0);
    auto x = dist.pdf(0.0);
    auto p = dist.cdf(0.5);
    auto c = dist.cdfInverse(p);
    auto l = dist.logLikehood(0.0);

    import std.math;
    assert(approxEqual(log(x), l));
    assert(approxEqual(c, 0.5));
}
unittest
{
    import lbfgsd.autodiff;
    alias Var = Variable!(double, 4);

    auto gamma = Var(1, 0);
    auto lambda = Var(1, 0);
    auto delta = Var(1, 0);
    auto xi = Var(1, 0);
    auto x = Var(0);

    auto dist = johnsonSUDistribution(gamma, lambda, delta, xi);
    auto tx = log(dist.pdf(x));
    auto tl = dist.logLikehood(x);

    import std.algorithm, std.math : approxEqual;
    assert(approxEqual(tx.a, tl.a));
    assert(equal!approxEqual(tx.d[], tl.d[]));

    auto x1 = log(dist.pdf(0.0));
    auto x2 = dist.logLikehood(0.0);
    assert(approxEqual(x1.a, x2.a));
    assert(equal!approxEqual(x1.d[], x2.d[]));
}

struct GumbelDistribution(T)
{
public:
    this(in T mu, in T eta)
    {
        this.mu = mu;
        this.eta = eta;
    }

public:
    auto pdf(U)(U x)
    {
        auto y = exp((mu - x) / eta);
        return exp(-y) * y / eta;
    }

    auto cdf(U)(U p)
    {
        return exp(-exp((mu - p) / eta));
    }

    auto cdfInverse(U)(U x)
    {
        return mu - eta * log(-log(x));
    }

    auto logLikehood(U)(U x)
    {
        auto z = (mu - x) / eta;
        return z - exp(z) - log(eta);
    }

public:
    T mu;
    T eta;
}

GumbelDistribution!T gumbelDistribution(T)(T mu, T eta)
{
    return typeof(return)(mu, eta);
}

unittest
{
    auto dist = gumbelDistribution(1.0, 1.0);
    auto p = dist.pdf(0.0);
    auto x = dist.cdf(0.5);
    auto c = dist.cdfInverse(x);
    auto l = dist.logLikehood(0.0);

    import std.math;
    assert(approxEqual(log(p), l));
    assert(approxEqual(c, 0.5));
}

unittest
{
    import lbfgsd.autodiff;
    alias Var = Variable!(double, 4);

    auto mu = Var(1, 0);
    auto eta = Var(1, 0);
    auto x = Var(0.1);
    auto p = Var(0.25);

    auto dist = gumbelDistribution(mu, eta);
    auto tx = dist.pdf(x);
    auto tl = dist.logLikehood(x);

    auto lx = log(tx);
    import std.algorithm, std.math : approxEqual;
    assert(approxEqual(lx.a, tl.a));
    assert(equal!approxEqual(lx.d[], tl.d[]));

    auto x1 = log(dist.pdf(0.30));
    auto x2 = dist.logLikehood(0.30);
    assert(approxEqual(x1.a, x2.a));
    assert(equal!approxEqual(x1.d[], x2.d[]));

    auto c = dist.cdf(p);
    auto tp = dist.cdfInverse(c);
    assert(approxEqual(p.a, tp.a));
    assert(equal!approxEqual(p.d[], tp.d[]));
}
