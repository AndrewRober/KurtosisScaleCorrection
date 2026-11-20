using DeviationBaisExperiment;
using Microsoft.VisualStudio.TestTools.UnitTesting;

namespace MSUnitTest
{
    [TestClass]
    public class ArithmeticTests
    {
        private static readonly int[] OddSample = { 1, 2, 3, 4, 5 };
        private static readonly int[] EvenSample = { 1, 2, 3, 4, 5, 6 };

        [TestMethod]
        public void StandardDeviationBiases_OddMean_MatchesExact()
        {
            var result = Arithmetic.StandardDeviationBiases(OddSample);
            Assert.AreEqual(1.5811388, Math.Round(result.withoutBias, 7));
            Assert.AreEqual(1.4142136, Math.Round(result.withBias, 7));
        }

        [TestMethod]
        public void StandardDeviationBiases_NonIntegerMean_FixesLegacyTruncationBug()
        {
            // True mean = 3.5, true W = 17.5 -> Bessel = sqrt(3.5), MLE = sqrt(17.5 / 6)
            var fixedRes = Arithmetic.StandardDeviationBiases(EvenSample);
            var buggyRes = Arithmetic.LegacyBuggyStandardDeviationBiases(EvenSample);

            Assert.AreEqual(Math.Sqrt(3.5), fixedRes.withoutBias, 1e-12);
            Assert.AreEqual(Math.Sqrt(17.5 / 6.0), fixedRes.withBias, 1e-12);
            // Legacy int division truncated 3.5 -> 3, giving W = 19 instead of 17.5
            Assert.AreEqual(Math.Sqrt(19.0 / 5.0), buggyRes.withoutBias, 1e-12);
            Assert.IsTrue(buggyRes.withoutBias > fixedRes.withoutBias + 0.07);
        }

        [TestMethod]
        public void VarianceAndStatistics_MatchExact()
        {
            Assert.AreEqual(2.0, Arithmetic.Variance(OddSample), 1e-12);
            Assert.AreEqual(17.5 / 6.0, Arithmetic.Variance(EvenSample), 1e-12);

            var oddStats = Arithmetic.CalculateStatistics(OddSample);
            Assert.AreEqual(3.0, oddStats.mean, 1e-12);
            Assert.AreEqual(3.0, oddStats.median, 1e-12);
            Assert.AreEqual(1.5, oddStats.Q1, 1e-12);
            Assert.AreEqual(4.5, oddStats.Q3, 1e-12);

            var evenStats = Arithmetic.CalculateStatistics(EvenSample);
            Assert.AreEqual(3.5, evenStats.mean, 1e-12);
            Assert.AreEqual(3.5, evenStats.median, 1e-12);
            Assert.AreEqual(2.0, evenStats.Q1, 1e-12);
            Assert.AreEqual(5.0, evenStats.Q3, 1e-12);
        }

        [TestMethod]
        public void Theorem2_OptimalVarianceDivisor_RecoversSteinAndHeavyTails()
        {
            // Gaussian (kappa = 3.0) -> exact Stein divisor (n + 1) for all n >= 2
            for (int n = 2; n <= 64; n++)
                Assert.AreEqual(n + 1.0, KurtosisPowerScale.OptimalVarianceDivisor(n, 3.0), 1e-12);

            // Uniform (kappa = 1.8) at n = 6 -> 5 + (5*1.8 - 3)/6 = 6.0 (matches MLE n!)
            Assert.AreEqual(6.0, KurtosisPowerScale.OptimalVarianceDivisor(6, 1.8), 1e-12);

            // Heavy-tailed outlier channel (kappa = 35, n = 16) -> far larger than Bessel (15)
            double cHeavy = KurtosisPowerScale.OptimalVarianceDivisor(16, 35.0);
            Assert.IsTrue(cHeavy > 45.0);
        }

        [TestMethod]
        public void Theorem2And3_KpsBeatsBesselAndMleOnHeavyTailedMonteCarlo()
        {
            // Simulate heavy-tailed Student-t(df=5) where true variance = 5/3, true kurtosis = 9.0
            var rng = new Random(12345);
            int n = 16, trials = 8000;
            double trueVar = 5.0 / 3.0;
            double trueStd = Math.Sqrt(trueVar);

            double mseVarBessel = 0, mseVarMle = 0, mseVarKps = 0;
            double mseStdBessel = 0, mseStdKps = 0;
            double[] sample = new double[n];

            for (int t = 0; t < trials; t++)
            {
                for (int i = 0; i < n; i++)
                {
                    // Student-t(df=5) via Z / sqrt(ChiSq(5)/5)
                    double z = NextGaussian(rng);
                    double chi2 = 0.0;
                    for (int k = 0; k < 5; k++)
                    {
                        double g = NextGaussian(rng);
                        chi2 += g * g;
                    }
                    sample[i] = z / Math.Sqrt(chi2 / 5.0);
                }

                var (_, w, _) = KurtosisPowerScale.ComputeMoments(sample);
                double vBessel = w / (n - 1.0);
                double vMle = w / n;
                double vKps = KurtosisPowerScale.EstimateVarianceMseOptimal(sample);

                mseVarBessel += (vBessel - trueVar) * (vBessel - trueVar);
                mseVarMle += (vMle - trueVar) * (vMle - trueVar);
                mseVarKps += (vKps - trueVar) * (vKps - trueVar);

                double sBessel = Math.Sqrt(vBessel);
                double sKps = KurtosisPowerScale.EstimatePowerScaleMseOptimal(sample, power: 1.0);
                mseStdBessel += (sBessel - trueStd) * (sBessel - trueStd);
                mseStdKps += (sKps - trueStd) * (sKps - trueStd);
            }

            Assert.IsTrue(mseVarKps < mseVarBessel && mseVarKps < mseVarMle,
                $"Expected KPS variance MSE ({mseVarKps}) < Bessel ({mseVarBessel}) and MLE ({mseVarMle})");
            Assert.IsTrue(mseStdKps < mseStdBessel,
                $"Expected KPS std MSE ({mseStdKps}) < Bessel ({mseStdBessel})");
        }

        [TestMethod]
        public void Theorem4_EmpiricalBayesGrpo_HandlesCollapsedAndHeterogeneousGroups()
        {
            var eb = new EmpiricalBayesGroupNormalizer(priorStrengthNu0: 4.0, emaBeta: 0.9);

            // Collapsed binary group (all zeros) -> exact zeros, no NaN/Inf
            double[] collapsed = { 0.0, 0.0, 0.0, 0.0 };
            double[] advCollapsed = eb.NormalizeGroup(collapsed);
            CollectionAssert.AreEqual(new[] { 0.0, 0.0, 0.0, 0.0 }, advCollapsed);

            // Normal group -> zero-sum centered advantages without explosion
            double[] active = { 0.0, 0.0, 0.0, 1.0 };
            double[] advActive = eb.NormalizeGroup(active);
            Assert.AreEqual(0.0, advActive.Sum(), 1e-10);
            Assert.IsTrue(advActive[3] > 0.0 && advActive[3] < 5.0);
        }

        private static double NextGaussian(Random rng)
        {
            double u1 = 1.0 - rng.NextDouble();
            double u2 = 1.0 - rng.NextDouble();
            return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
        }
    }
}
