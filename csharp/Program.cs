using System.Text;

namespace DeviationBaisExperiment
{
    public static class Program
    {
        private const int SMALL_DS_SIZE = 1000, MEDIUM_DS_SIZE = 10_000, LARGE_DS_SIZE = 1_000_000;
        private const int DS_MIN_VALUE = 0, DS_MAX_VALUE = 1000;
        private const double SAMPLING_START = 0.005, SAMPLING_END = 0.1, SAMPLING_INC = 0.005;
        private static readonly int[] RANDOM_SEEDS = { 0, 1 };

        public static void Main(string[] args)
        {
            Console.WriteLine("===Kurtosis & Power-Scale Bias Correction (Upgraded from DeviationBaisExperiment)===");
            RunForensicAudit();
            Experiment1Fixed(monteCarloTrials: 200);
            Experiment2And3Fixed(steps: 25);
        }

        private static void RunForensicAudit()
        {
            int[] evenSample = { 1, 2, 3, 4, 5, 6 };
            var fixedRes = Arithmetic.StandardDeviationBiases(evenSample);
            var legacyBug = Arithmetic.LegacyBuggyStandardDeviationBiases(evenSample);

            Console.WriteLine("\n[Forensic Audit] Testing {1, 2, 3, 4, 5, 6} (true mean = 3.5):");
            Console.WriteLine($"  Legacy int-division bug Bessel std : {legacyBug.withoutBias:F6} (Error: +{(legacyBug.withoutBias - fixedRes.withoutBias) / fixedRes.withoutBias * 100:F2}%)");
            Console.WriteLine($"  Fixed exact double-mean Bessel std : {fixedRes.withoutBias:F6}");
        }

        private static void Experiment1Fixed(int monteCarloTrials)
        {
            var r1 = new Random(RANDOM_SEEDS[0]);
            int[] smallDs = Enumerable.Range(0, SMALL_DS_SIZE).Select(_ => r1.Next(DS_MIN_VALUE, DS_MAX_VALUE)).ToArray();
            int[] mediumDs = Enumerable.Range(0, MEDIUM_DS_SIZE).Select(_ => r1.Next(DS_MIN_VALUE, DS_MAX_VALUE)).ToArray();
            int[] largeDs = Enumerable.Range(0, LARGE_DS_SIZE).Select(_ => r1.Next(DS_MIN_VALUE, DS_MAX_VALUE)).ToArray();

            double smallStd = Arithmetic.StandardDeviation(smallDs);
            double mediumStd = Arithmetic.StandardDeviation(mediumDs);
            double largeStd = Arithmetic.StandardDeviation(largeDs);

            Console.WriteLine($"\n[Experiment 1 Fixed] Population Stds: Small={smallStd:F2}, Medium={mediumStd:F2}, Large={largeStd:F2}");

            int samplingCount = (int)Math.Round((SAMPLING_END - SAMPLING_START) / SAMPLING_INC) + 1;
            double[] rates = Enumerable.Range(1, samplingCount).Select(i => i * SAMPLING_INC).ToArray();

            Directory.CreateDirectory("Results");
            EvaluateDataset("small-ds.csv", smallDs, smallStd, rates, monteCarloTrials, seed: RANDOM_SEEDS[1]);
            EvaluateDataset("medium-ds.csv", mediumDs, mediumStd, rates, monteCarloTrials, seed: RANDOM_SEEDS[1] + 100);
            EvaluateDataset("large-ds.csv", largeDs, largeStd, rates, trials: 25, seed: RANDOM_SEEDS[1] + 200);
        }

        private static void EvaluateDataset(string fileName, int[] dataset, double truePopStd, double[] rates, int trials, int seed)
        {
            var rng = new Random(seed);
            var sb = new StringBuilder();
            sb.AppendLine("Sample Size,Sample Percentage,Actual Std,MLE Mean,Bessel Mean,KPS-MSE Mean,MLE RMSE,Bessel RMSE,KPS-MSE RMSE");

            foreach (double rate in rates)
            {
                int n = Math.Max(2, (int)Math.Round(rate * dataset.Length));
                double sumMle = 0, sumBessel = 0, sumKps = 0;
                double sqErrMle = 0, sqErrBessel = 0, sqErrKps = 0;

                double[] buffer = new double[n];
                int[] intBuf = new int[n];

                for (int t = 0; t < trials; t++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        int val = dataset[rng.Next(0, dataset.Length)]; // Fixed: samples across entire dataset.Length!
                        intBuf[k] = val;
                        buffer[k] = val;
                    }

                    // Materialized sample array (fixes lazy IEnumerable re-execution bug!)
                    // Correct tuple order & dataset reference std (fixes lines 159, 168, 169 bugs!)
                    (double bessel, double mle) = Arithmetic.StandardDeviationBiases(intBuf);
                    double kps = KurtosisPowerScale.EstimatePowerScaleMseOptimal(buffer, power: 1.0, knownKappa: 1.8);

                    sumMle += mle;
                    sumBessel += bessel;
                    sumKps += kps;

                    sqErrMle += (mle - truePopStd) * (mle - truePopStd);
                    sqErrBessel += (bessel - truePopStd) * (bessel - truePopStd);
                    sqErrKps += (kps - truePopStd) * (kps - truePopStd);
                }

                sb.AppendLine($"{n},{rate * 100:F1}%,{truePopStd:F3},{sumMle / trials:F3},{sumBessel / trials:F3},{sumKps / trials:F3}," +
                              $"{Math.Sqrt(sqErrMle / trials):F3},{Math.Sqrt(sqErrBessel / trials):F3},{Math.Sqrt(sqErrKps / trials):F3}");
            }

            File.WriteAllText(Path.Combine("Results", fileName), sb.ToString());
        }

        private static void Experiment2And3Fixed(int steps)
        {
            var rng = new Random(42);
            var sb = new StringBuilder();
            sb.AppendLine("Dataset Size,Sample Size,Mean Gap (Bessel - MLE),Squared Gap");

            for (int i = 1; i <= steps; i++)
            {
                int dsSize = i * SMALL_DS_SIZE;
                int[] ds = Enumerable.Range(0, dsSize).Select(_ => rng.Next(DS_MIN_VALUE, DS_MAX_VALUE)).ToArray();
                int sampleSize = Math.Max(5, (int)(0.01 * dsSize));

                // Materialized once so MAD, MSE, and MAPE inspect the exact same sample!
                int[] sample = Enumerable.Range(0, sampleSize).Select(_ => ds[rng.Next(0, ds.Length)]).ToArray();
                (double bessel, double mle) = Arithmetic.StandardDeviationBiases(sample);
                double gap = bessel - mle;
                sb.AppendLine($"{dsSize},{sampleSize},{gap:F4},{gap * gap:F6}");
            }

            File.WriteAllText(Path.Combine("Results", "exp2-3-fixed.csv"), sb.ToString());
            Console.WriteLine("[Experiment 2 & 3 Fixed] Saved clean CSV outputs under Results/.");
        }
    }
}
