namespace DeviationBaisExperiment
{
    /// <summary>
    /// Upgraded, numerically stable O(n) statistical routines fixing the original repository's:
    ///   1. Integer-division mean truncation bug: (data.Sum() / data.Length) on int[]
    ///   2. O(n^2) inner-loop re-summation inside .AsParallel().Sum(...)
    ///   3. 32-bit signed integer overflow on large populations (N >= 10^6)
    /// Includes legacy bug reproduction helper exclusively for forensic audit benchmarking.
    /// </summary>
    public static class Arithmetic
    {
        public static double Median(this IEnumerable<int> source)
        {
            ArgumentNullException.ThrowIfNull(source);
            int[] sorted = source.OrderBy(v => v).ToArray();
            if (sorted.Length == 0)
                throw new InvalidOperationException("Empty collection");

            int mid = sorted.Length / 2;
            return sorted.Length % 2 == 0
                ? (sorted[mid - 1] + sorted[mid]) / 2.0
                : sorted[mid];
        }

        public static double Mode(this int[] sortedSource)
        {
            ArgumentNullException.ThrowIfNull(sortedSource);
            if (sortedSource.Length == 0)
                throw new InvalidOperationException("Empty collection");

            int modeCount = 1, currentCount = 1, mode = sortedSource[0];
            for (int i = 1; i < sortedSource.Length; i++)
            {
                if (sortedSource[i] == sortedSource[i - 1])
                {
                    currentCount++;
                }
                else
                {
                    if (currentCount > modeCount)
                    {
                        modeCount = currentCount;
                        mode = sortedSource[i - 1];
                    }
                    currentCount = 1;
                }
            }

            if (currentCount > modeCount)
                mode = sortedSource[^1];

            return mode;
        }

        public static (double Q1, double Q2, double Q3) CalculateQuartiles(int[] sortedData)
        {
            ArgumentNullException.ThrowIfNull(sortedData);
            if (sortedData.Length == 0)
                throw new ArgumentException("Data must contain at least one element", nameof(sortedData));

            int half = sortedData.Length / 2;
            if (sortedData.Length % 2 == 0)
            {
                return (
                    CalculateQuartile(sortedData.AsSpan(0, half)),
                    CalculateQuartile(sortedData.AsSpan()),
                    CalculateQuartile(sortedData.AsSpan(half, half))
                );
            }
            else
            {
                return (
                    CalculateQuartile(sortedData.AsSpan(0, half)),
                    CalculateQuartile(sortedData.AsSpan()),
                    CalculateQuartile(sortedData.AsSpan(half + 1, half))
                );
            }
        }

        public static double CalculateQuartile(ReadOnlySpan<int> sortedData)
        {
            if (sortedData.Length == 0)
                throw new ArgumentException("Data must contain at least one element", nameof(sortedData));

            int mid = sortedData.Length / 2;
            return sortedData.Length % 2 == 0
                ? (sortedData[mid - 1] + sortedData[mid]) / 2.0
                : sortedData[mid];
        }

        public static double SumOfSquaredDeviations(ReadOnlySpan<int> data)
        {
            if (data.Length == 0)
                throw new ArgumentException("Data must contain at least one element", nameof(data));

            long total = 0;
            for (int i = 0; i < data.Length; i++)
                total += data[i];

            double mean = (double)total / data.Length;
            double ssd = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                double diff = data[i] - mean;
                ssd += diff * diff;
            }
            return ssd;
        }

        /// <summary>
        /// Returns (withoutBias: Bessel n-1 std, withBias: MLE n std) in O(n) exact double-precision arithmetic.
        /// </summary>
        public static (double withoutBias, double withBias) StandardDeviationBiases(int[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length < 2)
                throw new ArgumentException("At least 2 elements required for Bessel correction.", nameof(data));

            double ssd = SumOfSquaredDeviations(data);
            return (Math.Sqrt(ssd / (data.Length - 1)), Math.Sqrt(ssd / data.Length));
        }

        /// <summary>
        /// Reproduces the original 2023 integer-division truncation bug exclusively for audit quantification.
        /// </summary>
        public static (double withoutBias, double withBias) LegacyBuggyStandardDeviationBiases(int[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            long intMeanTruncated = data.Sum(v => (long)v) / data.Length;
            double ssd = 0.0;
            for (int i = 0; i < data.Length; i++)
            {
                double diff = data[i] - intMeanTruncated;
                ssd += diff * diff;
            }
            return (Math.Sqrt(ssd / (data.Length - 1)), Math.Sqrt(ssd / data.Length));
        }

        public static double StandardDeviation(int[] data) => Math.Sqrt(Variance(data));

        public static double Variance(int[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            return SumOfSquaredDeviations(data) / data.Length;
        }

        public static double VarianceWithoutBias(int[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length < 2)
                throw new ArgumentException("At least 2 elements required.", nameof(data));
            return SumOfSquaredDeviations(data) / (data.Length - 1);
        }

        public static (double mean, double median, double mode, double range,
            double IQR, double Q1, double Q2, double Q3)
            CalculateStatistics(int[] data)
        {
            ArgumentNullException.ThrowIfNull(data);
            if (data.Length == 0)
                throw new ArgumentException("Data must contain at least one element", nameof(data));

            long sum = 0;
            for (int i = 0; i < data.Length; i++)
                sum += data[i];
            double mean = (double)sum / data.Length;

            int[] sortedData = (int[])data.Clone();
            Array.Sort(sortedData);

            double mode = sortedData.Mode();
            double range = sortedData[^1] - sortedData[0];
            var quartiles = CalculateQuartiles(sortedData);

            return (mean, quartiles.Q2, mode, range, quartiles.Q3 - quartiles.Q1, quartiles.Q1, quartiles.Q2, quartiles.Q3);
        }
    }
}
