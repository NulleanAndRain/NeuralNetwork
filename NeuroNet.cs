using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Neuro
{
    internal class NeuroNet
    {

        private Func<double, double>? ActivatonFunc;

        private const int NN_LAYERS = 5;
        private const int INTERNAL_LAYER_NEURONS = 128;
        public const int OUTPUT_NEURONS = 10;

        #region activation functions

        private double Softsign(double d)
        {
            return d / (1 + Math.Abs(d));
        }

        #endregion

        #region normalization

        private double NormalizeByte(byte b) => 1d * b / byte.MaxValue;

        #endregion

        private double[][] values;
        private double[][] weights;

        public NeuroNet(bool generateRandWeights = true)
        {
            var rnd = new Random(DateTime.Now.Millisecond);
            values = new double[NN_LAYERS][];
            weights = new double[NN_LAYERS][];

            values[0] = new double[DatasetReader.IMAGE_PIXELS];
            values[^1] = new double[OUTPUT_NEURONS];

            weights[0] = new double[DatasetReader.IMAGE_PIXELS];
            weights[0] = weights[0].Select(_ => rnd.NextDouble()).ToArray();
            weights[^1] = new double[OUTPUT_NEURONS];
            weights[^1] = weights[^1].Select(_ => rnd.NextDouble()).ToArray();
            for (int i = 1; i < NN_LAYERS - 1; i++)
            {
                values[i] = new double[INTERNAL_LAYER_NEURONS];
                weights[i] = new double[INTERNAL_LAYER_NEURONS];
                if (generateRandWeights)
                    weights[i] = weights[i].Select(_ => rnd.NextDouble()).ToArray();
            }
            ActivatonFunc = Softsign;
        }

        public NeuroNet(double[][] weights) : this(false)
        {
            if (weights is null)
            {
                throw new ArgumentNullException(nameof(weights));
            }

            this.weights = weights;
        }

        public string SerializeWeigths() => JsonConvert.SerializeObject(weights);

        public double[] Run(byte[] image)
        {
            if (image.Length != DatasetReader.IMAGE_PIXELS) 
                image = image.Take(DatasetReader.IMAGE_PIXELS).ToArray();

            var input = image.Select(b => NormalizeByte(b)).ToArray();

            for (int n = 0; n < values[0].Length; n++)
            {
                var w = weights[0][n];
                var sum = 0d;
                foreach (var i in input)
                {
                    sum += i * w;
                }
                values[0][n] = ActivatonFunc(sum);
            }

            for (int layer = 1; layer < NN_LAYERS; layer++)
            {
                for (int n = 1; n < NN_LAYERS; n++)
                {
                    var w = weights[layer][n];
                    var sum = 0d;
                    foreach (var v in values[layer - 1])
                    {
                        sum += v * w;
                    }
                    values[layer][n] = ActivatonFunc(sum);
                }
            }

            return values[^1].ToArray();
        }

        public double _get_w_0_0() => weights[0][0];
    }
}
