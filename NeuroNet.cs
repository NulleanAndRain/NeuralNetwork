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
        private Func<double, double>? DerivativeFunc;

        private const int NN_LAYERS = 5;
        private const int INTERNAL_LAYER_NEURONS = 128;
        public const int OUTPUT_NEURONS = 10;

        #region activation functions

        private double Softsign(double d)
        {
            return d / (1 + Math.Abs(d));
        }

        private double SoftsignDerivative(double d)
        {
            var temp = 1 + Math.Abs(d);
            return 1 / temp * temp;
        }

        #endregion

        #region normalization

        private double NormalizeByte(byte b) => 1d * b / byte.MaxValue;

        #endregion

        private double[][] values;
        private double[][][] weights;

        public NeuroNet(bool generateRandWeights = true)
        {
            var rnd = new Random(DateTime.Now.Millisecond);
            values = new double[NN_LAYERS][];
            weights = new double[NN_LAYERS][][];

            for (int layer = 0; layer < NN_LAYERS; layer++)
            {
                if (layer == NN_LAYERS - 1)
                {
                    values[layer] = new double[OUTPUT_NEURONS];
                    weights[layer] = new double[OUTPUT_NEURONS][];
                }
                else
                {
                    values[layer] = new double[INTERNAL_LAYER_NEURONS];
                    weights[layer] = new double[INTERNAL_LAYER_NEURONS][];
                }
                for (var neuron_index = 0; neuron_index < weights[layer].Length; neuron_index++)
                {
                    if (layer == 0) weights[layer][neuron_index] = new double[1];
                    else weights[layer][neuron_index] = new double[weights[layer - 1].Length];

                    if (generateRandWeights)
                        weights[layer][neuron_index] = weights[layer][neuron_index].Select(_ => rnd.NextDouble()).ToArray();
                }
            }
            ActivatonFunc = Softsign;
            DerivativeFunc = SoftsignDerivative;
        }

        public NeuroNet(double[][][] weights) : this(false)
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
                var n_weight = weights[0][n][0];
                //var sum = input.Zip(n_weigths, (i, w) => i * w).Sum();
                values[0][n] = ActivatonFunc(n_weight * input[n]);
            }

            for (int layer = 1; layer < NN_LAYERS; layer++)
            {
                for (int n = 0; n < values[layer].Length; n++)
                {
                    var n_weights = weights[layer][n];
                    var sum = values[layer - 1].Zip(n_weights, (i, w) => i * w).Sum();
                    values[layer][n] = ActivatonFunc(sum);
                }
            }

            return values[^1].ToArray();
        }

        #region learning

        public void Learn(ImageData data)
        {
            var expected = new double[10];
            expected[(int)data.Label] = 1d;

            var predicted = Run(data.ImagePixels);


        }

        // sigma = (expected - predicted) * dF
        // dw = sigma * neuron_val * learn_rate

        // sigma_next = Sum(*all sigmas from prev layer* * weigths) 

        // MeanedSquareError
        private double MSE(double[] observed, double[] predicted) =>
            observed.Zip(predicted, (p, o) => (o - p) * (o - p)).Sum() / observed.Length;

        #endregion
    }
}
