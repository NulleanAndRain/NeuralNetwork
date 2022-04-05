using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Newtonsoft.Json;

namespace Neuro
{
    internal class NeuroNet
    {

        private Func<double, double> ActivatonFunc;
        private Func<double, double> DerivativeFunc;

        private const int NN_LAYERS = 5;
        private const int INTERNAL_LAYER_NEURONS = 32;
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

        #endregion

        private double[][] values;
        private double[][][] weights;

        private double[] _last_input;

        private double[][] _neurons_sums;

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
                else if (layer == 0)
                {
                    weights[layer] = new double[DatasetReader.IMAGE_PIXELS][];
                    values[layer] = new double[DatasetReader.IMAGE_PIXELS];
                }
                else
                {
                    values[layer] = new double[INTERNAL_LAYER_NEURONS];
                    weights[layer] = new double[INTERNAL_LAYER_NEURONS][];
                }
                for (var neuron_index = 0; neuron_index < weights[layer].Length; neuron_index++)
                {
                    if (layer == 0) weights[layer][neuron_index] = new double[1];
                    else weights[layer][neuron_index] = new double[values[layer - 1].Length];

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

        public double[] Run(double[] image)
        {
            if (image.Length != DatasetReader.IMAGE_PIXELS) 
                image = image.Take(DatasetReader.IMAGE_PIXELS).ToArray();

            _last_input = image;

            // first layer
            for (int neuron = 0; neuron < values[0].Length; neuron++)
            {
                var n_weight = weights[0][neuron][0]; // neurons on 1st layer have only 1 input and weight each

                var sum = n_weight * image[neuron];
                values[0][neuron] = ActivatonFunc(sum);

                if (_neurons_sums != null)
                {
                    _neurons_sums[0][neuron] = sum;
                }
            }

            // other layers
            for (int layer = 1; layer < NN_LAYERS; layer++)
            {
                for (int neuron = 0; neuron < values[layer].Length; neuron++)
                {
                    var n_weights = weights[layer][neuron];
                    var sum = values[layer - 1].Zip(n_weights, (n, w) => n * w).Sum();

                    if (_neurons_sums != null)
                    {
                        _neurons_sums[layer][neuron] = sum;
                    }

                    values[layer][neuron] = ActivatonFunc(sum);
                }
            }

            return values[^1].ToArray();
        }

        #region learning

        private double[][] _sigmas;
        private double[][][] _weightDeltas;

        const double LEARN_RATE = 0.9d;

        public void InitLearn()
        {
            _sigmas = new double[NN_LAYERS][];
            _weightDeltas = new double[NN_LAYERS][][];
            _neurons_sums = new double[NN_LAYERS][];

            for (int i = 0; i < NN_LAYERS; i++)
            {
                _sigmas[i] = new double[values[i].Length];
                _neurons_sums[i] = new double[values[i].Length];

                _weightDeltas[i] = new double[weights[i].Length][];
                for (int j = 0; j < weights[i].Length; j++)
                {
                    _weightDeltas[i][j] = new double[weights[i][j].Length];
                }
            }
        }

        // sigma = (expected - predicted) * dF
        // dw = sigma * neuron_val * learn_rate

        // sigma_next = Sum(*all sigmas from prev layer* * weigths) 
        public void Learn(ImageData data, int era, int index)
        {
            var expected = new double[10];
            expected[data.Label] = 1d;

            var predicted = Run(data.Pixels);

            // last layer
            var lastLayer = values[^1];

            _sigmas[^1] = expected.Zip(predicted, (e, p) => e - p).ToArray();


            for (int neuron = 0; neuron < values[^1].Length; neuron++)
            {
                for (int n_input = 0; n_input < weights[^1][neuron].Length; n_input++)
                {
                    _weightDeltas[^1][neuron][n_input] = 
                        DerivativeFunc(_neurons_sums[^1][neuron]) * 
                        values[^2][n_input] * 
                        _sigmas[^1][neuron] * 
                        LEARN_RATE;
                }
            }

            for (var layer = NN_LAYERS - 2; layer > 0; layer--)
            {
                for (int neuron = 0; neuron < values[layer + 1].Length; neuron++)
                {
                    for (int n_input = 0; n_input < weights[layer + 1][neuron].Length; n_input++)
                    {
                        var s_sum = _sigmas[layer + 1].Zip(weights[layer + 1][neuron], (s, w) => s * w).Sum() / _sigmas[layer + 1].Length;
                        _sigmas[layer][neuron] = s_sum;

                        _weightDeltas[layer][neuron][n_input] +=
                            DerivativeFunc(_neurons_sums[layer - 1][neuron]) *
                            values[layer - 1][neuron] *
                            s_sum *
                            LEARN_RATE;
                    }
                }
            }

            for (var neuron = 0; neuron < values[0].Length; neuron++)
            {
                var s_sum = _sigmas[0].Zip(weights[0][neuron], (s, w) => s * w).Sum() / _sigmas[0].Length;
                _sigmas[0][neuron] = s_sum;

                _weightDeltas[0][neuron][0] +=
                    DerivativeFunc(_last_input[neuron]) *
                    _last_input[neuron] *
                    s_sum *
                    LEARN_RATE;
            }
            Console.WriteLine($"era {era}: learned image #{index} ({data.Label})");
        }

        public void ApplyWeightDeltas()
        {
            for (var i = 0; i < weights.Length; i++)
            for (int j = 0; j < weights[i].Length; j++)
            for (int k = 0; k < weights[i][j].Length; k++)
            {
                weights[i][j][k] -= _weightDeltas[i][j][k];
            }
        }


        // MeanedSquareError
        private double MSE(double[] observed, double[] predicted) =>
            observed.Zip(predicted, (p, o) => (o - p) * (o - p)).Sum() / observed.Length;

        #endregion
    }
}
