using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace Neuro
{
    internal class MatrixNeuroNet
    {
        private Matrix<double>[] _weightedSums;
        private Matrix<double>[] _values;
        private Matrix<double>[] _weights;

        private const int NN_LAYERS = 4;
        private const int INTERNAL_NEURONS = 32;
        private const int IMAGE_PIXELS = DatasetReader.IMAGE_PIXELS;
        private const int OUTPUT_NEURONS = 10;
        private const double LEARN_RATE = 1;

        public MatrixNeuroNet()
        {
            _values = new Matrix[NN_LAYERS];
            _weights = new Matrix[NN_LAYERS];
            _weightedSums = new Matrix[NN_LAYERS];

            //_values[0] = Matrix.Build.Dense(IMAGE_PIXELS, 1);
            //_weightedSums[0] = Matrix.Build.Dense(IMAGE_PIXELS, 1);
            _weights[0] = Matrix.Build.Random(IMAGE_PIXELS, 1);

            //_values[^1] = Matrix.Build.Dense(OUTPUT_NEURONS, 1);
            //_weightedSums[^1] = Matrix.Build.Dense(OUTPUT_NEURONS, 1);
            _weights[^1] = Matrix.Build.Random(INTERNAL_NEURONS, OUTPUT_NEURONS);

            int rows = IMAGE_PIXELS;
            for (int i = 1; i < NN_LAYERS - 1; i++)
            {
                //_values[i] = Matrix.Build.Dense(INTERNAL_NEURONS, 1);
                //_weightedSums[i] = Matrix.Build.Dense(INTERNAL_NEURONS, 1);
                _weights[i] = Matrix.Build.Random(rows, INTERNAL_NEURONS);
                rows = INTERNAL_NEURONS;
            }
        }

        #region funcs

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

        public double[] Run(double[] image)
        {
            const Zeros SKIP_ZEROS = Zeros.AllowSkip;
            //var input_matrix = Matrix<double>.Build.DenseOfRowArrays(image);
            //_weightedSums[0] = input_matrix.Multiply(_weights[0]);
            var weighted_input = image.Zip(_weights[0].Column(0), (x, w) => x * w).ToArray();
            _weightedSums[0] = Matrix<double>.Build.DenseOfRowArrays(weighted_input);
            _values[0] = _weightedSums[0].Map(Softsign, SKIP_ZEROS);

            for (int layer = 1; layer < NN_LAYERS; layer++)
            {
                _weightedSums[layer] = _values[layer - 1].Multiply(_weights[layer]);
                _values[layer] = _weightedSums[layer].Map(Softsign, SKIP_ZEROS);
            }

            return _values[^1].Row(0).ToArray();
        }

        public void Learn(ImageData data, int era, int index)
        {
            const Zeros SKIP_ZEROS = Zeros.AllowSkip;

            var predicted = Run(data.Pixels);
            var expected = new double[OUTPUT_NEURONS];
            expected[data.Label] = 1;

            var mse = predicted.Zip(expected, (p, e) => (p - e) * (p * e)).Sum() / predicted.Length;

            var sigma = predicted.Zip(expected, (p, e) => p - e).Sum() / predicted.Length;

            var deltas = new Matrix<double>[NN_LAYERS];

            // m1(x1, y1) * m2(x2, y2) = m3(x1, y2)

            {
                const int layer = NN_LAYERS - 1;

                var derivative = _weightedSums[layer].Map(SoftsignDerivative, SKIP_ZEROS);

                deltas[layer] = _values[layer - 1]
                    .TransposeThisAndMultiply(derivative)
                    .Multiply(sigma);
            }

            for (int layer = NN_LAYERS - 2; layer > 0; layer--)
            {
                var derivative = _weightedSums[layer].Map(SoftsignDerivative, SKIP_ZEROS);

                deltas[layer] = deltas[layer + 1]
                    .TransposeAndMultiply(_weights[layer + 1])
                    .TransposeAndMultiply(derivative)
                    .Multiply(_values[layer - 1])
                    .Transpose();
            }

            {
                const int layer = 0;

                var values = Matrix<double>.Build.DenseOfDiagonalArray(data.Pixels);
                var derivative = _weightedSums[layer].Map(SoftsignDerivative, SKIP_ZEROS);

                deltas[layer] = deltas[layer + 1]
                    .TransposeAndMultiply(_weights[layer + 1])
                    .Multiply(values)
                    .TransposeAndMultiply(derivative);
            }

            for (int layer = 0; layer < NN_LAYERS; layer++)
            {
                _weights[layer].Add(deltas[layer].Multiply(LEARN_RATE));
            }

            Console.WriteLine($"era {era}: learned image #{index} ({data.Label}) | error: {mse}");
        }

        //private double 

        public void LoadWeigths(string str)
        {
            var w = Newtonsoft.Json.JsonConvert.DeserializeObject<double[][][]>(str);
            for (int i = 0; i < w.Length; i++)
            {
                _weights[i] = Matrix<double>.Build.DenseOfRowArrays(w[i]);
            }
        }

        public string SerializeWeigths()
        {
            var w = _weights.Select(w => w.ToRowArrays()).ToArray();
            return Newtonsoft.Json.JsonConvert.SerializeObject(w);
        }
    }
}
