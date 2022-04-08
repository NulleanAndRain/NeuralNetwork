using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using MathNet.Numerics.Distributions;

namespace Neuro
{
    internal class MatrixNeuroNet
    {
        private Matrix<double>[] _weightedSums;
        private Matrix<double>[] _values;
        private Matrix<double>[] _weights;

        private Matrix<double>[] _deltas;
        private Matrix<double>[] _acc;


        private const int IMAGE_PIXELS = DatasetReader.IMAGE_PIXELS;
        private const int OUTPUT_NEURONS = 10;
        private const double LEARN_RATE = 0.8;


        private const int NN_LAYERS = 3;
        private readonly int[] INTERNAL_NEURONS = new int[NN_LAYERS] 
        {
            IMAGE_PIXELS, 
            192, 
            //128, 
            //128, 
            //32, 
            OUTPUT_NEURONS 
        };

        public MatrixNeuroNet()
        {
            _values = new Matrix[NN_LAYERS];
            _weights = new Matrix[NN_LAYERS];
            _weightedSums = new Matrix[NN_LAYERS];

            _deltas = new Matrix<double>[NN_LAYERS];
            _acc = new Matrix<double>[NN_LAYERS];

            _weights[0] = Matrix.Build.Random(IMAGE_PIXELS, 1);
            _values[0] = Matrix.Build.Dense(1, IMAGE_PIXELS);
            _weightedSums[0] = Matrix.Build.Dense(1, IMAGE_PIXELS);

            _weights[^1] = Matrix.Build.Random(INTERNAL_NEURONS[^2], OUTPUT_NEURONS);
            _values[^1] = Matrix.Build.Dense(1, OUTPUT_NEURONS);
            _weightedSums[^1] = Matrix.Build.Dense(1, OUTPUT_NEURONS);

            var dist = Normal.WithMeanVariance(0.4, 0.1);
            for (int i = 1; i < NN_LAYERS - 1; i++)
            {
                _weights[i] = Matrix.Build.Random(INTERNAL_NEURONS[i - 1], INTERNAL_NEURONS[i], dist);
                _values[i] = Matrix.Build.Dense(1, INTERNAL_NEURONS[i]);
                _weightedSums[i] = Matrix.Build.Dense(1, INTERNAL_NEURONS[i]);
            }

            for (int i = 0; i < NN_LAYERS; i++)
            {
                _deltas[i] = Matrix.Build.Dense(_weights[i].RowCount, _weights[i].ColumnCount);
                _acc[i] = Matrix.Build.Dense(_weights[i].RowCount, _weights[i].ColumnCount);
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
            var weighted_input = image.Zip(_weights[0].Column(0), (x, w) => x * w).ToArray();
            _weightedSums[0] = Matrix<double>.Build.DenseOfRowArrays(weighted_input);
            _values[0] = _weightedSums[0].Map(Softsign, SKIP_ZEROS);

            for (int layer = 1; layer < NN_LAYERS; layer++)
            {
                _values[layer - 1].Multiply(_weights[layer], _weightedSums[layer]);
                _weightedSums[layer].Map(Softsign, _values[layer], SKIP_ZEROS);
            }

            return _values[^1].Row(0).ToArray();
        }

        public void Learn(ImageData data, int era, int index)
        {
            const Zeros SKIP_ZEROS = Zeros.AllowSkip;

            var predicted = Run(data.Pixels);
            var expected = new double[OUTPUT_NEURONS];
            expected[data.Label] = 1;

            var exp = Matrix<double>.Build.DenseOfRowArrays(expected);

            //var mse = predicted.Zip(expected, (p, e) => (p - e) * (p - e)).Sum() / predicted.Length;
            //var sigma = predicted.Zip(expected, (p, e) => e - p).Sum() / predicted.Length;

            //var _a = _values[^1]
            //.TransposeThisAndMultiply(_values[^1])
            //.Inverse()
            //.TransposeAndMultiply(_values[^1])
            //.Transpose();

            //var _y = _values[^1] * _a;
            //var errorMatrix = exp - _a;
            //var sigma = errorMatrix;
            var sigma = exp - _values[^1];

            var vec = sigma.Column(0);

            var mse = vec.Select(s => s * s).Sum() / sigma.ColumnCount;

            // m1(x1, y1) * m2(x2, y2) = m3(x1, y2)

            {
                const int layer = NN_LAYERS - 1;

                //_weightedSums[layer].Map(SoftsignDerivative, _weightedSums[layer], SKIP_ZEROS);

                _values[layer - 1]
                    //.TransposeThisAndMultiply(_weightedSums[layer])
                    .TransposeThisAndMultiply(sigma, _deltas[layer]);
                    //.TransposeAndMultiply(_weights[layer + 1], _deltas[layer]);
            }

            for (int layer = NN_LAYERS - 2; layer > 0; layer--)
            {
                _weightedSums[layer].Map(SoftsignDerivative, _weightedSums[layer], SKIP_ZEROS);

                _values[layer - 1]
                    .TransposeThisAndMultiply(_weightedSums[layer])
                    .Multiply(_deltas[layer + 1])
                    .TransposeAndMultiply(_weights[layer + 1], _deltas[layer]);
                //.Transpose(_deltas[layer]);
            }

            {
                const int layer = 0;

                var values = Matrix<double>.Build.DenseOfDiagonalArray(data.Pixels);
                _weightedSums[layer].Map(SoftsignDerivative, _weightedSums[layer], SKIP_ZEROS);

                _weights[layer + 1].TransposeAndMultiply(_deltas[layer + 1])
                    .Multiply(values)
                    .TransposeAndMultiply(_weightedSums[layer], _deltas[layer]);
            }

            var r = LEARN_RATE * mse;
            for (int layer = 0; layer < NN_LAYERS; layer++)
            {
                _deltas[layer].Multiply(r, _deltas[layer]);
                //_acc[layer].Add(_deltas[layer], _acc[layer]);
                _weights[layer].Add(_acc[layer], _weights[layer]);
            }

            Console.WriteLine($"era {era}: learned image #{index.ToString().PadRight(4)} ({data.Label}) | error: {mse}");
        }

        public void InitLearn()
        {
            _deltas = new Matrix<double>[NN_LAYERS];
            _acc = new Matrix<double>[NN_LAYERS];
            for (int i = 0; i < NN_LAYERS; i++)
            {
                _deltas[i] = Matrix.Build.Dense(_weights[i].RowCount, _weights[i].ColumnCount);
                _acc[i] = Matrix.Build.Dense(_weights[i].RowCount, _weights[i].ColumnCount);
            }
        }

        public void ApplyWeightDeltas()
        {
            for (int layer = 0; layer < NN_LAYERS; layer++)
            {
                _weights[layer].Add(_acc[layer], _weights[layer]);
            }
        }

        public void LoadWeigths(string str)
        {
            var w = Newtonsoft.Json.JsonConvert.DeserializeObject<double[][][]>(str);
            if (w == null) return;
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
