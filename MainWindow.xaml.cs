using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;
using System.Threading;
using System.Runtime.InteropServices;
using System.Diagnostics;

namespace Neuro
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        private const double DPI = 96d;
        private readonly PixelFormat PIXEL_FORMAT_BYTE;
        private readonly int BM_STRIDE;
        private const string WEIGTHTS_PATH = "./weigths.json";

        private MatrixNeuroNet _nn;
        private List<ImageData> images;

        private int indexCurrent = 0;

        #region console

        [DllImport("Kernel32")]
        public static extern void AllocConsole();

        [DllImport("Kernel32")]
        public static extern void FreeConsole();

        [DllImport("kernel32.dll")]
        static extern IntPtr GetConsoleWindow();

        [DllImport("user32.dll")]
        static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);

        const int SW_HIDE = 0;
        const int SW_SHOW = 5;
        private IntPtr _console;

        private void ShowConsole() => ShowWindow(_console, SW_SHOW);
        private void HidConsole() => ShowWindow(_console, SW_HIDE);

        #endregion

        public MainWindow()
        {
            PIXEL_FORMAT_BYTE = PixelFormats.Gray8;
            BM_STRIDE = (DatasetReader.IMAGES_SIZE_X * PIXEL_FORMAT_BYTE.BitsPerPixel + 7) / 8;
            InitializeComponent();

            images = DatasetReader.GetImages();
            _nn = new();

            AllocConsole();
            _console = GetConsoleWindow();
            ShowWindow(_console, SW_HIDE);
            UpdateUI();
        }

        #region controls

        private void PrintVerdict(double[] output)
        {
            var _l = output.ToList();
            var maxVal = _l.Max();
            var indexMax = _l.IndexOf(maxVal);
            var msg = $"Possibly it is digit {indexMax} (possibility: {(maxVal * 100).ToString("F2")}%)";
            verdict.Content = msg;

            debug.Content = '[' + string.Join(", ", output.Select(o => o.ToString("F2").PadLeft(5))) + ']';
        }

        private void CheckThis(object sender, RoutedEventArgs e)
        {
            var img = images[indexCurrent];
            var output = _nn.Run(img.Pixels);
            PrintVerdict(output);
        }

        private void Button_NextIndex(object sender, RoutedEventArgs e)
        {
            indexCurrent++;
            indexCurrent %= DatasetReader.ITEMS_COUNT;
            UpdateUI();
        }

        private void Button_PrevIndex(object sender, RoutedEventArgs e)
        {
            indexCurrent--;
            if (indexCurrent < 0) indexCurrent = DatasetReader.ITEMS_COUNT - 1;
            UpdateUI();
        }

        private void Button_ResetIndex(object sender, RoutedEventArgs e)
        {
            indexCurrent = 0;
            UpdateUI();
        }

        private void UpdateUI()
        {
            var img = images[indexCurrent];
            _nn.Run(img.Pixels);
            if (img != null)
            {
                var bm = BitmapSource.Create(
                    DatasetReader.IMAGES_SIZE_X,
                    DatasetReader.IMAGES_SIZE_Y,
                    DPI, DPI,
                    PIXEL_FORMAT_BYTE,
                    null,
                    img.PixelBytes,
                    BM_STRIDE);
                ImgOutput.Source = bm;
                LabelOutput.Content = img.Label.ToString();


                var bm_weights = BitmapSource.Create(
                    DatasetReader.IMAGES_SIZE_X,
                    DatasetReader.IMAGES_SIZE_Y,
                    DPI, DPI,
                    PIXEL_FORMAT_BYTE,
                    null,
                    _nn.GetInputWeights().Select(d => (byte)(d*255)).ToArray(),
                    BM_STRIDE);

                ImgOutput_Weights.Source = bm_weights;

                var bm_weightedInputs = BitmapSource.Create(
                    DatasetReader.IMAGES_SIZE_X,
                    DatasetReader.IMAGES_SIZE_Y,
                    DPI, DPI,
                    PIXEL_FORMAT_BYTE,
                    null,
                    img.Pixels.Zip(_nn.GetInputWeights(), (px, w) => (byte)(px * w * 255)).ToArray(),
                    BM_STRIDE);

                ImgOutput_WeightedImg.Source = bm_weightedInputs;
            }
            Index.Content = indexCurrent;
        }

        private void Button_Save(object sender, RoutedEventArgs e)
        {
            File.WriteAllText(WEIGTHTS_PATH, _nn.SerializeWeigths());
        }

        private void Button_Load(object sender, RoutedEventArgs e)
        {
            if (!File.Exists(WEIGTHTS_PATH)) return;
            var w_str = File.ReadAllText(WEIGTHTS_PATH);
            _nn.LoadWeigths(w_str);
        }

        private void Button_Reset(object sender, RoutedEventArgs e)
        {
            _nn = new();
            UpdateUI();
        }

        private void ImagesCount_ValueChanged(object sender, RoutedPropertyChangedEventArgs<double> e)
        {
            if (ImgCountLabel == null) return;
            ImgCountLabel.Content = (int?)ImagesCount?.Value ?? 500;
        }

        #endregion

        #region learning

        // nn learning
        private const int LEARN_ERAS = 1;
        private void Button_Learn(object sender, RoutedEventArgs e)
        {
            ShowConsole();

            int data_length = (int)ImagesCount.Value;
            var erasParse = int.TryParse(Eras.Text, out int eras);
            if (!erasParse) eras = LEARN_ERAS;

            var test_data = images.Take(data_length).ToArray();
            var sw = Stopwatch.StartNew();
            for (var era = 0; era < eras; era++)
            {
                _nn.InitLearn();
                var index = 0;
                foreach (var img in test_data)
                {
                    _nn.Learn(img, era, index);
                    index++;
                }
                _nn.ApplyWeightDeltas();
            }
            sw.Stop();
            Console.WriteLine($"learned on {data_length} images in {eras} eras in {sw.Elapsed.ToString("G")}");
            Console.ReadKey();

            HidConsole();
        }

        #endregion
    }
}
