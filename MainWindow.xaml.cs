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

namespace Neuro
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        const double DPI = 96d;
        private readonly PixelFormat PIXEL_FORMAT;
        private readonly int BM_STRIDE;

        private NeuroNet _nn;

        private int indexCurrent = 0;

        private List<ImageData> images;

        private const int LEARN_ERAS = 5;

        private const string WEIGTHTS_PATH = "./weigths.json";

        public MainWindow()
        {
            PIXEL_FORMAT = PixelFormats.Gray8;
            BM_STRIDE = (DatasetReader.IMAGES_SIZE_X * PIXEL_FORMAT.BitsPerPixel + 7) / 8;
            InitializeComponent();

            images = DatasetReader.GetImages();
            _nn = new();

            UpdateUI();
        }

        private void PrintVerdict(double[] output)
        {
            var _l = output.ToList();
            var maxVal = _l.Max();
            var indexMax = _l.IndexOf(maxVal);
            var msg = $"Possibly it is digit {indexMax} (possibility: {(maxVal * 100).ToString("F2")}%)";
            verdict.Content = msg;

            debug.Content = '[' + string.Join(", ", output.Select(o => o.ToString("F2"))) + ']';
        }

        private void CheckThis(object sender, RoutedEventArgs e)
        {
            var img = images[indexCurrent];
            var output = _nn.Run(img.ImagePixels);
            PrintVerdict(output);
        }

        private void Button_Learn(object sender, RoutedEventArgs e)
        {

        }

        private void Button_NextIndex(object sender, RoutedEventArgs e)
        {
            indexCurrent++;
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
            if (img != null)
            {
                var bm = BitmapSource.Create(
                    DatasetReader.IMAGES_SIZE_X,
                    DatasetReader.IMAGES_SIZE_Y,
                    DPI, DPI,
                    PIXEL_FORMAT,
                    null,
                    img.ImagePixels,
                    BM_STRIDE);
                ImgOutput.Source = bm;
                LabelOutput.Content = img.Label.ToString();
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
            var w = Newtonsoft.Json.JsonConvert.DeserializeObject<double[][]>(w_str);
            _nn = new(w);
        }
    }
}
