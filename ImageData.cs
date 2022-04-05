using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuro
{
    internal class ImageData
    {
        public byte[] PixelBytes { get; set; }
        public double[] Pixels { get; private set; }
        public byte Label { get; set; }

        public ImageData(byte[] imagePixels, byte label)
        {
            PixelBytes = imagePixels;
            Label = label;

            Pixels = imagePixels.Select(b => NormalizeByte(b)).ToArray();
        }
        private double NormalizeByte(byte b) => 1d * b / byte.MaxValue;
    }
}
