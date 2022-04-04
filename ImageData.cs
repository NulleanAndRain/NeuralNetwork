using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuro
{
    internal class ImageData
    {
        public byte[]? ImagePixels { get; set; }
        public byte? Label { get; set; }

        public ImageData(byte[]? imagePixels, byte? label)
        {
            ImagePixels = imagePixels;
            Label = label;
        }

        public byte[] GetImgPixels3Col()
        {
            if (ImagePixels == null) return new byte[0];
            var pixels = new List<byte>(ImagePixels.Length * 3);
            foreach(var pixel in ImagePixels)
            {
                pixels.Add(pixel);
                pixels.Add(pixel);
                pixels.Add(pixel);
            }
            return pixels.ToArray();
        }
    }
}
