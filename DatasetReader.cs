using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Neuro
{
    internal static class DatasetReader
    {
        // http://yann.lecun.com/exdb/mnist/
        const string IMAGES_PATH = @"./dataset/t10k-images.idx3-ubyte";
        const string LABELS_PATH = @"./dataset/t10k-labels.idx1-ubyte";

        public const int IMAGES_SIZE_X = 28;
        public const int IMAGES_SIZE_Y = 28;
        public const int IMAGE_PIXELS = IMAGES_SIZE_X * IMAGES_SIZE_Y;

        public const int ITEMS_COUNT = 10_000;

        public static List<ImageData> GetImages()
        {
            var img_bytes = File.ReadAllBytes(IMAGES_PATH).Skip(4).ToArray();
            var labels_bytes = File.ReadAllBytes(LABELS_PATH).Skip(2).ToList();

            var res = new List<ImageData>(ITEMS_COUNT);
            var i = 0;
            foreach (var label in labels_bytes)
            {
                var pixels = img_bytes.Skip(i * IMAGE_PIXELS).Take(IMAGE_PIXELS).ToArray();
                var data = new ImageData(pixels, label);
                res.Add(data);
                i++;
            }

            return res;
        }
    }
}
