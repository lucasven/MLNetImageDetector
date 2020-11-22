using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace ImageDetector
{
    public class ImageNetData
    {
        [LoadColumn(0)] public string ImagePath;
        [LoadColumn(1)] public string Label;

        /// <summary>
        /// Load the contents of a TSV file as an object sequence representing images and labels
        /// </summary>
        /// <param name="file">Name of the TSV file</param>
        /// <returns>sequence of objects containing result of TSV</returns>
        public static IEnumerable<ImageNetData> ReadFromCsv(string file)
        {
            return File.ReadAllLines(file)
                .Select(x => x.Split('\t'))
                .Select(x => new ImageNetData
                {
                    ImagePath = x[0],
                    Label = x[1]
                });
        }
    }

    /// <summary>
    /// prediction class that holds a model prediction
    /// </summary>
    public class ImageNetPrediction
    {
        [ColumnName("softmax2")]
        public float[] PredictLabels;
    }
}
