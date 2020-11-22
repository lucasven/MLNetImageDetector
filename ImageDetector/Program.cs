using Microsoft.ML;
using System;
using System.IO;
using System.Linq;

namespace ImageDetector
{
    class Program
    {
        static void Main(string[] args)
        {
            //create a machine learning context
            var mlContext = new MLContext();

            //load the tsv file with data
            var data = mlContext.Data.LoadFromTextFile<ImageNetData>("./images/tags.tsv", hasHeader: true);

            var pipeline = mlContext.Transforms
                .LoadImages(
                outputColumnName: "input",
                imageFolder: "images",
                inputColumnName: nameof(ImageNetData.ImagePath))
                .Append(mlContext.Transforms
                .ResizeImages(
                    outputColumnName: "input",
                    imageWidth: 224,
                    imageHeight: 224,
                    inputColumnName: "input"))
                .Append(mlContext.Transforms.ExtractPixels(
                    outputColumnName: "input", 
                    interleavePixelColors: true, 
                    offsetImage: 117))
                .Append(mlContext.Model.LoadTensorFlowModel("./models/tensorflow_inception_graph.pb")
                .ScoreTensorFlowModel(
                    outputColumnNames: new[] { "softmax2" },
                    inputColumnNames: new[] { "input" },
                    addBatchDimensionInput: true));

            Console.WriteLine("Start training model...");
            var model = pipeline.Fit(data);
            Console.WriteLine("Model training complete");


            var engine = mlContext.Model.CreatePredictionEngine<ImageNetData, ImageNetPrediction>(model);

            var labels = File.ReadAllLines("models/imagenet_comp_graph_label_strings.txt");

            Console.WriteLine("Predicting image contents...");
            var images = ImageNetData.ReadFromCsv("images/tags.tsv");
            foreach (var image in images)
            {
                Console.WriteLine($" [{image.ImagePath}]: ");
                var prediction = engine.Predict(image).PredictLabels;

                var i = 0;
                var best = (from p in prediction
                            select new { Index = i++, Prediction = p })
                            .OrderByDescending(p => p.Prediction)
                            .First();
                var predictedLabel = labels[best.Index];

                Console.Write($"{predictedLabel} {(predictedLabel != image.Label ? "**WRONG**" : "")}");
            }
        }
    }
}
