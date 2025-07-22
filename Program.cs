using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.IO;
using System.Linq;

namespace AI_Regression_Crop_Recommendation
{
    // Data Class
    public class CropDetails
    {
        [LoadColumn(0)]
        public float temperature;

        [LoadColumn(1)]
        public float humidity;

        [LoadColumn(2)]
        public float ph;

        [LoadColumn(3)]
        public float label;
    }

    // CropPrediction is the result returned prediction by the model 
    public class CropPrediction
    {
        [ColumnName("Score")]
        public float Label;
    }

    class Program
    {
        static readonly string _cropDataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Crop_recommendation.csv");

        static void Main(string[] args)
        {
           MLContext mlContext = new MLContext(seed: 0);
           var model = Train(mlContext, _cropDataPath);
           Evaluate(mlContext, model);
            Console.ReadLine();
        }
        
        // Train() function
       public static ITransformer Train(MLContext mlContext, string cropDataPath)
        {
            Console.WriteLine("Training Started.....");
            // IDataView holds the training dataset
            IDataView dataView = mlContext.Data.LoadFromTextFile<CropDetails>(cropDataPath, hasHeader: true, separatorChar: ',');
            var pipeline = mlContext.Transforms.Conversion.ConvertType("Label", "label")
                            .Append(mlContext.Transforms.Concatenate("Features", "temperature", "humidity", "ph"))
                            .Append(mlContext.Regression.Trainers.FastTree());
            
            //create the model
            Console.WriteLine("Creating Model.....");
            var model = pipeline.Fit(dataView);
            Console.WriteLine("Model created.....");

            // return trained model
            return model;
        }

        // Evaluate the data
        private static void Evaluate(MLContext mlContext, ITransformer model)
        {
            Console.WriteLine("Evaluating Data.....");
            IDataView dataView = mlContext.Data.LoadFromTextFile<CropDetails>(_cropDataPath, hasHeader: true, separatorChar: ',');
            var prediction = model.Transform(dataView);
            var metrics = mlContext.Regression.Evaluate(dataView, "label", "label");
            
            TestSinglePrediction(mlContext, model);
        }

        // Test Single Data
        private static void TestSinglePrediction(MLContext mlContext, ITransformer model)
        {            
            var predictionFunction = mlContext.Model.CreatePredictionEngine<CropDetails, CropPrediction>(model);
            var CropSample = new CropDetails()
            {
                temperature = 37.2f,
                humidity = 80.4f,
                ph = 5.8f,
                label = 0f 
            };
            Console.WriteLine("Predicting data.....");
            Console.WriteLine("");
            var prediction = predictionFunction.Predict(CropSample);
            Console.WriteLine("_____________________________________");
            Console.WriteLine("         Crop Details");
            Console.WriteLine("_____________________________________");
            Console.WriteLine("");
            Console.WriteLine($"    Temperature: {CropSample.temperature}");
            Console.WriteLine($"    Humidity: {CropSample.humidity}");
            Console.WriteLine($"    pH: {CropSample.ph}");
            Console.WriteLine("");
            Console.WriteLine("_____________________________________");
            Console.WriteLine("         Predicted Data");
            Console.WriteLine("_____________________________________");
            Console.WriteLine(" ");
            Console.WriteLine($"    Predicted Value: {prediction.Label}");
            Console.WriteLine(" ");
            // crop recommendation
            if (prediction.Label > 0 && prediction.Label < 1)
                  Console.WriteLine("    Predicted Crop: Rice");
            else if(prediction.Label > 1 && prediction.Label < 2)
                    Console.WriteLine("    Predicted Crop: Maize");
            else if (prediction.Label > 2 && prediction.Label < 3)
                Console.WriteLine("    Predicted Crop: Chickpea");
            else if (prediction.Label > 3 && prediction.Label < 4)
                Console.WriteLine("    Predicted Crop: KidneyBeans");
            else if (prediction.Label > 4 && prediction.Label < 5)
                Console.WriteLine("    Predicted Crop: Pigeonpeas");
            else if (prediction.Label > 5 && prediction.Label < 6)
                Console.WriteLine("    Predicted Crop: Mothbeans");
            else if (prediction.Label > 6 && prediction.Label < 7)
                Console.WriteLine("    Predicted Crop: Mungbean");
            else if (prediction.Label > 7 && prediction.Label < 8)
                Console.WriteLine("    Predicted Crop: Blackgram");
            else if (prediction.Label > 8 && prediction.Label < 9)
                Console.WriteLine("    Predicted Crop: Lentil");
            else if (prediction.Label > 9 && prediction.Label < 10)
                Console.WriteLine("    Predicted Crop: Pomegranate");
            else if (prediction.Label > 10 && prediction.Label < 11)
                Console.WriteLine("    Predicted Crop: Banana");
            else if (prediction.Label > 11 && prediction.Label < 12)
                Console.WriteLine("    Predicted Crop: Mango");
            else if (prediction.Label > 12 && prediction.Label < 13)
                Console.WriteLine("    Predicted Crop: Grapes");
            else if (prediction.Label > 13 && prediction.Label < 14)
                Console.WriteLine("    Predicted Crop: Watermelon");
            else if (prediction.Label > 14 && prediction.Label < 15)
                Console.WriteLine("    Predicted Crop: Muskmelon");
            else if (prediction.Label > 15 && prediction.Label < 16)
                Console.WriteLine("    Predicted Crop: Apple");
            else if (prediction.Label > 16 && prediction.Label < 17)
                Console.WriteLine("    Predicted Crop: Orange");
            else if (prediction.Label > 17 && prediction.Label < 18)
                Console.WriteLine("    Predicted Crop: Papaya");
            else if (prediction.Label > 18 && prediction.Label < 19)
                Console.WriteLine("    Predicted Crop: Coconut");
            else if (prediction.Label > 19 && prediction.Label < 20)
                Console.WriteLine("    Predicted Crop: Cotton");
            else if (prediction.Label > 20 && prediction.Label < 21)
                Console.WriteLine("    Predicted Crop: Jute");
            else 
                Console.WriteLine("    Predicted Crop: Coffee");
            Console.WriteLine("_____________________________________");
        }
    }
}