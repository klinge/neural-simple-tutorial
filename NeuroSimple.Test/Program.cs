using NeuroSimple.Layers;
using NeuroSimple.Cost;
using NeuroSimple.Metrics;
using System;

namespace NeuroSimple.Test
{
    class Program
    {
        static void Main(string[] args)
        {
/*
            //
            //1. Test tensor and operations
            //
            Operations K = new Operations();

            //Load array to the tensor
            NDArray a = new NDArray(3, 6);
            a.Load(1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 8, 7, 6, 5, 4, 3, 2, 1);
            a.Print("Load array");

            //Transpose of the matrix
            NDArray t = a.Transpose();
            t.Print("Transpose");

            //Create a tensor with all value 5
            NDArray b = new NDArray(6, 3);
            b.Fill(5);
            b.Print("Constant 5");

            //Create a tensor with all value 3
            NDArray c = new NDArray(6, 3);
            c.Fill(3);
            c.Print("Constant 3");

            // Subtract two tensor
            b = b - c;

            //Perform dot product
            NDArray r = K.Dot(a, b);
            r.Print("Dot product");
*/
            //
            //2. Test layers and activations
            //
/*
            //Load array to the tensor
            NDArray x = new NDArray(1, 3);
            x.Load(1, 2, 3);
            x.Print("Load array");

            //Create two layers, one with 6 neurons and another with 1
            FullyConnected fc1 = new FullyConnected(3, 6, "relu");
            FullyConnected fc2 = new FullyConnected(6, 1, "sigmoid");

            //Connect input by passing data from one layer to another
            fc1.Forward(x);
            x = fc1.Output;
            x.Print("FC1 Output");

            fc2.Forward(x);
            x = fc2.Output;
            x.Print("FC2 Output");

*/

            //
            //3. Test cost functions and metrics
            //

            //Load array to the tensor
            NDArray x = new NDArray(3, 3);
            x.Load(2, 4, 6, 1, 3, 5, 2, 3, 5);
            x.Print("Load X values");

            NDArray y = new NDArray(3, 1);
            y.Load(20, 15, 15);
            y.Print("Load Y values");

            //Create two layers one with 6 neurons and another with 1
            FullyConnected fc1 = new FullyConnected(3, 6, "relu");
            FullyConnected fc2 = new FullyConnected(6, 1, "relu");

            //Connect input by passing data from one layer to the other
            fc1.Forward(x);
            fc2.Forward(fc1.Output);
            var preds = fc2.Output;
            preds.Print("Predictions");

            //Calculate mean square error between predicted and expected values
            BaseCost cost = new BinaryCrossEntropy();
            var costValues = cost.Forward(preds, y);
            costValues.Print("BCE Cost");

            //Calculate the mean absolute value for the predicted vs expected values
            BaseMetric metric = new BinaryAccuracy();
            var metricValues = metric.Calculate(preds, y);
            metricValues.Print("Acc Metric");

            //Backpropagation starts here
            //Calculate gradient cost function
            var grad = cost.Backward(preds, y);
            //Then the fc2 layer by passing cost function grad into the layer backward function
            fc2.Backward(grad);
            //The grad of the fc2 is stored in the InputGrad property, pass it to fc1
            fc1.Backward(fc2.InputGrad);

            //Print parameters for both layers along with the Grad
            fc1.PrintParams();
            fc2.PrintParams();
        }
    }
}