using System;
using NeuroSimple.Layers;
using NUnit.Framework;

namespace NeuroSimple.Test
{

    public class NeuroSimpleTests
    {
        [Test]
        public void CreateAndManipulateTensorsTest()
        {
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

            Console.ReadLine();

            Assert.AreEqual(0, 0);
        }

        [Test]
        public void CreateLayerTest()
        {
            //Load array to the tensor
            NDArray x = new NDArray(1, 3);
            x.Load(1, 2, 3);
            x.Print("Load array");

            //Create two layers, one with 6 neurons and another with 1
            FullyConnected fc1 = new FullyConnected(3, 6, "relu");

            //Connect input by passing data from one layer to another
            fc1.Forward(x);
            x = fc1.Output;
            x.Print("FC1 Output");

            FullyConnected fc2 = new FullyConnected(6, 1, "sigmoid");
            fc2.Forward(x);
            x = fc2.Output;
            x.Print("FC2 Output");

            Console.ReadLine();
        }
    }
}