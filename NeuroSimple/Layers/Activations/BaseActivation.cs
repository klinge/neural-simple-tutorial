using System;
using System.Collections.Generic;
using System.Text;

namespace NeuroSimple.Layers.Activations
{
    public class BaseActivation : BaseLayer
    {
        public BaseActivation(string name) : base(name)
        {

        }

        public static BaseActivation Get(string name)
        {
            BaseActivation baseActivation = null;
            switch (name)
            {
                case "relu":
                    baseActivation = new ReLU();
                    break;
                case "sigmoid":
                    baseActivation = new Sigmoid();
                    break;
                default:
                    break;
            }

            return baseActivation;
        }
    }
}