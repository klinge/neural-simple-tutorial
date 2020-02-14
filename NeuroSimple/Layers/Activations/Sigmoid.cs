using System;
using System.Collections.Generic;

namespace NeuroSimple.Layers.Activations
{
    public class Sigmoid : BaseActivation
    {
        public Sigmoid() : base("sigmoid")
        {

        }

        public override void Forward(NDArray x)
        {
            base.Forward(x);
            Output = Exp(x) / (1 + Exp(x));
        }
    }
}