using System;
using System.Collections.Generic;
using System.Text;

namespace NeuroSimple.Metrics
{
    public class BinaryAccuracy : BaseMetric
    {
        public BinaryAccuracy() : base("binary_accurary")
        {
        }

        public override NDArray Calculate(NDArray preds, NDArray labels)
        {
            var output = Round(Clip(preds, 0, 1));
            return Mean(output == labels);
        }
    }
}