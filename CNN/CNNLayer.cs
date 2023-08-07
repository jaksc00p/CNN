using CNN.Utils;

namespace CNN
{
    /// <summary>
    /// Can be either a convolution layer or a pooling layer
    /// </summary>
    public class CNNLayer
    {
        protected int Kx { get; set; }
        protected int Ky { get; set; }
        protected int NrInputChannels { get; set; }
        protected int NrOutputChannels { get; set; }
        protected int Stride { get; set; } = 1;

        public virtual void CreateRandomParameters()
        {

        }
      
        public virtual Tensor Update(Tensor inputChannels)
        {
            return new Tensor(0, 0, 0, 0);
        }

        public virtual void MakeTrainingStep(double learningRate, int step)
        {

        }

    }
}
