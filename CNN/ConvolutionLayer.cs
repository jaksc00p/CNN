using CNN.Utils;

namespace CNN
{
    /// <summary>
    /// IMplements a convolution layer with a kernel and bias tensor. The values of both are trainable.
    /// </summary>
    public class ConvolutionLayer : CNNLayer
    {
        // Convolution parameters
        private readonly Tensor Kernels;
        private readonly Tensor Biases;

        private Optimizer KernelOptimizer, BiasOptimizer;

        /// <summary>
        /// Create a convolution layer with a given window size and stride lebgth (same in both directions)
        /// </summary>
        /// <param name="kx"></param>
        /// <param name="ky"></param>
        /// <param name="stride"></param>
        /// <param name="nrInputChannels"></param>
        /// <param name="nrOutputChannels"></param>
        public ConvolutionLayer(int kx, int ky, int stride, int nrInputChannels, int nrOutputChannels)
        {
            Kx = kx;
            Ky = ky;
            Stride = stride;
            NrInputChannels = nrInputChannels;
            NrOutputChannels = nrOutputChannels;

            Kernels = new Tensor(NrOutputChannels, NrInputChannels, Ky, Kx);
            Biases = new Tensor(NrOutputChannels);

            KernelOptimizer = new Optimizer(Kernels);
            BiasOptimizer = new Optimizer(Biases);
        }

        public override void CreateRandomParameters()
        {
            Kernels.GenerateNormalRandomValues(3);
        }

        /// <summary>
        /// Perform a convolution, followed by a ReLU activation, for a batch of feature maps.
        /// </summary>
        /// <returns></returns>
        public override Tensor Update(Tensor inputChannels)
        {
            int batchSize = inputChannels.GetLength(0);
            int Nx = inputChannels.GetLength(3);
            int Ny = inputChannels.GetLength(2);
            int Nx_out = Kx == Nx ? 1 : (Nx - Kx) / Stride + 1;
            int Ny_out = Ky == Ny ? 1 : (Ny - Ky) / Stride + 1;

            Tensor outputChannels = new Tensor(batchSize, NrOutputChannels, Ny_out, Nx_out);
            Convolute(inputChannels, batchSize, Nx, Ny, outputChannels);
            outputChannels.ReLU();

            return outputChannels;
        }

        /// <summary>
        /// Perform a convolution for a batch of images. No padding is used so the sizes to the 
        /// output chammels are smaller than for the input channels.
        /// </summary>
        /// <param name="inputChannels"></param>
        private void Convolute(Tensor inputChannels, int batchSize, int Nx, int Ny, Tensor outputChannels)
        {
            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int co = 0; co < NrOutputChannels; co++)
                {
                    // Move kernel over wole image
                    int nic = 0;
                    for (int ni = 0; ni <= Ny - Ky; ni += Stride)
                    {
                        int njc = 0;
                        for (int nj = 0; nj <= Nx - Kx; nj += Stride)
                        {
                            outputChannels[batch, co, nic, njc] = Biases[co];

                            // Convolute kernel
                            for (int ci = 0; ci < NrInputChannels; ci++)
                            {
                                for (int i = 0; i < Ky; i++)
                                {
                                    for (int j = 0; j < Kx; j++)
                                    {
                                        outputChannels[batch, co, nic, njc] += Kernels[co, ci, i, j] * inputChannels[batch, ci, ni + i, nj + j];
                                    }
                                }
                            }
                            njc++;
                        }
                        nic++;
                    }
                }
            }
        }

        public override void MakeTrainingStep(double learningRate, int step)
        {
            KernelOptimizer.MakeTrainingStep(learningRate, step, Kernels);
            BiasOptimizer.MakeTrainingStep(learningRate, step, Biases);
        }

    }
}
