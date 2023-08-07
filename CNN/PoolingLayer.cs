using CNN.Utils;

namespace CNN
{
    public class PoolingLayer : CNNLayer
    {
        /// <summary>
        /// Create a (max) pooling layer with a given window size and stride length (same in both directions).
        /// </summary>
        /// <param name="kx"></param>
        /// <param name="ky"></param>
        /// <param name="stride"></param>
        /// <param name="nrChannels"></param>
        public PoolingLayer(int kx, int ky, int stride, int nrChannels)
        {
            Kx = kx;
            Ky = ky;
            Stride = stride;
            NrInputChannels = nrChannels;
            NrOutputChannels = nrChannels;
        }

        /// <summary>
        /// Perform a downsampling by implementing max pooling for a batch of feature maps. 
        /// </summary>
        /// <param name="inputChannels"></param>
        /// <returns></returns>
        public override Tensor Update(Tensor inputChannels)
        {
            int batchSize = inputChannels.GetLength(0);
            int nrChannels = inputChannels.GetLength(1);
            int Nx = inputChannels.GetLength(3);
            int Ny = inputChannels.GetLength(2);
            int Nx_out = Kx == Nx ? 1 : (Nx - Kx) / Stride + 1;
            int Ny_out = Ky == Ny ? 1 : (Ny - Ky) / Stride + 1;

            Tensor outputChannels = new Tensor(batchSize, nrChannels, Ny_out, Nx_out);

            for (int batch = 0; batch < batchSize; batch++)
            {
                for (int co = 0; co < nrChannels; co++)
                {
                    // Move window over wole image
                    int nic = 0;
                    for (int ni = 0; ni <= Ny - Ky; ni += Stride)
                    {
                        int njc = 0;
                        for (int nj = 0; nj <= Nx - Kx; nj += Stride)
                        {
                            outputChannels[batch, co, nic, njc] = inputChannels[batch, co, ni, nj];
                            for (int i = 0; i < Ky; i++)
                            {
                                for (int j = 0; j < Kx; j++)
                                {
                                    if (inputChannels[batch, co, ni + i, nj + j] > outputChannels[batch, co, nic, njc])
                                        outputChannels[batch, co, nic, njc] = inputChannels[batch, co, ni + i, nj + j];
                                }
                            }
                            njc++;
                        }
                        nic++;
                    }
                }
            }

            return outputChannels;
        }
    }
}
