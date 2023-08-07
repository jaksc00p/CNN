using CNN.Utils;
using static CNN.NeuralNetwork;

namespace CNN
{
    public class ConvolutionNeuralNet
    {
        private readonly List<CNNLayer> CNNLayers = new();
        private NeuralNetwork net = null;

        public void AddRandomConvolutionLayer(int kx, int ky, int stride, int nrInputChannels, int nrOutputChannels)
        {
            ConvolutionLayer CL = new(kx, ky, stride, nrInputChannels, nrOutputChannels);
            CL.CreateRandomParameters();
            CNNLayers.Add(CL);
        }

        public void AddPoolingLayer(int kx, int ky, int stride, int nrChannels)
        {
            PoolingLayer PL = new(kx, ky, stride, nrChannels);
            CNNLayers.Add(PL);
        }

        public void AddRandomNeuralNetwork(int batchSize, int[] layerSizes, string activationFunction,
            string outputLayerActivationFunction, double dropoutRate)
        {
            net = GenerateRandomNeuralNetwork(batchSize, layerSizes, activationFunction,
                outputLayerActivationFunction, dropoutRate);
        }

        public void Train(double[,,] inputImages, double[] values, int startIndex, int endIndex, double learningRate,
            int nrTrainingSteps, int batchSize, int nrEpochs)
        {
            int nrInputImages = endIndex - startIndex;
            int yRes = inputImages.GetLength(1);
            int xRes = inputImages.GetLength(2);
            int nrInputChannels = 1;

            Console.WriteLine("Training");
            for (int epoch = 0; epoch < nrEpochs; epoch++)
            {
                Console.WriteLine();
                Console.WriteLine("Epoch: " + (epoch + 1).ToString());
                for (int batch = 0; batch < nrInputImages / batchSize; batch++)
                {
                    Console.WriteLine();
                    Console.WriteLine("Batch: " + (batch + 1).ToString());
                    var inputData = new Tensor(batchSize, nrInputChannels, yRes, xRes);
                    var outputData = new double[batchSize];
                    for (int image = 0; image < batchSize; image++)
                    {
                        int channel = 0;
                        for (int i = 0; i < yRes; i++)
                        {
                            for (int j = 0; j < xRes; j++)
                            {
                                inputData[image, channel, i, j] = new Rev(inputImages[startIndex + batch * batchSize + image, i, j]);
                            }
                        }
                        outputData[image] = values[batch * batchSize + image];
                    }

                    int iter = 0;
                    while (iter++ < nrTrainingSteps)
                    {
                        double loss = MakeTrainingStep(inputData, outputData, learningRate, iter);
                        if (iter % 10 == 0)
                            Console.WriteLine(loss.ToString());
                    }

                }
            }
        }

        public void Infer(double[,,] inputImages, double[] values, int startIndex, int endIndex)
        {
            int yRes = inputImages.GetLength(1);
            int xRes = inputImages.GetLength(2);
            int nrInputChannels = 1;

            var inputData = new Tensor(1, nrInputChannels, yRes, xRes);
            double[] outputData = new double[1];

            Console.WriteLine();
            Console.WriteLine("Inference");
            for (int image = startIndex; image < endIndex; image++)
            {
                int channel = 0;
                for (int i = 0; i < yRes; i++)
                {
                    for (int j = 0; j < xRes; j++)
                    {
                        inputData[0, channel, i, j] = new Rev(inputImages[image, i, j]);
                    }
                }
                outputData[0] = values[image];
                EvaluateLossFunction(inputData, outputData, false);
            }

        }

        public double MakeTrainingStep(Tensor inputImages, double[] observations, double learningRate, int step)
        {
            var loss = EvaluateLossFunction(inputImages, observations, true);
            loss.CalculateDerivative(1);
            foreach (CNNLayer cnnlayer in CNNLayers)
            {
                cnnlayer.MakeTrainingStep(learningRate, step);
            }
            net.MakeTrainingStep(learningRate, step);

            return loss;
        }

        public Rev EvaluateLossFunction(Tensor inputImages, double[] observations, bool isTraining)
        {
            var data = inputImages;
            foreach (CNNLayer cnnlayer in CNNLayers)
            {
                data = cnnlayer.Update(data);
            }
            net.Evaluate(Tensor.Flatten(data), isTraining);

            return net.EvaluateLossFunction(observations, isTraining);
        }

    }
}
