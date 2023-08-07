using CNN.Utils;

namespace CNN
{
    /// <summary>
    /// Contains values, weights and biases for the actual layer. The biases and weights can be trained.
    /// Regularization can be performed through the use of dropouts (except for the output layer).
    /// Activation functions can be either Linear, ReLU, or Softmax.
    /// </summary>
    public class NeuralNetLayer
    {
        public Tensor V;    // Value vector
        private Tensor B;   // Bias vector
        private Tensor T;   // Connection matrix to previous layer

        private Optimizer TOptimizer, BOptimizer;

        private NeuralNetLayer previousLayer;
        private ActivationFunctions activationFunction = ActivationFunctions.ReLU;

        private double DropoutRate { get; set; }
        public int LayerSize { get; private set; }
        public int BatchSize { get; private set; }


        public enum ActivationFunctions
        {
            Linear,
            ReLU,
            Softmax,
        }

        /// <summary>
        /// Contructor for the layer
        /// </summary>
        /// <param name="activationFunction">Can be either Linear, ReLU, or Softmax</param>
        public NeuralNetLayer(int batchSize, NeuralNetLayer previousLayer, int layerSize, string activationFunction, double dropoutRate)
        {
            this.previousLayer = previousLayer;
            LayerSize = layerSize;
            BatchSize = batchSize;

            if (previousLayer != null)
            {
                B = new Tensor(LayerSize);
                T = new Tensor(LayerSize, previousLayer.LayerSize);

                TOptimizer = new Optimizer(T);
                BOptimizer = new Optimizer(B);
            }

            if (!Enum.TryParse(activationFunction, out this.activationFunction))
                throw new ArgumentException("Unknown activation function: " + activationFunction);

            DropoutRate = dropoutRate;
        }

        public void CreateRandomParameters()
        {
            if (previousLayer == null)
                return;

            T.GenerateNormalRandomValues(1);
        }

        public void SetInputBatchSize(int batchSize)
        {
            V = new Tensor(batchSize, LayerSize);
        }

        public void SetInputValue(int batch, int i, Rev val)
        {
            if (previousLayer != null)
                throw new InvalidOperationException("Inputs can only be set on first layer");

            V[batch, i] = val;
        }

        public Rev GetOutputValue(int batch, int i)
        {
            return V[batch, i];
        }

        /// <summary>
        /// Perform V = sigma(T * V0 + B)
        /// </summary>
        public void Update(bool isTraining)
        {
            if (V == null)
                V = new Tensor(BatchSize, LayerSize);

            if (previousLayer == null)
                return;

            if (isTraining && DropoutRate > 0)
                previousLayer.SetDropoutNodes();

            if (activationFunction == ActivationFunctions.Linear)
            {
                UpdateLinear();
            }
            else if (activationFunction == ActivationFunctions.ReLU)
            {
                UpdateReLU();
            }
            else if (activationFunction == ActivationFunctions.Softmax)
            {
                UpdateSoftmax();
            }

        }

        private void UpdateLinear()
        {
            Tensor V0 = previousLayer.V;
            V = Tensor.MatVecMul(T, V0);
            V = V.VecAdd(B);
        }

        private void UpdateReLU()
        {
            Tensor V0 = previousLayer.V;
            V = Tensor.MatVecMul(T, V0);
            V = V.VecAdd(B);
            V.ReLU();
        }

        private void UpdateSoftmax()
        {
            Tensor V0 = previousLayer.V;
            V = Tensor.MatVecMul(T, V0);
            V = V.VecAdd(B);
            V = V.Softmax();
        }

        /// <summary>
        /// Set dropout nodes for current layer randomly
        /// </summary>
        /// <param name="isTraining"></param>
        public void SetDropoutNodes()
        {
            bool[] DropoutMask = new bool[LayerSize];
            for (int i = 0; i < LayerSize; i++)
            {
                DropoutMask[i] = RandomNumbers.Instance.GetNextUniformNumber() < DropoutRate;
            }

            V = V.Dropout(DropoutMask, DropoutRate);
        }

        public void MakeTrainingStep(double learningRate, int step)
        {
            if (previousLayer == null)
                return;

            TOptimizer.MakeTrainingStep(learningRate, step, T);
            BOptimizer.MakeTrainingStep(learningRate, step, B);
        }

    }
}
