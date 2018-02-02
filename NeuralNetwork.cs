using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

/// <summary>
/// Simple MLP Neural Network, BackPropogation
/// </summary>
public class NeuralNetwork
{
    private int[] _layer; //layer information
    private Layer[] _layers; //layers in the network

    /// <summary>
    /// Save NeuralNetwork values
    /// </summary>
    /// <param name="fileName">file name or path</param>
    public void SaveNetwork(string fileName)
    {
        var text = "";
        text += _layer.Length.ToString();
        text += " ";
        text = _layer.Aggregate(text, (current, x) => current + (x + " "));
        using (var file = new StreamWriter(fileName))
        {
            file.Write(text);
            file.WriteLine("");
            file.WriteLine(_layers.Length.ToString());
            foreach (var t in _layers)
            {
                var tex = t.SaveinString();
                foreach (var x in tex)
                {
                    file.WriteLine(x);
                }
            }
        }
    }

    /// <summary>
    /// Loads NeuralNetwork values
    /// </summary>
    /// <param name="fileName">file name or path</param>
    public void LoadNetwork(string fileName)
    {
        var data = File.ReadAllLines(fileName);
        var spl = data[0].Split(' ');
        var pos = 2;
        _layer = new int[int.Parse(spl[0])];
        for (var i = 0; i < _layer.Length; i++)
        {
            _layer[i] = int.Parse(spl[i + 1]);
        }
        _layers = new Layer[int.Parse(data[1])];
        for (var i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Layer(_layer[i], _layer[i + 1]);
            for (var y = 0; y < _layer[i + 1]; y++)
            {
                spl = data[pos].Split(' ');
                for (var x = 0; x < _layer[i]; x++)
                {
                    _layers[i].weights[y, x] = float.Parse(spl[x]);
                }
                pos++;
            }
        }
    }
    /// <summary>
    /// Constructor setting up layers
    /// </summary>
    /// <param name="layer">Layers of this network</param>
    public NeuralNetwork(IReadOnlyList<int> layer)
    {
        //deep copy layers
        _layer = new int[layer.Count];
        for (var i = 0; i < layer.Count; i++)
        {_layer[i] = layer[i];}

        //creates neural layers
        _layers = new Layer[layer.Count - 1];

        for (var i = 0; i < _layers.Length; i++)
        {
            _layers[i] = new Layer(layer[i], layer[i + 1]);
        }
    }

    /// <summary>
    /// High level feedforward for this network
    /// </summary>
    /// <param name="inputs">Inputs to be feed forwared</param>
    /// <returns></returns>
    public float[] FeedForward(float[] inputs)
    {
        //feed forward
        _layers[0].FeedForward(inputs);
        for (var i = 1; i < _layers.Length; i++)
        {
            _layers[i].FeedForward(_layers[i - 1].outputs);
        }

        return _layers[_layers.Length - 1].outputs; //return output of last layer
    }

    /// <summary>
    /// High level back porpagation
    /// Note: It is expexted the one feed forward was done before this back prop.
    /// </summary>
    /// <param name="expected">The expected output form the last feedforward</param>
    public void BackProp(float[] expected)
    {
        // run over all layers backwards
        for (var i = _layers.Length - 1; i >= 0; i--)
        {
            if (i == _layers.Length - 1)
            {
                _layers[i].BackPropOutput(expected); //back prop output
            }
            else
            {
                _layers[i].BackPropHidden(_layers[i + 1].gamma, _layers[i + 1].weights); //back prop hidden
            }
        }

        //Update weights
        foreach (var th in _layers)
        {
            th.UpdateWeights();
        }
    }

    /// <summary>
    /// Each individual layer in the ML{
    /// </summary>
    private class Layer
    {
        int numberOfInputs; //# of neurons in the previous layer
        int numberOfOuputs; //# of neurons in the current layer


        public float[] outputs; //outputs of this layer
        private float[] _inputs; //inputs in into this layer
        public float[,] weights; //weights of this layer
        private float[,] weightsDelta; //deltas of this layer
        public float[] gamma; //gamma of this layer
        private float[] error; //error of the output layer

        private static Random random = new Random(); //Static random class variable

        /// <summary>
        /// Constructor initilizes vaiour data structures
        /// </summary>
        /// <param name="numberOfInputs">Number of neurons in the previous layer</param>
        /// <param name="numberOfOuputs">Number of neurons in the current layer</param>
        public Layer(int numberOfInputs, int numberOfOuputs)
        {
            this.numberOfInputs = numberOfInputs;
            this.numberOfOuputs = numberOfOuputs;

            //initilize datastructures
            outputs = new float[numberOfOuputs];
            _inputs = new float[numberOfInputs];
            weights = new float[numberOfOuputs, numberOfInputs];
            weightsDelta = new float[numberOfOuputs, numberOfInputs];
            gamma = new float[numberOfOuputs];
            error = new float[numberOfOuputs];

            InitilizeWeights(); //initilize weights
        }
        /// <summary>
        /// Return a string of info
        /// </summary>
        public string[] SaveinString()
        {
            var s = new string[numberOfOuputs];
            var pos = 0;
            for (var y = 0; y < numberOfOuputs; y++)
            {
                for (var x = 0; x < numberOfInputs; x++)
                {
                    s[pos] += weights[y, x] + " ";
                }
                pos++;
            }
            return s;
        }
        /// <summary>
        /// Initilize weights between -0.5 and 0.5
        /// </summary>
        private void InitilizeWeights()
        {
            for (var i = 0; i < numberOfOuputs; i++)
            {
                for (var j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] = (float)random.NextDouble() - 0.5f;
                }
            }
        }

        /// <summary>
        /// Feedforward this layer with a given input
        /// </summary>
        /// <param name="inputs">The output values of the previous layer</param>
        /// <returns></returns>
        public float[] FeedForward(float[] inputs)
        {
            _inputs = inputs;// keep shallow copy which can be used for back propagation

            //feed forwards
            for (var i = 0; i < numberOfOuputs; i++)
            {
                outputs[i] = 0;
                for (var j = 0; j < numberOfInputs; j++)
                {
                    outputs[i] += inputs[j] * weights[i, j];
                }

                outputs[i] = (float)Math.Tanh(outputs[i]);
            }

            return outputs;
        }

        /// <summary>
        /// TanH derivate 
        /// </summary>
        /// <param name="value">An already computed TanH value</param>
        /// <returns></returns>
        private static float TanHDer(float value)
        {
            return 1 - (value * value);
        }

        /// <summary>
        /// Back propagation for the output layer
        /// </summary>
        /// <param name="expected">The expected output</param>
        public void BackPropOutput(float[] expected)
        {
            //Error dervative of the cost function
            for (var i = 0; i < numberOfOuputs; i++)
                error[i] = outputs[i] - expected[i];

            //Gamma calculation
            for (var i = 0; i < numberOfOuputs; i++)
                gamma[i] = error[i] * TanHDer(outputs[i]);

            //Caluclating detla weights
            for (var i = 0; i < numberOfOuputs; i++)
            {
                for (var j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * _inputs[j];
                }
            }
        }

        /// <summary>
        /// Back propagation for the hidden layers
        /// </summary>
        /// <param name="gammaForward">the gamma value of the forward layer</param>
        /// <param name="weightsFoward">the weights of the forward layer</param>
        public void BackPropHidden(float[] gammaForward, float[,] weightsFoward)
        {
            //Caluclate new gamma using gamma sums of the forward layer
            for (var i = 0; i < numberOfOuputs; i++)
            {
                gamma[i] = 0;

                for (var j = 0; j < gammaForward.Length; j++)
                {
                    gamma[i] += gammaForward[j] * weightsFoward[j, i];
                }

                gamma[i] *= TanHDer(outputs[i]);
            }

            //Caluclating detla weights
            for (var i = 0; i < numberOfOuputs; i++)
            {
                for (var j = 0; j < numberOfInputs; j++)
                {
                    weightsDelta[i, j] = gamma[i] * _inputs[j];
                }
            }
        }

        /// <summary>
        /// Updating weights
        /// </summary>
        public void UpdateWeights()
        {
            for (var i = 0; i < numberOfOuputs; i++)
            {
                for (var j = 0; j < numberOfInputs; j++)
                {
                    weights[i, j] -= weightsDelta[i, j] * 0.033f;
                }
            }
        }
    }
}
