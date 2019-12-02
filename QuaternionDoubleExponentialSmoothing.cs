using System.Numerics;
using MathNet.Numerics.LinearAlgebra;
namespace DomMandy.Math
{
    /// <summary>
    /// object intended to cover quaternion moving average filters
    /// see: https://en.wikipedia.org/wiki/Moving_average
    /// </summary>
    internal class QuaternionDoubleExponentialSmoothing
    {
        public QuaternionDoubleExponentialSmoothing(float dataSmoothing = .1f, float trendSmoothing = .9f)
        {
            this.dataSmoothing = dataSmoothing;
            this.trendSmoothing = trendSmoothing;
        }
        readonly float dataSmoothing = .1f;
        readonly float trendSmoothing = .9f;
        //!< current weighted sum of quaternion self-outer-product matrices
        Matrix<float> sum;
        Matrix<float> oldSum;
        //!< prediction of trend changing sum over time
        Matrix<float> prediction;
        //!< 
        Matrix<float> smoothed;
        //!< current quaternion in vector form
        Vector<float> qVector;
        /// <summary>
        /// a first-order infinite impulse response filter that applies weighting factors which decrease exponentially.
        /// The weighting for each older datum decreases exponentially, never reaching zero.
        /// see: https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing
        /// </summary>
        /// <param name="quaternion">data sample for the current frame</param>
        /// <param name="deltaTime">time in seconds since the last frame</param>
        /// <returns>double exponentially smoothed moving average</returns>
        public Quaternion Add(Quaternion quaternion, float deltaTime)
        {
            #region bookkeeping
            // lazy initialization
            qVector = qVector ?? Vector<float>.Build.Dense(4);
            sum = sum ?? Matrix<float>.Build.Dense(4, 4, 0f);
            oldSum = oldSum ?? sum;
            prediction = prediction ?? Matrix<float>.Build.Dense(4, 4, 0f);
            // put quaternion into local vector structure
            for (int i = 0; i < qVector.Count; i++)
            {
                qVector[i] = quaternion[i];
            }
            #endregion
            #region framerateIndependentWeights see: http://www.rorydriscoll.com/2016/03/07/frame-rate-independent-damping-using-lerp/
            var sData = Mathf.Pow(dataSmoothing, deltaTime);
            var sTrend = Mathf.Pow(trendSmoothing, deltaTime);
            #endregion
            #region quaternionAveraging see: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
            // double exponential smoothing
            oldSum = sum;
            sum = sData * (sum + prediction) + (1f - sData) * qVector.OuterProduct(qVector);
            prediction = sTrend * prediction + (1f - sTrend) * (sum - oldSum);
            // eigen value/vector decomposition
            var evd = sum.Evd(Symmetricity.Symmetric);
            var eigenValues = evd.EigenValues;
            // find maximum eigenvalue
            var max = eigenValues[0].Real;
            var index = 0;
            for (int i = 1; i < eigenValues.Count; i++)
            {
                index = eigenValues[i].Real > max
                    ? i
                    : index;
            }
            // eigenvector corresponding to maximum eigenvalue
            var eigenVector = evd.EigenVectors.Column(index);
            #endregion
            // convert back into quaternion structure
            var average = Quaternion.Normalize(new Quaternion(eigenVector[0], eigenVector[1], eigenVector[2], eigenVector[3]));
            return average;
        }
    }
}
