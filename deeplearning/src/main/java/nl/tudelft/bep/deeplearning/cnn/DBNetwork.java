package nl.tudelft.bep.deeplearning.cnn;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import nl.tudelft.bep.deeplearning.datafetcher.MatrixDataFetcher;
import nl.tudelft.bep.deeplearning.datafetcher.MatrixDatasetIterator;

import java.util.Collections;

/**
 * Created by agibsonccc on 9/11/14.
 *
 * ***** NOTE: This example has not been tuned. It requires additional work to produce sensible results *****
 */
public class DBNetwork {

    private static Logger log = LoggerFactory.getLogger(DBNetwork.class);

    public static void main(String[] args) throws Exception {
        int numRows = 28;
        int numColumns = 28;
        int outputNum = 10;
        int numSamples = 60000;
        int batchSize = 100;
        int iterations = 10;
        int seed = 123;
        int listenerFreq = batchSize / 5;

        log.info("Load data....");
    	String fileName = "/1200_s_navg_can";
		MatrixDataFetcher fetcher = new MatrixDataFetcher(fileName, seed, 0.0, 0.75);
		DataSetIterator mnistTrain = new MatrixDatasetIterator(batchSize, fetcher);
		DataSetIterator mnistTest = new MatrixDatasetIterator(batchSize,
				new MatrixDataFetcher(fileName, seed, 0.75, 1.0));

		numRows = fetcher.getWidth();
		numColumns = fetcher.getHeight();
		outputNum = fetcher.getOutputNum();

        log.info("Build model....");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
           .seed(seed)
           .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
           .gradientNormalizationThreshold(1.0)
           .iterations(iterations)
           .momentum(0.5)
           .momentumAfter(Collections.singletonMap(3, 0.9))
           .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
           .list(2)
           .layer(0, new RBM.Builder().nIn(numRows*numColumns).nOut(200)
                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                         .visibleUnit(RBM.VisibleUnit.BINARY)
                         .hiddenUnit(RBM.HiddenUnit.BINARY)
                         .build())
//           .layer(1, new RBM.Builder().nIn(500).nOut(250)
//                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
//                         .visibleUnit(RBM.VisibleUnit.BINARY)
//                         .hiddenUnit(RBM.HiddenUnit.BINARY)
//                         .build())
//           .layer(2, new RBM.Builder().nIn(250).nOut(200)
//                         .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
//                         .visibleUnit(RBM.VisibleUnit.BINARY)
//                         .hiddenUnit(RBM.HiddenUnit.BINARY)
//                         .build())
           .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                         .nIn(200).nOut(outputNum).build())
           .pretrain(true).backprop(false)
           .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

        log.info("Train model....");
        model.fit(mnistTrain); // achieves end to end pre-training

        log.info("Evaluate model....");
        Evaluation eval = new Evaluation(outputNum);

        
        while(mnistTest.hasNext()) {
            DataSet testMnist = mnistTest.next();
            INDArray predict2 = model.output(testMnist.getFeatureMatrix());
            eval.eval(testMnist.getLabels(), predict2);
        }

        log.info(eval.stats());
        log.info("****************Example finished********************");

    }

}
