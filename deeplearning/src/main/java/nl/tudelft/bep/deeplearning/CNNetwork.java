package nl.tudelft.bep.deeplearning;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.deeplearning4j.nn.conf.layers.setup.ConvolutionLayerSetup;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CNNetwork {

	private static final Logger log = LoggerFactory.getLogger(CNNetwork.class);

	/**
	 * Based on LenetMnistExample by agibsonccc on 9/16/15.
	 */
	public static void main(String[] args) throws Exception {
		int batchSize = 100;
		int nEpochs = 10;
		int seed = 123;

		int width = 113;
		int height = width;
		
		log.info("Load data....");
		MatrixDataFetcher fetcher = new MatrixDataFetcher("", false, seed, width, height);
		DataSetIterator mnistTrain = new MatrixDatasetIterator(batchSize,
				fetcher);
		DataSetIterator mnistTest = new MatrixDatasetIterator(batchSize,
				new MatrixDataFetcher("", false, seed, width, height));

		int outputNum = fetcher.getOutputNum();
		

		
		
		
		
		
		
		
		
		
		
		
		
		log.info("Build model....");
		MultiLayerConfiguration.Builder builder = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .iterations(1)
                .regularization(true).l2(0.0005)
                .learningRate(0.01)
                .weightInit(WeightInit.XAVIER)
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(Updater.NESTEROVS).momentum(0.9)
                .list(6)
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(20)
                        .activation("identity")
                        .build())
                .layer(1, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .nIn(1)
                        .stride(1, 1)
                        .nOut(50)
                        .activation("identity")
                        .build())
                .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX)
                        .kernelSize(2,2)
                        .stride(2,2)
                        .build())
                .layer(4, new DenseLayer.Builder()
                		.activation("relu")
                        .nOut(50).build())
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        .nOut(outputNum)
                        .activation("softmax")
                        .build())
                .backprop(true).pretrain(false);
        new ConvolutionLayerSetup(builder,width,height,1);

		MultiLayerConfiguration conf = builder.build();
		MultiLayerNetwork model = new MultiLayerNetwork(conf);
		model.init();

		log.info("Train model....");
		model.setListeners(new ScoreIterationListener(1));
		for (int i = 0; i < nEpochs; i++) {
			model.fit(mnistTrain);
			log.info("*** Completed epoch {} ***", i);

			log.info("Evaluate model....");
			Evaluation eval = new Evaluation(outputNum);
			while (mnistTest.hasNext()) {
				DataSet ds = mnistTest.next();
				INDArray output = model.output(ds.getFeatureMatrix());
				eval.eval(ds.getLabels(), output);
			}
			log.info(eval.stats());
			mnistTest.reset();
		}
		log.info("****************Example finished********************");
	}
}
