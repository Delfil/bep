package nl.tudelft.bep.deeplearning.deepbelief;

import java.io.File;
import java.io.IOException;
import java.util.Collections;
import java.util.Scanner;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
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
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


public class DBNetwork {
	
	private static int rows = 112;
	private static int cols = 112;
	private static int iter = 100;
	private static int seed = 100;
	private static int outputNum = 5;
	private static int batch = 100;
	private static Logger log = LoggerFactory.getLogger(DBNetwork.class);
	private static int listenerFreq = batch / 100;
	

	public static void main(String[] args) throws IOException, InterruptedException {

		DataSetIterator data = getData("112x112_100sampledata.meta");
		DataSet temp = data.next();
		temp.normalizeZeroMeanZeroUnitVariance();
		temp.shuffle();
		SplitTestAndTrain split = temp.splitTestAndTrain(0.65);
		DataSet trainData = split.getTrain();
		DataSet testData = split.getTest();
		
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue)
                .gradientNormalizationThreshold(1.0)
                .iterations(iter)
                .momentum(0.5)
                .momentumAfter(Collections.singletonMap(3, 0.9))
                .optimizationAlgo(OptimizationAlgorithm.CONJUGATE_GRADIENT)
                .list(4)
                .layer(0, new RBM.Builder().nIn(rows*cols).nOut(500)
                              .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                              .visibleUnit(RBM.VisibleUnit.BINARY)
                              .hiddenUnit(RBM.HiddenUnit.BINARY)
                              .build())
                .layer(1, new RBM.Builder().nIn(500).nOut(250)
                              .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                              .visibleUnit(RBM.VisibleUnit.BINARY)
                              .hiddenUnit(RBM.HiddenUnit.BINARY)
                              .build())
                .layer(2, new RBM.Builder().nIn(250).nOut(200)
                              .weightInit(WeightInit.XAVIER).lossFunction(LossFunction.RMSE_XENT)
                              .visibleUnit(RBM.VisibleUnit.BINARY)
                              .hiddenUnit(RBM.HiddenUnit.BINARY)
                              .build())
                .layer(3, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).activation("softmax")
                              .nIn(200).nOut(outputNum).build())
                .pretrain(true).backprop(false)
                .build();

             MultiLayerNetwork model = new MultiLayerNetwork(conf);
             model.init();
             model.setListeners(Collections.singletonList((IterationListener) new ScoreIterationListener(listenerFreq)));

             log.info("Train model....");
             model.fit(trainData); // achieves end to end pre-training


             log.info("Evaluate model....");
             @SuppressWarnings("rawtypes")
             Evaluation eval = new Evaluation(outputNum);
             INDArray predict2 = model.output(testData.getFeatureMatrix());
             eval.eval(testData.getLabels(), predict2);
            
             log.info(eval.stats());
	}
	
	private static DataSetIterator getData(String meta) throws IOException, InterruptedException {
        //We do not want to skip any lines as we have a meta file
		int numLinesToSkip = 0;
		
		//Reading the meta data file
		File file = new ClassPathResource(meta).getFile();
		Scanner sc = new Scanner(file);
		String fileName = sc.next();
		int batchSize = sc.nextInt();
		int labelIndex = sc.nextInt();
		int numClasses = sc.nextInt();
		sc.close();
        String delimiter = ",";
        
        //Create the record reader based on the meta data.
        RecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource(fileName).getFile()));
        	
        return new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);
    }
}
