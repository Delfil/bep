package nl.tudelft.bep.deeplearning;

import java.io.File;
import java.io.IOException;
import java.util.Scanner;

import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class nnIris {
	private static Logger log = LoggerFactory.getLogger(nnIris.class);
	
	public static void main(String[] args) throws IOException, InterruptedException {
		  final int numInputs = 12750;
	        int outputNum = 5;
	        int iterations = 1000;
	        long seed = 6;

	        DataSetIterator data = getData("alldata.meta");
	        DataSet next = data.next();

	        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
	                .iterations(iterations)
	                .learningRate(0.01)
	                .regularization(true).l2(1e-4)
	                .list(3)
	                .layer(0, new DenseLayer.Builder().nIn(numInputs).nOut(5)
	                        .activation("tanh")
	                        .weightInit(WeightInit.XAVIER)
	                        .build())
	                .layer(1, new DenseLayer.Builder().nIn(5).nOut(5)
	                        .activation("tanh")
	                        .weightInit(WeightInit.XAVIER)
	                        .build())
	                .layer(2, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
	                        .weightInit(WeightInit.XAVIER)
	                        .activation("softmax")
	                        .nIn(5).nOut(outputNum).build())
	                .backprop(true).pretrain(false)
	                .build();

	        //run the model
	        MultiLayerNetwork model = new MultiLayerNetwork(conf);
	        model.init();
	        model.setListeners(new ScoreIterationListener(100));

	        //Normalize the full data set. Our DataSet 'next' contains the full 150 examples
	        next.normalizeZeroMeanZeroUnitVariance();
	        next.shuffle();
	        //split test and train
	        SplitTestAndTrain testAndTrain = next.splitTestAndTrain(0.65);  //Use 65% of data for training

	        DataSet trainingData = testAndTrain.getTrain();
	        model.fit(trainingData);

	        //evaluate the model on the test set
	        Evaluation eval = new Evaluation(5);
	        DataSet test = testAndTrain.getTest();
	        INDArray output = model.output(test.getFeatureMatrix());
	        eval.eval(test.getLabels(), output);
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
