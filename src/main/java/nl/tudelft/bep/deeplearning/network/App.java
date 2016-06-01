package nl.tudelft.bep.deeplearning.network;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.result.ResultUtil;
import nl.tudelft.bep.deeplearning.network.result.csv.ComputeAverageAccuracyFiller;

public final class App {
	private static final int DEFAULT_EPOCHS = 10;
	private static final int DEFAULT_ITERATIONS = 20;
	private static final String DEFAULT_FILE_NAME = "results";

	private App() {
	}

	/**
	 * Produce a table with the average accuracies of all available network data
	 * set combinations.
	 * 
	 * @param args[0]
	 *            the number of epochs to select the data on
	 * @param args[1]
	 *            the number of iterations required to compute the average
	 * @param args[2]
	 *            the file name to save to
	 */
	public static void main(final String[] args) {

		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("1x1");

		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1");

		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(9, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("9x1");

		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(16, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("16x1");
		
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 1");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(5).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 5");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(10).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 10");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(15).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 15");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(25).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 25");
	
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1, stride");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 max pooling");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 avg pooling");
		
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(2, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("2x1");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("3x1");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(5, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x1");
		
		//To test next
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("1x1 20");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(30).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x1 30");
		
		run((args.length >= 1 && isInteger(args[0])) ? Integer.parseInt(args[0]) : DEFAULT_EPOCHS,
				(args.length >= 2 && isInteger(args[1])) ? Integer.parseInt(args[1]) : DEFAULT_ITERATIONS,
				(args.length >= 3 ? args[2] : DEFAULT_FILE_NAME));
	}

	private static boolean isInteger(final String string) {
		return string.matches("-?\\d+?");
	}

	private static void run(final int epochs, final int iterations, final String fileName) {
		ResultUtil.generateCSV(epochs, new ComputeAverageAccuracyFiller(iterations), fileName);
	}
}
