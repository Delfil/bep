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
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("only conv baseline");
		
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("only conv with stride 2x2");
		
//		CNN.buildExampleCNN(
//				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(2, 2).nOut(1).activation("identity").build(),
//				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
//				new DenseLayer.Builder().activation("relu").nOut(100).build(),
//				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
//				.backprop(true).pretrain(false).finish("conv with stride 2x2 and pooling max 2x2 2x2");
//		CNN.buildExampleCNN(
//				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(2, 2).nOut(1).activation("identity").build(),
//				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
//				new DenseLayer.Builder().activation("relu").nOut(100).build(),
//				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
//				.backprop(true).pretrain(false).finish("conv with stride 2x2 and pooling avg 2x2 2x2");
		
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("conv without stride and pooling max 2x2 2x2");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("conv without stride and pooling avg 2x2 2x2");
		
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 1");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(5).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 5");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(10).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 10");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(15).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 15");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(25).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 25");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(3, 3).nIn(1).stride(1, 1).nOut(30).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("nout 30");
		CNN.buildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(100).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("only conv baseline 2.0");

		run((args.length >= 1 && isInteger(args[0])) ? Integer.parseInt(args[0]) : DEFAULT_EPOCHS,
				(args.length >= 2 && isInteger(args[1])) ? Integer.parseInt(args[1]) : DEFAULT_ITERATIONS,
				(args.length >= 3 ? args[2] : DEFAULT_FILE_NAME));
	}

	private static boolean isInteger(final String string) {
		return string.matches("-?\\d+?");
	}

	private static void run(final int epochs, final int iterations, final String fileName) {
		ResultUtil.generateCSV(epochs, new ComputeAverageAccuracyFiller(iterations), fileName);
		//ResultUtil.generateLists(10, new ListAccuracy(), "dataList");
	}
}
