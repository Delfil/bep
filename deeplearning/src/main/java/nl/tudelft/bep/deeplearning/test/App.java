package nl.tudelft.bep.deeplearning.test;

import java.io.IOException;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import nl.tudelft.bep.deeplearning.network.CNN;

public class App {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(10));
		String network1 = saveNetworks();
		String network2 = saveNetwork2();
		// String data1 = "100_Genes/100_Genes";
		String data1 = "cluster_v1";
		String data2 = data1;

		int epochs1 = 10;
		int epochs2 = 10;

		System.out.println("Start Data creation");
		test(network1, data1, 10, epochs1);
//		test(network2, data2, 10, epochs2);

		System.out.println("Start TTest...");
//		System.out.println("TTest: " + ResultUtil.getTTest(network1, data1, epochs1, network2, data2, epochs2));
		System.out.println("Average accuracy1:" + ResultUtil.getAverageAccuracy(network1, data1, epochs1));
//		System.out.println("Average accuracy2:" + ResultUtil.getAverageAccuracy(network2, data2, epochs2));
	}

	private static String saveNetworks() {
		return CNN
				.BuildExampleCNN(
						new ConvolutionLayer.Builder(5, 5).stride(2, 2).nOut(10).activation("identity").build(),
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						new DenseLayer.Builder().activation("relu").nOut(50).build(),
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5)
								.activation("softmax").build())
				.backprop(true).pretrain(false).finish().getFileName();
	}

	private static String saveNetwork2() {
		return CNN
				.BuildExampleCNN(
						new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(10).activation("identity").build(),
						new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2)
								.build(),
						new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5)
								.activation("softmax").build())
				.backprop(true).pretrain(false).finish().getFileName();
	}

	private static void test(String networkFile, String dataFile, int iterations, int epochs) {
		Tester test1 = new Tester(networkFile, dataFile);
		test1.start(iterations, epochs);
	}
}
