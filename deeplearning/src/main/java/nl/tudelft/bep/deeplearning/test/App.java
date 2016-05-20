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
		addNetworks();
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(10), "results");
	}

	static String addNetworks() {
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
}
