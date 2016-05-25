package nl.tudelft.bep.deeplearning.network.result;

import java.io.IOException;

import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.SubsamplingLayer;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.result.csv.ComputeAverageAccuracyFiller;

public class App {
	public static void main(String[] args) throws IOException, ClassNotFoundException {
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(36, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("36 kernelsize 4 Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(36, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("36 kernelsize 4 Avg pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(36, 1).nIn(1).stride(4, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("36 kernelsize No pooling with 4 stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernelsize 4 Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernelsize 4 Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(4, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernelsize No pooling with 4 stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(4, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernelsize 4 stride 4 Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(4, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernelsize 4 stride 4 Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(16, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("16 kernelsize 4 Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(16, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("16 kernelsize 4 Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4 kernelsize 4 Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(4, 1).stride(4, 1).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4 kernelsize 4 Avg pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 1).nIn(1).stride(2, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4 kernelsize No pooling with 2 stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(16, 1).nIn(1).stride(4, 1).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("16 kernelsize No pooling with 4 stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(25, 1).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("25 kernel Simple Convolutional Neuron Network");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("Simple Neuron Network");
		
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(20), "results");
	}
}
