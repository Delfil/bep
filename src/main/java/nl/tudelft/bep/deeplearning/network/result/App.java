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
		ResultUtil.generateCSV(10, new ComputeAverageAccuracyFiller(20), "results");
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(6, 6).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("6x6 kernelsize Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(6, 6).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("6x6 kernelsize Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(6, 6).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("6x6 kernelsize No pooling with stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernelsize Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernelsize Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernelsize No pooling with stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernelsize 2x2 stride Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernelsize 2x2 stride Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 4).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x4 kernelsize Max pooling");

		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 4).nIn(1).stride(1, 1).nOut(20).activation("identity").build(),
				new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.AVG).kernelSize(2, 2).stride(2, 2).build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x4 kernelsize Avg pooling");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(4, 4).nIn(1).stride(2, 2).nOut(20).activation("identity").build(),
				new DenseLayer.Builder().activation("relu").nOut(20).build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("4x4 kernelsize No pooling with stride compensation");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("5x5 kernel Simple Convolutional Neuron Network");
		
		CNN.BuildExampleCNN(
				new ConvolutionLayer.Builder(1, 1).nIn(1).stride(1, 1).nOut(1).activation("identity").build(),
				new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD).nOut(5).activation("softmax").build())
				.backprop(true).pretrain(false).finish("Simple Neuron Network");
	}
}
