package nl.tudelft.bep.deeplearning;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration.ListBuilder;
import org.deeplearning4j.nn.conf.layers.Layer;

public class NNCBuilder extends NeuralNetConfiguration.Builder {

	protected boolean backprop = false;
	protected boolean pretrain = false;
	protected List<Layer> layers = new ArrayList<>();
	
	public NNCBuilder(){}

	public void add(Layer layer) {
		layers.add(layer);
	}

	@Override
	@Deprecated
	public ListBuilder list(int size) {
		return super.list(size);
	}

	@Override
	@Deprecated
	public NeuralNetConfiguration build() {
		return super.build();
	}

	public NNCBuilder backprop(boolean backprop) {
		this.backprop = backprop;
		return this;
	}

	public NNCBuilder pretrain(boolean pretrain) {
		this.pretrain = pretrain;
		return this;
	}

	public FinishedNNCBuilder finish() {
		return new FinishedNNCBuilder(this);
	}

	public List<Layer> getLayers() {
		return layers;
	}

	@Override
	public NNCBuilder clone() {
		NNCBuilder builder = (NNCBuilder) super.clone();
		builder.backprop = this.backprop;
		builder.pretrain = this.pretrain;
		builder.layers = new ArrayList<>(this.layers);
		return builder;
	}

	public boolean isBackprop() {
		return backprop;
	}

	public boolean isPretrain() {
		return pretrain;
	}
	
	
}
