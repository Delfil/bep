package nl.tudelft.bep.deeplearning.network.builder;

import java.util.ArrayList;
import java.util.List;

import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.Layer;

public class NNCBuilder extends NeuralNetConfiguration.Builder {

	private boolean backprop = false;
	private boolean pretrain = false;
	private List<Layer> layers = new ArrayList<>();

	public NNCBuilder() {
	}

	public void add(final Layer layer) {
		this.getLayers().add(layer);
	}

	@Override
	@Deprecated
	public NeuralNetConfiguration build() {
		return super.build();
	}

	public NNCBuilder backprop(final boolean backprop) {
		this.backprop = backprop;
		return this;
	}

	public NNCBuilder pretrain(final boolean pretrain) {
		this.setPretrain(pretrain);
		return this;
	}

	public FNNCBuilder finish() {
		return this.finish("");
	}

	public FNNCBuilder finish(final String description) {
		return new FNNCBuilder(this, description);
	}

	public List<Layer> getLayers() {
		return this.layers;
	}

	@Override
	public NNCBuilder clone() {
		NNCBuilder builder = (NNCBuilder) super.clone();
		builder.backprop = this.backprop;
		builder.setPretrain(this.isPretrain());
		builder.layers = new ArrayList<>(this.getLayers());
		return builder;
	}

	public boolean isBackprop() {
		return this.backprop;
	}

	public boolean isPretrain() {
		return this.pretrain;
	}

	/**
	 * @param backprop
	 *            the backprop to set
	 */
	public void setBackprop(final boolean backprop) {
		this.backprop = backprop;
	}

	/**
	 * @param pretrain
	 *            the pretrain to set
	 */
	public void setPretrain(final boolean pretrain) {
		this.pretrain = pretrain;
	}
}
