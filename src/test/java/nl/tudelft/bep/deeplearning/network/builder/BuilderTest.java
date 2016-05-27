package nl.tudelft.bep.deeplearning.network.builder;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration.Builder;
import org.deeplearning4j.nn.conf.layers.ConvolutionLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

public class BuilderTest {

	private String testFile;
	private NNCBuilder builder;

	@Before
	public void init() {
		this.builder = getTestBuilder();
		this.testFile = this.builder.finish().getFileName();
	}

	@After
	public void after() {
		new File(this.testFile).delete();
	}

	@Test
	public void loadTest() {
		assertEquals(FNNCBuilder.toJSON(this.builder),
				FNNCBuilder.toJSON(FNNCBuilder.load(this.testFile).getBuilder()));
	}

	@Test
	public void remakeTest() {
		assertEquals(FNNCBuilder.toJSON(this.builder), FNNCBuilder.toJSON(getTestBuilder()));
	}

	@Test
	public void similairNetworkTest() {
		FNNCBuilder builder2 = getTestBuilder().backprop(false).finish();
		String fileName = builder2.getFileName();
		assertFalse(this.testFile.equals(fileName));
		FNNCBuilder builder1 = this.builder.finish();
		assertTrue(builder1.getPathName().replaceAll("\\\\", "/")
				.contains(builder2.getPathName().substring(0, builder2.getPathName().length() - 1).replaceAll("\\\\", "/")));

		new File(fileName).delete();
	}

	@Test
	public void multipleBuildTest() {
		FNNCBuilder fb = this.builder.finish();
		fb.setSeed(1);
		Builder build = fb.build();
		assertTrue(build.isBackprop());
		assertFalse(build.isPretrain());
	}

	protected static NNCBuilder getTestBuilder() {
		NNCBuilder builder = CNN.buildExampleCNN(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(2, 2).build(),
				new OutputLayer.Builder().nOut(2).build());
		builder.backprop(true);
		builder.pretrain(false);
		return builder;
	}

}
