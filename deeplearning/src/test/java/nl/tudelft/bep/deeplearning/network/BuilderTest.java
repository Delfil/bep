package nl.tudelft.bep.deeplearning.network;

import static org.junit.Assert.*;

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
		builder = getTestBuilder();
		testFile = builder.finish().getFileName();

	}

	@After
	public void after() {
		new File(testFile).delete();
	}

	@Test
	public void loadTest() {
		assertEquals(FNNCBuilder.toJSON(builder), FNNCBuilder.toJSON(FNNCBuilder.load(testFile).builder));
	}

	@Test
	public void remakeTest() {
		assertEquals(FNNCBuilder.toJSON(builder), FNNCBuilder.toJSON(getTestBuilder()));
	}

	@Test
	public void similairNetworkTest() {
		FNNCBuilder builder2 = getTestBuilder().backprop(false).finish();
		String fileName = builder2.getFileName();
		assertTrue(testFile.contains("0"));
		assertTrue(fileName.contains("1"));
		FNNCBuilder builder1 = builder.finish();
		System.out.println(builder1.pathName);
		System.out.println(builder2.pathName);
		assertTrue(builder1.getPathName()
				.contains(builder2.getPathName().substring(0, builder2.getPathName().length() - 1)));
		new File(fileName).delete();
	}
	
	@Test
	public void multipleBuildTest() {
		FNNCBuilder fb = builder.finish();
		fb.setSeed(1);
		Builder build = fb.build();
		assertTrue(build.isBackprop());
		assertFalse(build.isPretrain());
	}
	
	protected NNCBuilder getTestBuilder() {
		NNCBuilder builder = CNN.BuildExampleCNN(new ConvolutionLayer.Builder().kernelSize(1, 1).stride(2, 2).build(),
				new OutputLayer.Builder().nOut(2).build());
		builder.backprop(true);
		builder.pretrain(false);
		return builder;
	}

	
}
