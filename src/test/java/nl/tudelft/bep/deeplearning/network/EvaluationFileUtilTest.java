package nl.tudelft.bep.deeplearning.network;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.util.ArrayList;
import java.util.List;

import org.apache.commons.io.FileUtils;
import org.deeplearning4j.eval.Evaluation;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import nl.tudelft.bep.deeplearning.network.builder.CNN;
import nl.tudelft.bep.deeplearning.network.builder.FNNCBuilder;
import nl.tudelft.bep.deeplearning.network.data.Data;
import nl.tudelft.bep.deeplearning.network.result.EvaluationFileUtil;

public class EvaluationFileUtilTest {

	private static final int EPOCH = 0;
	private static final long SEED = 0;
	private List<String> toRemove = new ArrayList<>();
	private FNNCBuilder builder;
	private Data data;

	@Before
	public void before() {
		this.data = Data.readDataSet("test_data/correct_1");
		this.builder = CNN.buildExampleCNN().finish();
		this.toRemove.add(this.builder.getPathName());
		this.toRemove.add(this.builder.getFileName());
		this.toRemove.add(EvaluationFileUtil.getEvalPathName(this.data, this.builder));
	}

	@Test
	public void evalExists() {
		assertFalse(EvaluationFileUtil.evalExistst(SEED, EPOCH, this.data, this.builder));
		EvaluationFileUtil.save(new Evaluation<>(), SEED, EPOCH, this.data, this.builder);
		assertTrue(EvaluationFileUtil.evalExistst(SEED, EPOCH, this.data, this.builder));
	}

	@Test
	public void loadTest() {
		Evaluation<Double> ev = new Evaluation<>(2);
		ev.eval(0, 1);
		ev.eval(1, 1);
		ev.eval(1, 1);
		ev.eval(1, 1);
		ev.eval(0, 0);
		EvaluationFileUtil.save(ev, SEED, EPOCH, this.data, this.builder);
		assertEquals(EvaluationFileUtil.load(SEED, EPOCH, this.data, this.builder).f1(), ev.f1(), 0.0);
		EvaluationFileUtil.save(ev, SEED, EPOCH + 1, this.data, this.builder);
		assertEquals(1, EvaluationFileUtil.load(EPOCH, this.data, this.builder).size());
		EvaluationFileUtil.save(ev, SEED + 1, EPOCH, this.data, this.builder);
		assertEquals(2, EvaluationFileUtil.load(EPOCH, this.data, this.builder).size());

	}

	@After
	public void after() {
		this.toRemove.forEach(name -> {
			File file = new File(name);
			if (!file.exists()) {
				return;
			}
			if (file.isFile()) {
				file.delete();
			} else {
				try {
					FileUtils.cleanDirectory(file);
				} catch (Exception e) {
					e.printStackTrace();
				}
				file.delete();
			}
		});
	}

}
