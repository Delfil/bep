package nl.tudelft.bep.deeplearning.network.data;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertTrue;

import java.io.File;
import java.io.IOException;

import org.junit.Test;

import nl.tudelft.bep.deeplearning.network.data.exception.MetaDataMatchException;
import nl.tudelft.bep.deeplearning.network.data.exception.UnknownMetaDataFileVersion;

public class DataTest {
	@Test
	public void correctReadTest() {
		String testFile = "test_data/correct_1";
		GeneExpressionDatabase data = LoadedGeneExpressionDatabase.Loader.load(testFile);
		assertEquals(1, data.getBatchSize());
		assertEquals(3, data.getExamples());
		assertEquals(3, data.getHeight());
		assertEquals(2, data.getWidth());
		assertEquals(129384957, data.getTimeStamp());
		assertEquals(LoadedGeneExpressionDatabase.Loader.DATA_FOLDER + "/" + testFile, data.getPath());
		assertEquals(1, data.getVersion());
		assertEquals(0.5, data.getTrainPercentage(), 0);
		assertEquals(2, data.getNumOutcomes());
	}

	@Test
	public void subSetTest() {
		String testFile = "test_data/correct_1";
		GeneExpressionDatabase data = LoadedGeneExpressionDatabase.Loader.load(testFile);
		assertEquals(3, data.getSubset(0.0, 1.0).size());
		assertEquals(1, data.getSubset(0.1, 0.9).size());
		assertEquals(2, data.getSubset(0.9, 1.0).size());
		assertEquals(0, data.getSubset(1.0, 0.1).size());
	}

	@Test
	public void nonexistingReadTest() {
		String testFile = "test_data/nonexisting_data";
		assertFalse(new File(testFile).exists());
		assertNull(LoadedGeneExpressionDatabase.Loader.findFile(testFile, ""));
	}

	@Test
	public void missingReadTest() {
		String testFile = "test_data/correct_1";
		assertTrue(new File(LoadedGeneExpressionDatabase.Loader.DATA_FOLDER + "/" + testFile).exists());
		assertNull(LoadedGeneExpressionDatabase.Loader.findFile(testFile, ".missing"));
	}

	@Test(expected = UnknownMetaDataFileVersion.class)
	public void badMetaReadTest() throws NumberFormatException, UnknownMetaDataFileVersion, IOException {
		String testFile = "test_data/bad_meta";
		LoadedGeneExpressionDatabase.Loader.readMetaFile(testFile);
	}

	@Test(expected = MetaDataMatchException.class)
	public void badDatReadTest()
			throws IOException, MetaDataMatchException, NumberFormatException, UnknownMetaDataFileVersion {
		String testFile = "test_data/bad_dat_1";
		assertTrue(new File(LoadedGeneExpressionDatabase.Loader.DATA_FOLDER + "/" + testFile).exists());
		LoadedGeneExpressionDatabase data = LoadedGeneExpressionDatabase.Loader.readMetaFile(testFile);
		LoadedGeneExpressionDatabase.Loader.readMatrices(data);
	}

	@Test(expected = NumberFormatException.class)
	public void badFormatTest() throws NumberFormatException, UnknownMetaDataFileVersion, IOException {
		String testFile = "test_data/bad_dat_2";
		LoadedGeneExpressionDatabase.Loader.readMetaFile(testFile);
	}
}
