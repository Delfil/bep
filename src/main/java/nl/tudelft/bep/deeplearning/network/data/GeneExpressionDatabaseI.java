package nl.tudelft.bep.deeplearning.network.data;

import java.util.List;

import org.nd4j.linalg.dataset.DataSet;

public interface GeneExpressionDatabaseI {

	List<DataSet> getSubset(final double start, final double end);

	String getPath();

	int getVersion();

	long getTimeStamp();

	int getExamples();

	int getWidth();

	int getHeight();

	int getNumOutcomes();

	int getBatchSize();

	double getTrainPercentage();

	String getName();
}
