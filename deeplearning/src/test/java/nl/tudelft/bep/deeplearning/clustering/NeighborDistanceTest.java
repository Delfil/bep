package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

public class NeighborDistanceTest {

	@Test
	public void testGetters() {
		int[] temp = new int[2];
		temp[0] = 1;
		temp[1] = 2;
		
		Double[] tempDistance = new Double[2];
		tempDistance[0] = 4.5;
		tempDistance[1] = 5.5;
		
		NeighborDistance result = new NeighborDistance(temp, tempDistance);
		
		assertArrayEquals(temp, result.getResults());
		assertArrayEquals(tempDistance, result.getDistances());
	}
	
	@Test
	public void testGetClosestDistance() {
		int[] temp = new int[2];
		temp[0] = 89;
		temp[1] = 2;
		
		Double[] tempDistance = new Double[2];
		tempDistance[0] = 4.5;
		tempDistance[1] = 5.5;
		
		NeighborDistance result = new NeighborDistance(temp, tempDistance);
		
		assertEquals(89, result.getPointIDWithClosestNeighbor());
	}	
}
