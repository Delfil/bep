package nl.tudelft.bep.deeplearning.clustering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertArrayEquals;

import org.junit.Test;

import nl.tudelft.bep.deeplearning.clustering.NeighborDistance;

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
		int[] temp = new int[3];
		temp[0] = 89;
		temp[1] = 2;
		temp[2] = 45;
		
		Double[] tempDistance = new Double[3];
		tempDistance[0] = 4.5;
		tempDistance[1] = 5.5;
		tempDistance[2] = 1.4;
		
		NeighborDistance result = new NeighborDistance(temp, tempDistance);
		
		assertEquals(45, result.getPointIDWithClosestNeighbor(1));
		assertEquals(89, result.getPointIDWithClosestNeighbor(2));
		assertEquals(2, result.getPointIDWithClosestNeighbor(3));
		
	}	
}
