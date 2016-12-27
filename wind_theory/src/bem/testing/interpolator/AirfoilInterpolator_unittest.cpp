/*
Author: Rohan Ramasamy
Data: 31/03/16

This file contains unit tests for AirfoilInterpolator
*/

// Linked Includes
#include "gtest/gtest.h"

#include <vector>
#include <iostream>

#include "../../interpolator/AirfoilInterpolator.h"


namespace bem 
{
	class AirfoilInterpolatorTest : public ::testing::Test {
	public:
		virtual void SetUp() {}

		virtual void TearDown() {}
	};

	TEST_F(AirfoilInterpolatorTest, ConstructorTest) {
		// Mismatched sizes
		std::vector<double> alphas = {0.0, 1.0, 2.0, 3.0};
		std::vector<double> liftCoefficients = {0.0, 1.0};
		std::vector<double> dragCoefficients = {0.0};
		EXPECT_ANY_THROW(AirfoilInterpolator(alphas, dragCoefficients, liftCoefficients));

		// Negative alpha
		alphas = {-1.0, 1.0, 2.0};
		liftCoefficients = {0.0, 1.0, 2.0, 3.0};
		dragCoefficients = {0.0, 1.0, 2.0, 3.0};
		EXPECT_ANY_THROW(AirfoilInterpolator(alphas, dragCoefficients, liftCoefficients));

		// Non-monotonic alpha
		alphas = {1.0, 0.5, 2.0};
		liftCoefficients = {0.0, 1.0, 2.0, 3.0};
		dragCoefficients = {0.0, 1.0, 2.0, 3.0};
		EXPECT_ANY_THROW(AirfoilInterpolator(alphas, dragCoefficients, liftCoefficients));

		// Data set too small
		alphas = {1.0, 0.5};
		liftCoefficients = {0.0, 1.0};
		dragCoefficients = {0.0, 1.0};
		EXPECT_ANY_THROW(AirfoilInterpolator(alphas, dragCoefficients, liftCoefficients));
	}

	TEST_F(AirfoilInterpolatorTest, InterpolateOnDataPoints) {
		std::vector<double> alphas = {0.0, 1.0, 2.0, 3.0};
		std::vector<double> liftCoefficients = {0.0, 1.5, 2.5, 3.0};
		std::vector<double> dragCoefficients = {0.0, 2.3, 4.6, 3.0};
		AirfoilInterpolator interpolator(alphas, dragCoefficients, liftCoefficients);

		for (size_t i = 0; i < alphas.size(); ++i) {
			EXPECT_EQ(liftCoefficients[i], interpolator.getLiftCoefficient(alphas[i]));
			EXPECT_EQ(dragCoefficients[i], interpolator.getDragCoefficient(alphas[i]));
		}
	}

	TEST_F(AirfoilInterpolatorTest, InterpolateOutsideDataLimits) {
		std::vector<double> alphas = {0.0, 1.0, 2.0, 3.0};
		std::vector<double> liftCoefficients = {0.0, 1.5, 2.5, 3.0};
		std::vector<double> dragCoefficients = {0.0, 2.3, 4.6, 3.0};
		AirfoilInterpolator interpolator(alphas, dragCoefficients, liftCoefficients);

		EXPECT_ANY_THROW(interpolator.getLiftCoefficient(3.6));
		EXPECT_ANY_THROW(interpolator.getDragCoefficient(3.6));
		EXPECT_ANY_THROW(interpolator.getLiftCoefficient(-0.1));
		EXPECT_ANY_THROW(interpolator.getDragCoefficient(-0.1));
	}

	TEST_F(AirfoilInterpolatorTest, OptimumAngle) {
		std::vector<double> alphas = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
		std::vector<double> liftCoefficients = {0.0, 1.5, 2.5, 3.1, 6.6, 3.3};
		std::vector<double> dragCoefficients = {0.0, 2.3, 4.6, 1.5, 2.2, 4.7};
		AirfoilInterpolator interpolator(alphas, dragCoefficients, liftCoefficients);

		EXPECT_EQ(4.0, interpolator.getIdealAngleOfAttack());
	}

}
