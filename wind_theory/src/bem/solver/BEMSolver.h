/**
 *  Author: Rohan Ramasamy
 *  Date: 27/12/16
 *
 *  This file contains a class for solving the blade element momentum equations to approximate
 * the performance of a wind turbine.
 */

 namespace bem {

 	class BEMSolver {
 	public:
 		/**
 		 * Construct BEM Solver. Rated input parameters define the optimum point. The 
 		 * AirfoilInterpolator defines the blade lift and drag characteristics. numPts
 		 * defines the number of segments the blade is split into. 
 		 */
 		BEMSolver(
 			double ratedPower,
 			double ratedAngularRotation,
 			double ratedWindSpeed,
 			AirfoilInterpolator airfoil,
 			int numPts
 			);

 		/**
 		 * Check to see if rated condition has been found. The radius is calculated, and 
 		 * the chord and twist have been set to the values of the ideal rotor with wake 
 		 * rotation.
 		 */
 		bool
 		isInitialised();

 		/**
 		 * Return the rated condition as a pair: first is the TSR and second is the 
 		 * power coefficient
 		 */
 		std::pair<double, double>
 		ratedCondition();

 		/**
 		 * Generate the power curve against tip speed ratio for a given twist.
 		 */
 		std::vector<std::vector<double> >
 		calculatePowerCurve(
 			double pitchTwist
 			);

 	private:
 		/**
 		 * Set to be initialised once rated condition is found.
 		 */ 
 		void
 		initialise();

 		/**
 		 * Use rated conditions to get TSR and rated power coefficient.
 		 */
 		std::pair<double, double>
 		findRatedCondition();

 		double mPRated, mWRated, mURated, mRadius;
 		std::pair<double, double> mRatedCondition;
 		AirfoilInterpolator mAirfoil;
 		std::vector<double> mChord, mInherantTwist;
 	};

 }