﻿#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <queue>
#include <set>

#include "Array2.h"

namespace gtasa_taxi_sim
{
	using Seconds = std::chrono::duration<double>;
	using LocationId = std::uint32_t;
	using FareId = std::uint32_t;

	struct Fare
	{
		Fare(Seconds avgDriveTime, LocationId destination) :
			m_avgDriveTime(avgDriveTime),
			m_destination(destination)
		{
		}

		[[nodiscard]] Seconds avgDriveTime() const
		{
			return m_avgDriveTime;
		}
		
		[[nodiscard]] LocationId destination() const
		{
			return m_destination;
		}

	private:
		Seconds m_avgDriveTime;
		LocationId m_destination;
	};

	struct TaxiMissionSimulationResult
	{
		Seconds totalTime;
	};

	struct TaxiMission
	{
		TaxiMission(LocationId numLocations) :
			m_avgNextFareSearchTime(numLocations, Seconds{0.0}),
			m_possibleFares(numLocations),
			m_numAllowedFares(numLocations, 0)
		{
		}

		void setNextFareSearchTime(LocationId location, Seconds time)
		{
			assert(location < numLocations());

			m_avgNextFareSearchTime[location] = time;
		}

		void addPossibleFare(LocationId from, const Fare& fare)
		{
			assert(from < numLocations());
			assert(fare.destination() < numLocations());

			m_possibleFares[from].emplace_back(fare);
			m_numAllowedFares[from] += 1;
		}

		[[nodiscard]] Seconds avgNextFareSearchTime(LocationId location) const
		{
			assert(location < numLocations());

			return m_avgNextFareSearchTime[location];
		}

		[[nodiscard]] LocationId numLocations() const
		{
			return static_cast<LocationId>(m_avgNextFareSearchTime.size());
		}

		template <typename RngT>
		[[nodiscard]] TaxiMissionSimulationResult simulateFares(LocationId startLocation, int numFares, RngT&& rng)
		{
			TaxiMissionSimulationResult result;

			result.totalTime = Seconds{ 0.0 };

			LocationId currentLocation = startLocation;
			for (int i = 0; i < numFares; ++i)
			{
				result.totalTime += avgNextFareSearchTime(currentLocation);

				const auto& [fare, isAllowed] = chooseRandomFare(currentLocation, rng);

				if (!isAllowed)
				{
					continue;
				}

				result.totalTime += fare.avgDriveTime();
				currentLocation = fare.destination();
			}

			return result;
		}

		template <typename RngT>
		[[nodiscard]] std::pair<const Fare&, bool> chooseRandomFare(LocationId from, RngT&& rng)
		{
			const auto& possibleFares = m_possibleFares[from];

			if (possibleFares.empty())
			{
				throw std::runtime_error("No fare from location " + std::to_string(from));
			}

			const auto fareId = std::uniform_int_distribution<FareId>(
				0, 
				static_cast<FareId>(possibleFares.size()) - 1
				)(rng);

			return { possibleFares[fareId], fareId < m_numAllowedFares[from] };
		}

		template <typename RngT>
		void optimize(int iterations, LocationId startLocation, int numFares, RngT&& rng)
		{
			constexpr double startTemperature = 1.1;
			constexpr double endTemperature = 1;
			constexpr double endTemperatureAtT = 0.5;

			for (int i = 0; i < iterations; ++i)
			{
				const double t = static_cast<double>(i) / (iterations - 1);

				const double temperature =
					t > endTemperatureAtT
					? endTemperature
					: (endTemperatureAtT - t) * startTemperature + t * endTemperature;

				optimizeSingleIteration(temperature, startLocation, numFares, rng);
			}
		}

		void print(LocationId start) const
		{
			for (LocationId from = 0; from < numLocations(); ++from)
			{
				std::cout << "L" << from << ": ";

				const auto& possibleFares = m_possibleFares[from];
				const auto numAllowedFares = m_numAllowedFares[from];

				for (FareId fare = 0; fare < numAllowedFares; ++fare)
				{
					std::cout << possibleFares[fare].destination() << " ";
				}

				std::cout << "-- ";

				for (FareId fare = numAllowedFares; fare < possibleFares.size(); ++fare)
				{
					std::cout << possibleFares[fare].destination() << " ";
				}

				std::cout << '\n';
			}

			std::cout << "\nReachable: ";
			for (auto loc : reachableLocations(start))
			{
				std::cout << loc << ' ';
			}
			std::cout << '\n';
		}

		[[nodiscard]] std::vector<LocationId> reachableLocations(LocationId start) const
		{
			std::set<LocationId> reachable;
			std::queue<LocationId> reachableQueue;
			reachableQueue.emplace(start);
			
			while (!reachableQueue.empty())
			{
				const LocationId current = reachableQueue.front();
				reachableQueue.pop();

				reachable.emplace(current);

				for (FareId i = 0; i < m_numAllowedFares[current]; ++i)
				{
					const auto& fare = m_possibleFares[current][i];
					const auto dest = fare.destination();

					if (reachable.count(dest) == 0)
					{
						reachableQueue.emplace(dest);
					}
				}
			}

			return std::vector<LocationId>(reachable.begin(), reachable.end());
		}

	private:
		std::vector<Seconds> m_avgNextFareSearchTime;
		std::vector<std::vector<Fare>> m_possibleFares;
		std::vector<FareId> m_numAllowedFares;

		template <typename RngT>
		void optimizeSingleIteration(double temperature, LocationId startLocation, int numFares, RngT&& rng)
		{
			constexpr int simulationIters = 10;

			Seconds totalTimeBefore{ 0.0 };
			for (int i = 0; i < simulationIters; ++i)
			{
				totalTimeBefore += simulateFares(startLocation, numFares, rng).totalTime;
			}

			auto [location, fare] = toggleRandomFare(rng);

			Seconds totalTimeAfter{ 0.0 };
			for (int i = 0; i < simulationIters; ++i)
			{
				totalTimeAfter += simulateFares(startLocation, numFares, rng).totalTime;
			}

			if (totalTimeAfter > totalTimeBefore * temperature)
			{
				(void)toggleFare(location, fare);
			}
		}

		// This operation is reversible with the same parameters.
		[[nodiscard]] FareId toggleFare(LocationId from, FareId fareId)
		{
			auto& possibleFares = m_possibleFares[from];

			if (fareId < m_numAllowedFares[from])
			{
				// we have to disallow this fare
				const auto newFareId = m_numAllowedFares[from] - 1;
				std::swap(possibleFares[fareId], possibleFares[newFareId]);
				m_numAllowedFares[from] -= 1;
				return newFareId;
			}
			else
			{
				// we have to allow this fare
				const auto newFareId = m_numAllowedFares[from];
				std::swap(possibleFares[fareId], possibleFares[newFareId]);
				m_numAllowedFares[from] += 1;
				return newFareId;
			}
		}

		template <typename RngT>
		[[nodiscard]] FareId toggleRandomFare(LocationId from, RngT&& rng)
		{
			const auto& possibleFares = m_possibleFares[from];

			if (possibleFares.size() <= 1)
			{
				throw std::runtime_error("Cannot toggle any fare if there is only 1 available.");
			}

			for (;;)
			{
				const auto fareId = std::uniform_int_distribution<FareId>(
					// If there is only one allowed fare then we cannot
					// toggle it. Choose one to enable instead.
					m_numAllowedFares[from] <= 1 ? 1 : 0,
					static_cast<FareId>(possibleFares.size()) - 1
					)(rng);

				return toggleFare(from, fareId);
			}
		}

		template <typename RngT>
		[[nodiscard]] std::pair<LocationId, FareId> toggleRandomFare(RngT&& rng)
		{
			// TODO: make this uniform

			for (;;)
			{
				const auto locationId = std::uniform_int_distribution<LocationId>(
					0,
					numLocations() - 1
					)(rng);

				// We need to find a location where we can even toggle a fare.
				if (m_possibleFares[locationId].size() <= 1)
				{
					continue;
				}

				const auto fareId = toggleRandomFare(locationId, rng);

				return { locationId, fareId };
			}
		}
	};
		
	void testBasic()
	{
		constexpr LocationId numLocations = 3;
		constexpr int numFares = 50;
		constexpr int optimizationIters = 10;
		constexpr int optimizationTries = 10;

		std::mt19937_64 rng(1234);

		auto model = TaxiMission(numLocations);

		for (LocationId i = 0; i < numLocations; ++i)
		{
			model.setNextFareSearchTime(i, Seconds{ 0.2 });

			for (LocationId j = 0; j < numLocations; ++j)
			{
				if (i == j)
				{
					continue;
				}

				model.addPossibleFare(i, Fare(Seconds{ (i + j) * 0.54 }, j));
			}
		}

		{
			LocationId startLocation = 0;

			Seconds total{ 0.0 };
			for (int i = 0; i < 10; ++i)
			{
				total += model.simulateFares(startLocation, numFares, rng).totalTime;
			}
			std::cout << "Before optimization: " << total.count() << "s\n";

			for (int i = 0; i < optimizationTries; ++i)
			{
				model.optimize(optimizationIters, startLocation, numFares, rng);

				total = Seconds{ 0.0 };
				for (int i = 0; i < 10; ++i)
				{
					total += model.simulateFares(startLocation, numFares, rng).totalTime;
				}
				std::cout << "After optimization try " << i << ": " << total.count() << "s\n";
			}

			model.print(startLocation);
		}
	}

	struct Point
	{
		double x, y;
	};

	[[nodiscard]] double distance(const Point& lhs, const Point& rhs)
	{
		const double dx = lhs.x - rhs.x;
		const double dy = lhs.y - rhs.y;
		return std::sqrt(dx * dx + dy * dy);
	}

	template <typename RngT>
	TaxiMission generateRandomModel(
		LocationId numLocations, 
		RngT&& rng)
	{
		constexpr double minX = 0.0;
		constexpr double minY = 0.0;
		constexpr double maxX = 800.0;
		constexpr double maxY = 600.0;
		constexpr double minFareSearchTime = 4.0;
		constexpr double maxFareSearchTime = 10.0;
		constexpr double minDistanceMultiplier = 0.8;
		constexpr double maxDistanceMultiplier = 1.2;
		constexpr double drivingSpeed = 40.0;

		auto randomPoint = [&]()
		{
			std::uniform_real_distribution<double> dx(minX, maxX);
			std::uniform_real_distribution<double> dy(minY, maxY);
			return Point{ dx(rng), dy(rng) };
		};

		auto model = TaxiMission(numLocations);

		std::vector<Point> locationPoints;
		for (LocationId i = 0; i < numLocations; ++i)
		{
			locationPoints.emplace_back(randomPoint());
		}

		for (LocationId from = 0; from < numLocations; ++from)
		{
			model.setNextFareSearchTime(from, Seconds{ std::uniform_real_distribution<double>(minFareSearchTime, maxFareSearchTime)(rng) });

			for (LocationId to = 0; to < numLocations; ++to)
			{
				if (from == to)
				{
					continue;
				}

				const double dm = std::uniform_real_distribution<double>(minDistanceMultiplier, maxDistanceMultiplier)(rng);
				const double rawDistance = distance(locationPoints[from], locationPoints[to]);
				const double distance = rawDistance * dm;
				const Seconds fareTime = Seconds{ distance / drivingSpeed };

				model.addPossibleFare(from, Fare{ fareTime, to });
			}
		}

		return model;
	}

	void testRandomModel()
	{
		constexpr LocationId numLocations = 20;
		constexpr LocationId startLocation = 0;
		constexpr int numFares = 50;
		constexpr int optimizationIters = 100;
		constexpr int optimizationTries = 100;

		std::mt19937_64 rng(1234);

		auto model = generateRandomModel(numLocations, rng);

		model.print(startLocation);

		{
			Seconds total{ 0.0 };
			for (int i = 0; i < 10; ++i)
			{
				total += model.simulateFares(startLocation, numFares, rng).totalTime;
			}
			std::cout << "Before optimization: " << total.count() << "s\n";

			for (int i = 0; i < optimizationTries; ++i)
			{
				model.optimize(optimizationIters, startLocation, numFares, rng);

				total = Seconds{ 0.0 };
				for (int i = 0; i < 10; ++i)
				{
					total += model.simulateFares(startLocation, numFares, rng).totalTime;
				}
				std::cout << "After optimization try " << i << ": " << total.count() << "s\n";
			}

			model.print(startLocation);
		}
	}
}

int main()
{
	gtasa_taxi_sim::testRandomModel();

	return 0;
}
