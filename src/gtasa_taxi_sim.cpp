#include <iostream>
#include <string>
#include <chrono>
#include <vector>
#include <cassert>
#include <random>
#include <cmath>
#include <queue>
#include <map>
#include <set>
#include <string_view>
#include <cstdlib>
#include <filesystem>
#include <fstream>

#include "Array2.h"

namespace gtasa_taxi_sim
{
	namespace fs = std::filesystem;

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

	struct SimulationResult
	{
		Seconds totalTime;
		Seconds averageTime;
		std::uint64_t numSimulations;

		void operator+=(const SimulationResult& rhs)
		{
			totalTime += rhs.totalTime;
			averageTime = 
				(averageTime * numSimulations + rhs.averageTime * rhs.numSimulations)
				/ (numSimulations + rhs.numSimulations);
			numSimulations += rhs.numSimulations;
		}
	};

	struct OptimizationParameters
	{
		std::uint64_t seed = 0x0123456789abcdef;
		double startTemperature = 1.3;
		double endTemperature = 1.0;
		double endTemperatureAfter = 0.67;
		std::uint64_t numFaresToComplete = 50;
		std::uint64_t numBatches = 10;
		std::uint64_t numTemperatureStages = 100;
		std::uint64_t numAveragedSimulations = 100;

		[[nodiscard]] static OptimizationParameters fromStream(std::istream& in)
		{
			using namespace std::literals;

			OptimizationParameters params{};

			for (std::string token; in >> token;)
			{
				if (token == "seed"sv)
				{
					in >> params.seed;
				}
				else if (token == "start_temperature"sv)
				{
					in >> params.startTemperature;
				}
				else if (token == "end_temperature"sv)
				{
					in >> params.endTemperature;
				}
				else if (token == "end_temperature_after"sv)
				{
					in >> params.endTemperatureAfter;
				}
				else if (token == "num_fares_to_complete"sv)
				{
					in >> params.numFaresToComplete;
				}
				else if (token == "num_batches"sv)
				{
					in >> params.numBatches;
				}
				else if (token == "num_temperature_stages"sv)
				{
					in >> params.numTemperatureStages;
				}
				else if (token == "num_averaged_simulations"sv)
				{
					in >> params.numAveragedSimulations;
				}
				else
				{
					throw std::runtime_error("Invalid parameter: " + token);
				}
			}

			return params;
		}
	};

	struct Model
	{
		static constexpr LocationId startLocation = 0;

		[[nodiscard]] static Model fromStream(std::istream& in)
		{
			struct FareSpec
			{
				Fare fare;
				LocationId from;
				bool isEnabled;
			};

			using namespace std::literals;

			std::vector<std::string> locationNames;
			std::map<std::string, LocationId> locationByName;
			std::vector<Seconds> avgFareSearchTimes;

			std::vector<FareSpec> fares;

			for (std::string token; in >> token;)
			{
				if (token == "loc"sv)
				{
					std::string name, avgFareSearchTimeStr;
					in >> name >> avgFareSearchTimeStr;

					if (locationByName.count(name))
					{
						throw std::runtime_error("Duplicated location: " + name);
					}

					const Seconds avgFareSearchTime = Seconds{ std::atof(avgFareSearchTimeStr.c_str()) };

					locationNames.emplace_back(name);
					locationByName.try_emplace(name, static_cast<LocationId>(locationNames.size() - 1));
					avgFareSearchTimes.emplace_back(avgFareSearchTime);
				}
				else if (token == "fare"sv)
				{
					std::string enabledStr, avgDriveTimeStr, destinationStr, fromStr;
					in >> fromStr >> destinationStr >> avgDriveTimeStr >> enabledStr;

					auto destinationIt = locationByName.find(destinationStr);
					if (destinationIt == locationByName.end())
					{
						throw std::runtime_error("Invalid location: " + destinationStr);
					}

					const LocationId destination = destinationIt->second;

					auto fromIt = locationByName.find(fromStr);
					if (fromIt == locationByName.end())
					{
						throw std::runtime_error("Invalid location: " + fromStr);
					}

					const LocationId from = fromIt->second;

					const Seconds avgDriveTime = Seconds{ std::atof(avgDriveTimeStr.c_str()) };

					if (enabledStr != "enabled"sv && enabledStr != "disabled"sv)
					{
						throw std::runtime_error("Invalid 'enabled' argument for fare: " + enabledStr);
					}

					const bool isEnabled = destinationStr == "enabled"sv;

					const Fare fare(avgDriveTime, destination);

					fares.emplace_back(FareSpec{ fare, from, isEnabled });
				}
				else
				{
					throw std::runtime_error("Invalid token: " + token);
				}
			}

			auto model = Model(static_cast<LocationId>(locationNames.size()));
			for (const auto& [fare, from, isEnabled] : fares)
			{
				model.addFare(from, fare, isEnabled);
			}

			for (LocationId i = 0; i < locationNames.size(); ++i)
			{
				model.setNextFareSearchTime(i, avgFareSearchTimes[i]);
			}

			return model;
		}

		Model(LocationId numLocations) :
			m_avgNextFareSearchTime(numLocations, Seconds{0.0}),
			m_fares(numLocations),
			m_numEnabledFares(numLocations, 0)
		{
		}

		void setNextFareSearchTime(LocationId location, Seconds time)
		{
			assert(location < numLocations());

			m_avgNextFareSearchTime[location] = time;
		}

		void addFare(LocationId from, const Fare& fare, bool enabled = true)
		{
			assert(from < numLocations());
			assert(fare.destination() < numLocations());

			m_fares[from].emplace_back(fare);

			if (enabled)
			{
				m_numEnabledFares[from] += 1;
			}
			else
			{
				(void)toggleFare(from, static_cast<FareId>(m_fares[from].size() - 1));
			}
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
		[[nodiscard]] SimulationResult simulateFares(std::uint64_t numFares, RngT&& rng)
		{
			SimulationResult result;

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

			result.averageTime = result.totalTime;
			result.numSimulations = 1;

			return result;
		}

		template <typename RngT>
		[[nodiscard]] SimulationResult simulateFares(std::uint64_t numFares, std::uint64_t numSimulations, RngT&& rng)
		{
			SimulationResult result;

			result.totalTime = Seconds{ 0.0 };
			result.averageTime = Seconds{ 0.0 };
			result.numSimulations = 0;

			for (std::uint64_t i = 0; i < numSimulations; ++i)
			{
				result += simulateFares(numFares, rng);
			}

			return result;
		}

		template <typename RngT>
		[[nodiscard]] std::pair<const Fare&, bool> chooseRandomFare(LocationId from, RngT&& rng)
		{
			const auto& fares = m_fares[from];

			if (fares.empty())
			{
				throw std::runtime_error("No fare from location " + std::to_string(from));
			}

			const auto fareId = std::uniform_int_distribution<FareId>(
				0, 
				static_cast<FareId>(fares.size()) - 1
				)(rng);

			return { fares[fareId], fareId < m_numEnabledFares[from] };
		}

		template <typename RngT>
		void optimize(int iterations, std::uint64_t numFares, RngT&& rng)
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
					: (endTemperatureAtT - t) / endTemperatureAtT * startTemperature + t * endTemperature;

				optimizeSingleIteration(temperature, numFares, rng);
			}
		}

		template <typename RngT>
		void optimize(const OptimizationParameters& params, std::ostream& report, RngT&& rng)
		{
			auto prevResult = simulateFares(params.numFaresToComplete, params.numAveragedSimulations, rng);
			auto bestResult = prevResult;
			auto bestState = *this;

			for (std::uint64_t batchId = 0; batchId < params.numBatches; ++batchId)
			{
				for (std::uint64_t i = 0; i < params.numTemperatureStages; ++i)
				{
					const double t = static_cast<double>(i) / (params.numTemperatureStages - 1);

					const double temperature =
						t > params.endTemperatureAfter
						? params.endTemperature
						: (params.endTemperatureAfter - t) / params.endTemperatureAfter * params.startTemperature + t * params.endTemperature;

					auto newResult = optimizeSingleBatch(params, prevResult, temperature, rng);

					if (newResult.totalTime < bestResult.totalTime)
					{
						bestResult = newResult;
						bestState = *this;

						report << "New best: " << newResult.averageTime.count() << "s avg.\n";
					}

					prevResult = newResult;
				}

				*this = bestState;
			}
		}

		void print(std::ostream& = std::cout) const
		{
			for (LocationId from = 0; from < numLocations(); ++from)
			{
				std::cout << "L" << from << ": ";

				const auto& possibleFares = m_fares[from];
				const auto numAllowedFares = m_numEnabledFares[from];

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
			for (auto loc : reachableLocations(startLocation))
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

				for (FareId i = 0; i < m_numEnabledFares[current]; ++i)
				{
					const auto& fare = m_fares[current][i];
					const auto dest = fare.destination();

					if (reachable.count(dest) == 0)
					{
						reachableQueue.emplace(dest);
					}
				}
			}

			return std::vector<LocationId>(reachable.begin(), reachable.end());
		}

		void saveToStream(std::ostream& out) const
		{
			for (LocationId loc = 0; loc < m_avgNextFareSearchTime.size(); ++loc)
			{
				out << "loc " << loc << " " << m_avgNextFareSearchTime[loc].count() << '\n';
			}

			out << "\n\n";

			for (LocationId from = 0; from < m_avgNextFareSearchTime.size(); ++from)
			{
				FareId fareId = 0;
				for (; fareId < m_numEnabledFares[from]; ++fareId)
				{
					const auto& fare = m_fares[from][fareId];
					out << "fare " << from << " " << fare.destination() << " " << fare.avgDriveTime().count() << " enabled\n";
				}

				out << "\n";

				for (; fareId < m_fares[from].size(); ++fareId)
				{
					const auto& fare = m_fares[from][fareId];
					out << "fare " << from << " " << fare.destination() << " " << fare.avgDriveTime().count() << " disabled\n";
				}

				out << "\n\n";
			}
		}

	private:
		std::vector<Seconds> m_avgNextFareSearchTime;
		std::vector<std::vector<Fare>> m_fares;
		std::vector<FareId> m_numEnabledFares;

		template <typename RngT>
		[[nodiscard]] SimulationResult optimizeSingleBatch(const OptimizationParameters& params, const SimulationResult& prevResult, double temperature, RngT&& rng)
		{
			auto [location, fare] = toggleRandomFare(rng);

			auto newResult = simulateFares(params.numFaresToComplete, params.numAveragedSimulations, rng);

			if (newResult.totalTime > prevResult.totalTime * temperature)
			{
				(void)toggleFare(location, fare);
			}

			return newResult;
		}

		template <typename RngT>
		void optimizeSingleIteration(double temperature, std::uint64_t numFares, RngT&& rng)
		{
			constexpr int simulationIters = 10;

			Seconds totalTimeBefore{ 0.0 };
			for (int i = 0; i < simulationIters; ++i)
			{
				totalTimeBefore += simulateFares(numFares, rng).totalTime;
			}

			auto [location, fare] = toggleRandomFare(rng);

			Seconds totalTimeAfter{ 0.0 };
			for (int i = 0; i < simulationIters; ++i)
			{
				totalTimeAfter += simulateFares(numFares, rng).totalTime;
			}

			if (totalTimeAfter > totalTimeBefore * temperature)
			{
				(void)toggleFare(location, fare);
			}
		}

		// This operation is reversible with the same parameters.
		[[nodiscard]] FareId toggleFare(LocationId from, FareId fareId)
		{
			auto& fares = m_fares[from];

			if (fareId < m_numEnabledFares[from])
			{
				// we have to disallow this fare
				const auto newFareId = m_numEnabledFares[from] - 1;
				std::swap(fares[fareId], fares[newFareId]);
				m_numEnabledFares[from] -= 1;
				return newFareId;
			}
			else
			{
				// we have to allow this fare
				const auto newFareId = m_numEnabledFares[from];
				std::swap(fares[fareId], fares[newFareId]);
				m_numEnabledFares[from] += 1;
				return newFareId;
			}
		}

		template <typename RngT>
		[[nodiscard]] FareId toggleRandomFare(LocationId from, RngT&& rng)
		{
			const auto& fares = m_fares[from];

			if (fares.size() <= 1)
			{
				throw std::runtime_error("Cannot toggle any fare if there is only 1 available.");
			}

			for (;;)
			{
				const auto fareId = std::uniform_int_distribution<FareId>(
					// If there is only one allowed fare then we cannot
					// toggle it. Choose one to enable instead.
					m_numEnabledFares[from] <= 1 ? 1 : 0,
					static_cast<FareId>(fares.size()) - 1
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
				if (m_fares[locationId].size() <= 1)
				{
					continue;
				}

				const auto fareId = toggleRandomFare(locationId, rng);

				return { locationId, fareId };
			}
		}
	};

	[[nodiscard]] Model loadModelFromFile(const fs::path& path)
	{
		std::ifstream file(path);
		if (!file.is_open())
		{
			throw std::runtime_error("File not found: " + path.string());
		}
		return Model::fromStream(file);
	}

	[[nodiscard]] OptimizationParameters loadOptimizationParameters(const fs::path& path)
	{
		std::ifstream file(path);
		if (!file.is_open())
		{
			throw std::runtime_error("File not found: " + path.string());
		}
		return OptimizationParameters::fromStream(file);
	}

	void testBasic()
	{
		constexpr LocationId numLocations = 3;
		constexpr int numFares = 50;
		constexpr int optimizationIters = 10;
		constexpr int optimizationTries = 10;

		std::mt19937_64 rng(1234);

		auto model = Model(numLocations);

		for (LocationId i = 0; i < numLocations; ++i)
		{
			model.setNextFareSearchTime(i, Seconds{ 0.2 });

			for (LocationId j = 0; j < numLocations; ++j)
			{
				if (i == j)
				{
					continue;
				}

				model.addFare(i, Fare(Seconds{ (i + j) * 0.54 }, j));
			}
		}

		{
			Seconds total{ 0.0 };
			for (int i = 0; i < 10; ++i)
			{
				total += model.simulateFares(numFares, rng).averageTime;
			}
			std::cout << "Before optimization: " << total.count() << "s\n";

			for (int i = 0; i < optimizationTries; ++i)
			{
				model.optimize(optimizationIters, numFares, rng);

				total = Seconds{ 0.0 };
				for (int i = 0; i < 10; ++i)
				{
					total += model.simulateFares(numFares, rng).totalTime;
				}
				std::cout << "After optimization try " << i << ": " << total.count() << "s\n";
			}

			model.print();
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
	Model generateRandomModel(
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

		auto model = Model(numLocations);

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

				model.addFare(from, Fare{ fareTime, to });
			}
		}

		return model;
	}

	void testRandomModel()
	{
		constexpr LocationId numLocations = 20;
		constexpr int numFares = 50;
		constexpr int optimizationIters = 100;
		constexpr int optimizationTries = 100;

		std::mt19937_64 rng(1234);

		auto model = generateRandomModel(numLocations, rng);

		model.print();

		{
			Seconds total{ 0.0 };
			for (int i = 0; i < 10; ++i)
			{
				total += model.simulateFares(numFares, rng).totalTime;
			}
			std::cout << "Before optimization: " << total.count() << "s\n";

			for (int i = 0; i < optimizationTries; ++i)
			{
				model.optimize(optimizationIters, numFares, rng);

				total = Seconds{ 0.0 };
				for (int i = 0; i < 10; ++i)
				{
					total += model.simulateFares(numFares, rng).totalTime;
				}
				std::cout << "After optimization try " << i << ": " << total.count() << "s\n";
			}

			model.print();
		}
	}

	void testFileModel()
	{
		const fs::path modelFilename = "../../../examples/model.model";
		const fs::path configFilename = "../../../examples/optimization.cfg";

		constexpr int numFares = 50;
		constexpr int optimizationIters = 100;
		constexpr int optimizationTries = 100;

		std::mt19937_64 rng(1234);

		auto model = loadModelFromFile(modelFilename);
		auto config = loadOptimizationParameters(configFilename);

		model.print();

		{
			std::cout << "Before optimization: " << model.simulateFares(numFares, 10, rng).averageTime.count() << "s\n";

			model.optimize(config, std::cout, rng);

			std::cout << "After optimization: " << model.simulateFares(numFares, 10, rng).averageTime.count() << "s\n";

			model.print();
		}
	}

	void process(const std::vector<std::string>& args)
	{
		if (args.size() < 3)
		{
			throw std::runtime_error("Invalid arguments to process.");
		}

		const fs::path configPath = args[0];
		const fs::path inputModelPath = args[1];
		const fs::path outputModelPath = args[2];

		std::mt19937_64 rng(1234);

		const auto config = loadOptimizationParameters(configPath);
		auto model = loadModelFromFile(inputModelPath);

		model.optimize(config, std::cout, rng);

		std::ofstream outfile(outputModelPath);
		model.saveToStream(outfile);
	}
}

void help()
{
	std::cout << "Usage: gtasa_taxi_sim.exe config_path input_model_path output_model_path\n";
}

int main(int argc, char** argv)
{
	if (argc < 4)
	{
		help();
		return 0;
	}

	try
	{
		gtasa_taxi_sim::process(std::vector<std::string>(argv + 1, argv + argc));
	}
	catch (std::runtime_error& e)
	{
		std::cout << e.what();
	}

	return 0;
}
