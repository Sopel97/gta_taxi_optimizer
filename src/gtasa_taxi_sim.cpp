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
#include <optional>
#include <future>

#include "Array2.h"

namespace gtasa_taxi_sim
{
	namespace fs = std::filesystem;

	using Seconds = std::chrono::duration<double>;
	using LocationId = std::uint32_t;
	using FareId = std::uint32_t;

	// Fare represents a fare that a game can generate.
	// It is not attached to an origin.
	// Whether a fare can be generated from a given origin
	// can be specified in the model.
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
		Seconds totalTime{ 0.0 };
		Seconds averageTime{ 0.0 };
		Seconds minTime{ std::numeric_limits<double>::max() };
		Seconds maxTime{ 0.0 };
		std::uint64_t numSimulations{ 0 };

		void operator+=(const SimulationResult& rhs)
		{
			totalTime += rhs.totalTime;
			averageTime = 
				(averageTime * numSimulations + rhs.averageTime * rhs.numSimulations)
				/ (numSimulations + rhs.numSimulations);
			minTime = std::min(minTime, rhs.minTime);
			maxTime = std::max(maxTime, rhs.maxTime);
			numSimulations += rhs.numSimulations;
		}
	};

	struct OptimizationParameters
	{
	private:
		// Seed to use for the PRNG.
		std::optional<std::uint64_t> m_seed = std::nullopt;

	public:
		[[nodiscard]] std::uint64_t getSeed() const
		{
			if (m_seed.has_value())
			{
				return *m_seed;
			}
			else
			{
				return std::chrono::high_resolution_clock::now().time_since_epoch().count();
			}
		}

		[[nodiscard]] std::uint64_t safeNumThreads() const
		{
			return std::clamp<std::uint64_t>(numThreads, 1, std::thread::hardware_concurrency());
		}

		// Optimizer uses simulated annealing.
		// Starting temperature of 1.3 means that
		// it will allow solutions at most 1.3 times worse
		// than the previous one at the start
		double startTemperature = 1.3;

		// End temperature for simulated annealing.
		// Generally should be 1.0 but could be lower.
		// Shouldn't be higher than 1.0.
		double endTemperature = 1.0;

		// The % of batch completed to fix the
		// temperature to the end temperature.
		// For example a value of 0.5 would mean that
		// endTemperature is applied from the half of the batch
		// all the way to the end.
		double endTemperatureAfter = 0.67;

		// Number of fares to complete in a single simulation.
		std::uint64_t numFaresToComplete = 50;

		// Number of startTemperature -> endTemperature
		// cycles to finish.
		std::uint64_t numBatches = 10;

		// Number of optimization steps (model permutations) 
		// within a single batch.
		std::uint64_t numTemperatureStages = 100;

		// Number of simulations to use for model quality estimate.
		std::uint64_t numAveragedSimulations = 100;

		std::uint64_t numThreads = 1;

		// The minimal number of fares to toggle in a single permutation stage.
		std::uint64_t minToggledFares = 1;

		// The maximal number of fares to toggle in a single permutation stage.
		std::uint64_t maxToggledFares = 1;

		[[nodiscard]] static OptimizationParameters fromStream(std::istream& in)
		{
			using namespace std::literals;

			OptimizationParameters params{};

			for (std::string token; in >> token;)
			{
				if (token == "seed"sv)
				{
					std::uint64_t seed;
					in >> seed;
					params.m_seed = seed;
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
				else if (token == "num_threads"sv)
				{
					in >> params.numThreads;
				}
				else if (token == "min_toggled_fares"sv)
				{
					in >> params.minToggledFares;
				}
				else if (token == "max_toggled_fares"sv)
				{
					in >> params.maxToggledFares;
				}
				else
				{
					throw std::runtime_error("Invalid parameter: " + token);
				}
			}

			return params;
		}
	};

	// This class represents the possible fares and also
	// which fares are rerolled (disabled).
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

		// Number of locations is fixed from the start
		// because otherwise we wouldn't know
		// how much memory to allocate for the state.
		Model(LocationId numLocations) :
			m_avgNextFareSearchTime(numLocations, Seconds{0.0}),
			m_faresFromLocation(numLocations),
			m_numEnabledFares(numLocations, 0)
		{
		}

		// Set the amount of time it takes on average
		// to find a new fare at the given location.
		void setNextFareSearchTime(LocationId location, Seconds time)
		{
			assert(location < numLocations());

			m_avgNextFareSearchTime[location] = time;
		}

		void addFare(LocationId from, const Fare& fare, bool enabled = true)
		{
			assert(from < numLocations());
			assert(fare.destination() < numLocations());

			m_faresFromLocation[from].emplace_back(fare);

			if (enabled)
			{
				m_numEnabledFares[from] += 1;
			}
			else
			{
				(void)toggleFare(from, static_cast<FareId>(m_faresFromLocation[from].size() - 1));
			}

			m_isFareLocationDistributionUpToDate = false;
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
			SimulationResult result{};

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
			result.minTime = result.totalTime;
			result.maxTime = result.totalTime;
			result.numSimulations = 1;

			return result;
		}

		template <typename RngT>
		[[nodiscard]] SimulationResult simulateFares(std::uint64_t numFares, std::uint64_t numSimulations, RngT&& rng)
		{
			SimulationResult result{};

			for (std::uint64_t i = 0; i < numSimulations; ++i)
			{
				result += simulateFares(numFares, rng);
			}

			return result;
		}

		template <typename RngT>
		[[nodiscard]] std::pair<const Fare&, bool> chooseRandomFare(LocationId from, RngT&& rng)
		{
			const auto& fares = m_faresFromLocation[from];

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

		// Tries to find a set of fares that minimizes the average
		// simulation time.
		// Uses simple simulation annealing.
		template <typename RngT = std::mt19937_64>
		void optimize(const OptimizationParameters& params, std::ostream& report)
		{
			const std::uint64_t numThreads = params.safeNumThreads();

			auto nextSeed = [rng = RngT(params.getSeed())] () mutable {
				return rng() * 6364136223846793005ull;
			};

			RngT rng(nextSeed());

			std::vector<RngT> threadRngs;
			for (std::uint64_t i = 0; i < numThreads - 1; ++i)
			{
				threadRngs.emplace_back(nextSeed());
			}

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

					auto newResult = optimizeSingleBatch<RngT>(
						params, 
						prevResult, 
						temperature, 
						rng, 
						threadRngs);

					if (newResult.averageTime < bestResult.averageTime)
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

				const auto& possibleFares = m_faresFromLocation[from];
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
					const auto& fare = m_faresFromLocation[current][i];
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
					const auto& fare = m_faresFromLocation[from][fareId];
					out << "fare " << from << " " << fare.destination() << " " << fare.avgDriveTime().count() << " enabled\n";
				}

				out << "\n";

				for (; fareId < m_faresFromLocation[from].size(); ++fareId)
				{
					const auto& fare = m_faresFromLocation[from][fareId];
					out << "fare " << from << " " << fare.destination() << " " << fare.avgDriveTime().count() << " disabled\n";
				}

				out << "\n\n";
			}
		}

	private:
		// For each location we store how much time on average
		// is needed to find a new fare.
		std::vector<Seconds> m_avgNextFareSearchTime;

		// For each location we store the fares that can be generated.
		std::vector<std::vector<Fare>> m_faresFromLocation;

		// First m_numEnabledFares[loc] fares in m_faresFromLocation[loc] are
		// enabled, the rest is disabled (will be rerolled).
		std::vector<FareId> m_numEnabledFares;

		// Used by the optimizer to choose a random fare with
		// equal probability for each.
		std::discrete_distribution<LocationId> m_fareLocationDistribution;
		bool m_isFareLocationDistributionUpToDate = false;

		template <typename RngT>
		[[nodiscard]] SimulationResult optimizeSingleBatch(
			const OptimizationParameters& params, 
			const SimulationResult& prevResult, 
			double temperature, 
			RngT& rng,
			std::vector<RngT>& threadRngs)
		{

			const std::uint64_t numFaresToToggle =
				std::uniform_int_distribution<std::uint64_t>(
					params.minToggledFares,
					params.maxToggledFares
					)(rng);

			std::vector<std::pair<LocationId, FareId>> toggledFares;
			for (std::uint64_t i = 0; i < numFaresToToggle; ++i)
			{
				toggledFares.emplace_back(toggleRandomFare(rng));
			}

			updateFareLocationDistribution();

			const std::uint64_t numThreads = params.safeNumThreads();
			const std::uint64_t jobSize = params.numAveragedSimulations / numThreads + 1;

			SimulationResult newResult;
			std::vector<std::future<SimulationResult>> threadResults;
			for (std::uint64_t i = 0; i < numThreads; ++i)
			{
				const bool isLast = i == numThreads - 1;
				auto& hereRng = isLast ? rng : threadRngs[i];

				auto job = [this, i, jobSize, &hereRng, &params](){
					return simulateFares(params.numFaresToComplete, jobSize, hereRng);
				};

				if (isLast)
				{
					// Run one job on the main thread.
					// It should be last so we don't block others.
					// There's also only numThreads - 1 rngs
					newResult = job();
				}
				else
				{
					threadResults.emplace_back(std::async(std::launch::async, job));
				}
			}

			for (auto& future : threadResults)
			{
				newResult += future.get();
			}

			if (newResult.averageTime > prevResult.averageTime * temperature)
			{
				// Fares have to be toggled back in the reverse order.
				for (std::uint64_t i = numFaresToToggle - 1; i < numFaresToToggle; --i)
				{
					auto& [location, fare] = toggledFares[i];
					(void)toggleFare(location, fare);
				}
			}

			return newResult;
		}

		// This operation is reversible with the same parameters.
		[[nodiscard]] FareId toggleFare(LocationId from, FareId fareId)
		{
			auto& fares = m_faresFromLocation[from];

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
			const auto& fares = m_faresFromLocation[from];

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

		void updateFareLocationDistribution()
		{
			if (m_isFareLocationDistributionUpToDate)
			{
				return;
			}

			std::vector<FareId> counts;
			counts.reserve(m_avgNextFareSearchTime.size());
			for (const auto& fares : m_faresFromLocation)
			{
				// We need to find a location where we can even toggle a fare.
				if (fares.size() <= 1)
				{
					continue;
				}

				counts.emplace_back(static_cast<FareId>(fares.size()));
			}

			m_fareLocationDistribution = 
				std::discrete_distribution<LocationId>(counts.begin(), counts.end());

			m_isFareLocationDistributionUpToDate = true;
		}

		template <typename RngT>
		[[nodiscard]] std::pair<LocationId, FareId> toggleRandomFare(RngT&& rng)
		{
			for (;;)
			{
				const auto locationId = m_fareLocationDistribution(rng);

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

	void process(const std::vector<std::string>& args)
	{
		if (args.size() < 3)
		{
			throw std::runtime_error("Invalid arguments to process.");
		}

		const fs::path configPath = args[0];
		const fs::path inputModelPath = args[1];
		const fs::path outputModelPath = args[2];

		const auto config = loadOptimizationParameters(configPath);
		auto model = loadModelFromFile(inputModelPath);

		model.optimize(config, std::cout);

		std::ofstream outfile(outputModelPath);
		model.saveToStream(outfile);
	}
}

void help()
{
	std::cout << "Usage: gtasa_taxi_sim.exe config_path input_model_path output_model_path\n";
	std::cout << "Example: gtasa_taxi_sim.exe examples/optimization.cfg examples/random.model examples/out.model\n";
}

int main(int argc, char** argv)
{
	gtasa_taxi_sim::process({ "../../../examples/optimization_mt.cfg", "../../../examples/random.model" , "../../../examples/out2.model" });
	return 0;
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
