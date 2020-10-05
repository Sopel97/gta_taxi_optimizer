# GTA San Andres Taxi Mission Optimizer

This project attemps to approach the problem of optimizing the taxi mission in GTA San Andreas.

The mission can be played in many locations which determine the possible fare locations. The player has to complete 50 fares. The player can skip a fare and a new one will be chosen randomly.

This program allows specifying a fare generation model and uses simulated annealing to optimize the model to minimize the average time it takes to complete the mission. Optimization changes the state of possible fares such that some of them are marked as to be rerolled.

## Building

This project uses cmake so you can generate the files needed for your build system of choice.

For example if you're using MSYS:
```
cmake -G "MSYS Makefiles" CMakeLists.txt
```

Alternative one may download the current windows binary from the releases.

## Usage

This is a command line application. Currently it has one way of being invoked.
```
gtasa_taxi_sim.exe config_path input_model_path output_model_path
```

For example
```
gtasa_taxi_sim.exe examples/optimization.cfg examples/random.model examples/out.model
```
will load the optimization parameters from `examples/optimization.cfg`, the initial model from `examples/random.model`, then will perform the optimization, and at the end output the resulting model (with some fares disabled) to `examples/out.model`. The resulting model can be loaded again and improved upon.

## Input file specification

### .model

This file specifies the model which corresponds to the taxi mission. It encodes constraints for the generated fares and empirical data for these fares such as for example average time to complete each fare.

The file consists of tokens separated by whitespaces. An entry is a fixed length chain of tokens, the first being the name of the entry type and the rest its parameters.

Currently there are 2 entry types: `loc` and `fare`.

```
loc name avg_time_to_find_a_new_fare_seconds
```

For example:
```
loc Ganton 4.5
loc Docks 7.5
```

defines two locations with different times required to find a fare.

Note: the name must not contain whitespaces (as otherwise it will be recognized as multiple tokens)

```
fare from_location_name to_location_name avg_driving_time_seconds [enabled/disabled]
```

`enabled`/`disabled` controls whether the fare is initially set to be used or rerolled respectively.

For example:
```
fare Ganton Docks 35.0 disabled
```

means that the game can generate a fare from Ganton to Docks which takes on average 35 seconds to complete and we initially set it to be rerolled.

### .cfg

This file can represent the parameters used for the optimization. It's structure is similar to .model files. Currently the following options are available.

`seed` - An integer in range 0..2^64-1 to use as a seed to the pseudo random number generator. Different seeds will yield different optimization paths. If ommited then current time will be used.

`start_temperature` - the optimizer uses simulated annealing. Starting temperature of 1.3 means that it will allow solutions at most 1.3 tomes worse than the previous one at the start.

`end_temperature` - End temperature for simulated annealing. Generally should be 1.0 but could be lower. Shouldn't be higher than 1.0.

`end_temperature_after` - The % of batch completed to fix the temperature to the end temperature. For example a value of 0.5 would mean that endTemperature is applied from the half of the batch all the way to the end.

`num_fares_to_complete` - Number of fares to complete in a single simulation.

`num_batches` - Number of startTemperature -> endTemperature cycles to finish.

`num_temperature_stages` - Number of optimization steps (model permutations) within a single batch.

`num_averaged_simulations` - Number of simulations to use for model quality estimate.

`num_threads` - Number of threads to use during optimization

`min_toggled_fares` - The minimal number of fares to toggle in a single permutation stage.

`max_toggled_fares` - The maximal number of fares to toggle in a single permutation stage.

`optimization_target` - What result should be optimizied. [avg/min/max]

The default settings are

```
start_temperature 1.3
end_temperature 1.0
end_temperature_after 0.67
num_fares_to_complete 50
num_batches 10
num_temperature_stages 100
num_averaged_simulations 100
num_threads 1
min_toggled_fares 1
max_toggled_fares 1
optimization_target avg
```