# GTA San Andres Taxi Mission Optimizer

This project attemps to approach the problem of optimizing the taxi mission in GTA San Andreas.

The mission can be played in many locations which determine the possible fare locations. The player has to complete 50 fares. The player can skip a fare and a new one will be chosen randomly.

This program allows specifying a fare generation model and uses simulated annealing to optimize the model to minimize the average time it takes to complete the mission. Optimization changes the state of possible fares such that some of them are marked as to be rerolled.