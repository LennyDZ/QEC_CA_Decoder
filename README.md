# Cellular Automaton for Surface Code Decoding

In the pursuit of fault-tolerant quantum computing, error correction appears to be essential. The most popular solution as of today is based on [stabilizer codes](https://en.wikipedia.org/wiki/Stabilizer_code), particularly, the surface code has been mostly considered for it's simple topology and promising performance.

During each correction round, stabilizer measurements are performed in order to detect errors (bit-flips and phase-flips). The results of these measurements is called the syndrome. Based on this syndrome, we can find the correction to apply. This requires some algorithmic computation which is commonly referred to as "decoding". This decoding phase has to be highly efficient as it will be repeated many times, and it shouldn't introduce delay on the quantum hardware (given its limited coherence time).

There is several ways to do it, the most popular is based on a graph problem representation called Minimum Weight Perfect Matching. In this project, we focused on a solution proposed [here](https://arxiv.org/pdf/1406.2338) (by M. Herold et al.). This solution, relies on cellular automata, it may not have optimal theoretical complexity, but it can make better use of GPU technologies in order to scale well. The original paper proposed a solution for Toric Codes (but such topology is hard to implement). Here, we add some small modifications in order to adapt the methode to surface code structures.

In this repo, we provide several Python implementations of our decoder and the framework to test them.
