# Reinforcement learning with PPO (Ray) integrated with cuPSS for simulating active nematohydrodynamics.

Building on recent advances enabling optogenetic control of experimental active materials, we present a model-free reinforcement learning (RL) framework that discovers spatiotemporal sequences of activity to drive a 2D active nematic system toward a prescribed dynamical steady-state. Unlike traditional model-based approaches, this framework does not require a detailed physics model, making it particularly suitable for experimental conditions with noise and uncertainties. Active nematics, which are prone to spontaneous defect proliferation and chaotic streaming dynamics in uncontrolled environments, can be guided into a range of alternative dynamical states using this approach. The RL based controller dynamically reconfigures the system, stabilizing emergent behaviors that do not correspond to natural attractors and would be otherwise inaccessible without control such as coherent flow without boundary and uniform nematic aligned along a chosen direction within a prescribed region starting from almost any initial condition. Our results provide a roadmap to leverage control methods to rationally design structure, dynamics, and function in a wide variety of active materials.

## RL Setup and Control Goals
<img src="https://github.com/ghoshsap/deep_rl_cupss/blob/main/images/rl_fig1.001.png" alt="Diagram" width="800" />

## PPO implementation and Network architecture
<img src="https://github.com/ghoshsap/deep_rl_cupss/blob/main/images/ppo_flow_chart_2.001.png" alt="Diagram" width="900" />

### Dependencies 

Install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) <br>
PDE solver installation: [cuPSS](https://github.com/fcaballerop/cuPSS). <br>
Install msgpack for server-client communication: [msgpack](https://github.com/msgpack/msgpack-c/tree/cpp_master).
```bash
 CUDA > 12.0
 Python = 3.10
 PyTorch = 2.4.0
```




#### Submitting a training job on HPCC Cluster

```bash
sbatch compute_new.sh logs
```

