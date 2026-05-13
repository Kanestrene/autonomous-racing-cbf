# Autonomous Racing using CBFs and CLFs

Real-time autonomous racing controller using Control Barrier Functions (CBFs), Control Lyapunov Functions (CLFs), and Quadratic Programming for safe path tracking and obstacle avoidance.

![demo](media/demo.gif)

---

## Overview

This project explores safety-critical control for autonomous racing vehicles by combining:

- Control Barrier Functions (CBFs) for collision avoidance
- Control Lyapunov Functions (CLFs) for trajectory tracking
- Quadratic Programming (QP) for real-time control optimization

The controller generates safe steering and velocity commands while maintaining stable path tracking and avoiding dynamic obstacles.

---

## Features

- Real-time obstacle avoidance
- CLF-based trajectory tracking
- CBF safety constraints
- Quadratic Programming controller
- Lookahead-based vehicle model
- Adjustable safety aggressiveness parameters
- Autonomous racing simulation
- Multi-agent racing scenarios
- Reinforcement learning integration
- Real-world vehicle integration
- Dynamic obstacle prediction

---

## System Architecture

```text
Reference Path
       ↓
Perception
       ↓
CBF + CLF QP Controller
       ↓
Vehicle Dynamics
       ↓
Control Commands
```

---

## Control Formulation

The controller is formulated as a Quadratic Program of the form:

$$
\min_u \frac{1}{2}(u - u_{nom})^T W (u - u_{nom})
$$

subject to:

$$
\dot{h}(x,u) + \alpha h(x) \ge 0
$$

where:
- $h(x)$ represents the safety barrier function
- $\alpha$ controls obstacle avoidance aggressiveness
- CLF constraints enforce trajectory convergence
---

## Results

### Path Tracking and Obstacle Avoidance

![tracking](media/lookhaed0.5lyp.pdf)

---

## Tech Stack

### Languages & Frameworks

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

### Tools

- Quadratic Programming
- Matplotlib
- Shapely

---

## Repository Structure

```text
Fig/                → Figures, plots, and paper assets
Pack1-Pack7/        → Simulation and controller development modules
Real1-Real3/        → Real-world vehicle experiments
SimuPWM/            → PWM and low-level control simulations
XIAO_ESP32C3_BT/    → ESP32 communication and embedded firmware
car/                → Vehicle modeling and racing control modules
```
---

## Installation

Clone the repository:

```bash
git clone https://github.com/Kanestrene/ControllerRC.git
cd ControllerRC
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the simulation:

```bash
python simulation1.py
```

---

## Future Work

- MPC-CBF hybrid control


---

## Author

Pedro Lopes

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Pedro%20Lopes-0077B5?style=flat&logo=linkedin&logoColor=white)](https://linkedin.com/in/pedro-lopes-3717b5210)

[![Email](https://img.shields.io/badge/Email-pedruxande%40gmail.com-D14836?style=flat&logo=gmail&logoColor=white)](mailto:pedruxande@gmail.com)
