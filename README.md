# **Gomoku AI Project**

This repository contains an AI agent designed to play the classic board game **Gomoku** on a 15x15 grid. The AI leverages Reinforcement Learning with a Deep Q-Network (DQN) architecture, achieving adaptive decision-making and a high win rate. The project also includes a Minimax algorithm for comparison and the option to play against a human opponent.

---

## **Table of Contents**

1. [Project Overview](#project-overview)  
2. [Features](#features)  
3. [Installation](#installation)  
4. [Usage](#usage)  
5. [Directory Structure](#directory-structure)  
6. [How to Train the Model](#how-to-train-the-model)  
7. [How to Evaluate the Model](#how-to-evaluate-the-model)  
8. [Playing Against the AI](#playing-against-the-ai)  
9. [Dependencies](#dependencies)  
10. [License](#license)

---

## **Project Overview**

Gomoku (also known as Five in a Row) is a strategy board game played on a 15x15 grid. The goal is to get five of your pieces in a row, horizontally, vertically, or diagonally. This AI was developed using:

- **Reinforcement Learning (DQN)**: Adaptive learning through rewards and penalties.  
- **Minimax Algorithm**: Classical AI approach for decision-making.  
- **Human-Controlled Policy**: Allows a human to play against the AI.

---

## **Features**

- **Deep Q-Network (DQN)** for AI decision-making.
- **Minimax Algorithm** for baseline comparison.
- **Performance Evaluation**: Compete different policies against each other.
- **Human Play**: Allows manual play against the AI.

---

## **Installation**

### 1. **Clone the Repository**

```bash
git clone https://github.com/yourusername/gomoku-ai.git
cd gomoku-ai
```

### 2. **Set Up a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate      # On Windows use: venv\Scripts\activate
```

### 3. **Install Dependencies**

```bash
pip install -r requirements.txt
```

**Create a `requirements.txt` with the following:**

```
numpy
torch
matplotlib
scipy
```

---

## **Usage**

### **Running the Competition**

To run a competition between the AI policies:

```bash
python code/compete.py
```

This will use the default policies (Minimax vs. Submission AI) and display the results.

### **Evaluating Performance**

To evaluate the AI's performance over multiple games:

```bash
python code/performance.py
```

This script runs 30 games (default) and visualizes the scores and runtimes.

### **Playing as a Human**

To play against the AI manually:

1. Modify `performance.py` to set the `MIN` policy to `Human`:
   ```python
   from policies.human import Human

   policies = {
       gm.MAX: Minimax(BOARD_SIZE, WIN_SIZE),
       gm.MIN: Human(BOARD_SIZE, WIN_SIZE),
   }
   ```

2. Run the competition script:
   ```bash
   python code/compete.py
   ```

You will be prompted to input your moves.

---

## **Directory Structure**

```
gomoku-ai/
├── code/
│   ├── __init__.py           # Module initializer
│   ├── compete.py            # Runs the AI competition
│   ├── gomoku.py             # Gomoku game logic
│   ├── performance.py        # Evaluates AI performance
│   ├── nn.pt                 # Pre-trained neural network model
│   ├── perf.pkl              # Performance data
│   └── policies/
│       ├── __init__.py       # Initializes policies module
│       ├── human.py          # Human player policy
│       ├── minimax.py        # Minimax algorithm for AI
│       ├── submission.py     # DQN AI implementation
│       └── test.py           # Simple test script
├── .gitignore                # Ignore unnecessary files
└── README.md                 # Project documentation
```

---

## **How to Train the Model**

If you want to retrain the neural network:

1. **Modify `submission.py`** to include your training parameters.
2. **Run the training** by calling the `train_nn()` method:
   
   ```python
   nn.train_nn(final_score)
   ```

3. **Save the model** using the `save_nn()` method.

---

## **Dependencies**

- **Python 3.x**
- **PyTorch**
- **NumPy**
- **SciPy**
- **Matplotlib**

---

## **License**

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## **Acknowledgments**

- **Reinforcement Learning** concepts inspired by classic DQN implementations.
- **Minimax Algorithm** for baseline AI logic.

---

Happy coding! 🎮
