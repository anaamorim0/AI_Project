# AI_Project

# Connect Four AI — Informed & Adversarial Search (A* & MCTS)

This project implements an intelligent agent to play the game Connect Four against a human player using two different AI strategies:
- **A*** (informed but non-adversarial search)
- **Monte Carlo Tree Search (MCTS)** (adversarial search)

## Game Description

Connect Four is a two-player strategy game where the goal is to connect four of your tokens in a row, column, or diagonal on a 7x6 grid. Players take turns dropping tokens into one of the 7 columns.

## Implemented Strategies

### A* Search
A non-adversarial strategy using a heuristic evaluation function based on:
- Segment evaluation (number of tokens aligned)
- Move bonus (favoring the player whose turn it is)
- Scores from -50 to +50 depending on threats and opportunities

### MCTS (Monte Carlo Tree Search)
An adversarial strategy using Upper Confidence Bound for Trees (UCT) to select optimal moves based on simulations.

## Interface
A simple console-based interface:
- Displays the current board state
- Asks for player input
- Displays computer's move and updated board

## How to Run
```bash
python connect_four.py
