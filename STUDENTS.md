# gunlab: student instructions

You train a multilayer perceptron (MLP) that aims a Robocode gun, and submit its weights as a single text file. The robot, its movement, its radar and its features are fixed; only your model differs from the other entries.

## Setup

Requirements: Python ≥ 3.11 and Poetry ≥ 2.0.

```bash
poetry new my-gun
cd my-gun
poetry add git+https://github.com/algoritmos-de-aprendizado/gunlab.git
```

Unzip `gunlab-handout.zip` and move the contents of its `handout/` folder into the project root:
- `data/train.csv` and `data/validation.csv`;
- `train_mlp.py`, the reference trainer;
- `gunlab.toml`.

## Workflow

1. Train. The reference trainer is a starting point:
   ```bash
   poetry run python train_mlp.py data/train.csv data/validation.csv submissions/<id>.txt
   ```
2. Check the file. This uses the same scoring code as the hidden test set:
   ```bash
   poetry run gunlab score submissions/<id>.txt
   ```
3. Optional: watch your robot fight.
   - Install classic Robocode and a Java Development Kit (JDK) with `javac`.
   - Set `[robocode] home` in `gunlab.toml`, then:
   ```bash
   poetry run gunlab setup
   poetry run gunlab watch submissions/<id>.txt sample.Walls
   ```

The README of the gunlab repository describes the dataset columns, the guess factor (GF) label, the hit window and the weights format.

## Reference numbers (validation, real-bullet waves)

| Model | hit | accuracy | MSE |
|---|---|---|---|
| Head-on aiming (always GF = 0) | 0.220 | – | – |
| Reference trainer (`train_mlp.py` as given) | 0.291 | 0.129 | 0.273 |

A submission is useful only if its hit is above head-on aiming. The goal is to beat the reference trainer.

## Rules

1. **One file per student:** `<id>.txt`, where `<id>` is your student ID. Use only letters, digits and `_`; the name becomes your robot's name.
2. **The file must load:** `gunlab score` must accept it with an empty `error` column. That means the `gunlab-mlp 1` format, 22 inputs, dense layers, and activations among identity, tanh, logistic and relu.
3. **At most 50,000 parameters,** weights and biases together. The robot runs the forward pass every turn within Robocode's time limit.
4. **Implementation:** train with NumPy only; no machine-learning frameworks.
5. **Data:** use only `train.csv` and `validation.csv`. Do not collect extra data, for example with `gunlab generate` against other robots: the hidden opponents must stay unseen.
6. **Deadline:** <date and time>, submitted to <where>.

## Scoring

- **Offline board:** hit on the real-bullet waves of a hidden test set, recorded against opponents that appear in neither CSV.
- **Arena board:** the hit rate of your robot's real bullets (hits per bullet fired) against hidden opponents, 35 rounds per battle.
- **Weight in the grade:** <how the two boards count>.

The final session shows the arena live on the projector.
