# gunlab

Dataset generator, offline leaderboard and live arena for neural aiming in classic Robocode.

Students train a multilayer perceptron (MLP) in Python and submit a single weights file. A fixed robot loads that file and aims with it, so entries differ only by their model. Battles run headless for the official board, or live on a projector so the class watches the robots compete.

There are two boards:
- the **offline board** scores each weights file on a hidden test CSV;
- the **arena board** ranks entries by the hit rate of their real bullets against fixed opponents.

## How it works

The RoboRumble opponents are classic Robocode Java archives (JARs), so the robots are written in Java. gunlab ships their sources, compiles them into a local Robocode installation and drives Robocode from Python. Students only use Python.

Both robots extend `GunBase`, which fixes the movement (orbit with random reversals and wall smoothing), the radar (lock) and the feature extraction:

- `gunlab.LoggerBot` fires real bullets at random guess factors (GF) and writes one comma-separated values (CSV) row per wave.
- `gunlab.NeuralBot` reads `weights.txt` and aims with the network. At the end of every round it writes its counters to `stats.txt`: whether the weights loaded, bullets fired, bullets that hit, waves, and waves whose prediction fell inside the hit window. Each submission is compiled as `arena.<Name>`, a subclass with its own weights and color.

Robocode lists robots compiled into `robots/` as development versions, with a trailing `*` (`gunlab.LoggerBot*`, `arena.<Name>*`).

Because both robots compute the features with the same code, the inputs seen in the arena have the same definition as the CSV columns.

Some opponents save what they learned about an enemy between battles. Before every battle, gunlab deletes any file an opponent saved about a `gunlab.*` or `arena.*` robot, so every battle starts from the same state.

Every tick the robot launches a virtual wave that travels at the speed of the bullet it would fire. When the wave reaches the enemy, the label is

```
GF = d · (β_break − β_fire) / asin(8 / v_b),   clipped to [−1, 1]
```

The symbols are:
- β_fire: the absolute bearing to the enemy at launch;
- β_break: the bearing from the wave origin to the enemy when the wave reaches it;
- v_b: the bullet speed;
- 8: the maximum robot speed;
- d ∈ {−1, +1}: the enemy's lateral direction at launch, the sign of its velocity component perpendicular to the line between the robots.

Waves that carried a real bullet have `real = 1`.

The enemy is 36 px wide, so a bullet aimed at a predicted GF hits when |predicted GF − GF| ≤ `gf_width`. Here `gf_width` = atan(18 / D) / asin(8 / v_b), and D is the distance from the wave origin to the enemy when the wave reaches it. This hit window defines the hit metrics below.

## Requirements

- Python ≥ 3.11 and Poetry ≥ 2.0. This is all `score` and training need.
- `setup`, `generate`, `arena`, `watch` and `bracket` also need a Java Development Kit (JDK) and classic Robocode, installed as below.

## Installing Java and Robocode

1. **JDK.** Version 21 is tested; 17 works too. On Debian or Ubuntu:
   ```bash
   sudo apt install openjdk-21-jdk
   java -version
   javac -version
   ```
   Both commands must answer; `javac: command not found` means only the runtime is installed. A `javac` older than `java` is fine. On another distribution, install its OpenJDK package that provides `javac`.
2. **Robocode.** Classic Robocode, not Tank Royale. Version 1.11.1 is tested:
   ```bash
   cd ~
   wget https://github.com/robo-code/robocode/releases/download/v1.11.1/robocode-1.11.1-setup.jar
   java -jar robocode-1.11.1-setup.jar
   ```
   If `wget` fails, download the `…-setup.jar` file from the Assets of https://github.com/robo-code/robocode/releases. When the installer asks for a folder, choose `~/robocode`, which is the default `home` in `gunlab.toml`; any other folder works if `home` is set to it. Check the installation:
   ```bash
   ls ~/robocode/libs/robocode.jar ~/robocode/robocode.sh ~/robocode/robots/sample/Walls.class
   ```
3. **Compile gunlab's robots into Robocode.** Go back to your project, the folder that contains `gunlab.toml` (`gunlab` for the instructor, `my-gun` for students), and run `setup` there:
   ```bash
   cd path/to/your/project
   poetry run gunlab setup
   ```
   Run it again after reinstalling Robocode, after changing a `.java` file, and after adding opponent JARs.

## Installation

Instructor:

```bash
git clone <repository URL> gunlab
cd gunlab
poetry install
cp gunlab.example.toml gunlab.toml
```

Set `[robocode] home` in `gunlab.toml` if Robocode is not in `~/robocode`, install Java and Robocode as above, then:

```bash
poetry run gunlab setup
```

Opponent JARs and tiers come next; see the instructor workflow below.

Students, in their own project:

```bash
poetry new my-gun
cd my-gun
poetry add git+https://github.com/algoritmos-de-aprendizado/gunlab.git
poetry run gunlab --help
```

If `poetry add` reports that the project's Python requirement is incompatible, set `requires-python = ">=3.11"` in the project's `pyproject.toml` and run it again.

Students who want to run `watch` locally also need Java and Robocode (above), a `gunlab.toml` with `[robocode] home`, and one `poetry run gunlab setup`.

## Configuration

`gunlab.toml` (paths are relative to the file; see `gunlab.example.toml`):

| Section | Keys |
|---|---|
| `[robocode]` | `home`; optional `launcher` (command list, default `robocode.sh` or `robocode.bat` in `home`); `quota_bytes`, the robot data-directory quota LoggerBot needs |
| `[battle]` | battlefield `width`, `height`; `rounds` per battle for `generate` and headless `arena` |
| `[live]` | `rounds` and `tps` (turns per second) for live battles |
| `[board]` | `bins`: number of GF bins used for `gf_bin` and for accuracy |
| `[data]` | `dir`; `battles_per_opponent` for `generate`; `virtual_stride`: every real-bullet wave is kept, and one virtual wave in this many |
| `[tiers]` | one list of robot names per tier, any name: `train`, `validation`, `test`, `candidates`… |
| `[arena]` | `submissions` and `results` folders; `opponents` of the arena board |

Robot names are `package.Class` for robots shipped as class files (`sample.Walls`) and `package.Class version` for JARs, matching the JAR file name `package.Class_version.jar`.

The instructor's `gunlab.toml` lists the hidden test opponents and is excluded by `.gitignore`.

## Instructor workflow

1. Download the opponent JARs into `<home>/robots/`, following [Getting opponent JARs](#getting-opponent-jars) below. Share CSV files, not JARs, since bot licenses vary.
2. Choose the tiers. The split is by opponent, so validation and test measure generalization to bots the model never saw:
   - **train:** the sample bots plus about 20 bots spread evenly over the ranking. Several kinds of movement matter more than many rows from a few bots.
   - **validation:** about 6 mid-ranked bots, used by students for model selection.
   - **test (hidden):** top-ranked bots. Keep their names out of anything students receive: gunlab can generate data against any bot.
3. Run `poetry run gunlab setup`. Run it again after adding JARs; it reports any name that does not match a JAR.
4. Generate the tiers:
   ```bash
   poetry run gunlab generate train
   poetry run gunlab generate validation
   poetry run gunlab generate test --battles 3
   ```
5. Choose the arena opponents. Top bots learn a consistent gun during the battle and dodge it, so their arena hit rate falls far below the offline one and stops reflecting the model. Pick bots that do not adapt:
   - put about 6 candidates in a `candidates` tier and run `gunlab generate candidates`;
   - score a reference model per opponent on `data/candidates.csv`;
   - run `gunlab arena reference.txt --headless --tier candidates`;
   - keep the bots whose arena hit rate stays close to their offline hit rate on real waves, and list them in `[arena] opponents`.

   Repeat the check with a stronger entry before the competition, since some bots switch movement only once they are hit often.
6. Give students `data/train.csv` and `data/validation.csv`. Keep `data/test.csv` and `data/candidates.csv` private.
7. Collect the submissions as `submissions/<student>.txt`.
8. Offline board on the hidden test set: `poetry run gunlab score --tier test`.
9. Arena board:
   - Official: `poetry run gunlab arena --headless` (rounds from `[battle]`).
   - Projector session: `poetry run gunlab arena` (live, rounds and speed from `[live]`).
   - Showcase against the top bots: `poetry run gunlab arena --tier test`.
   - Single match: `poetry run gunlab watch submissions/alice.txt "package.Class version"`.
   - Duel between two students: `poetry run gunlab watch submissions/alice.txt submissions/bob.txt`.
   - Knockout between students, live:
     ```bash
     poetry run gunlab bracket --seed-from results/<official arena board>.csv
     ```
     The top 8 entries are seeded by that board. In each duel the higher hit rate wins; bullet damage breaks ties, then the seed. When the number of entries is not a power of two, the top seeds skip the first round.

Boards and per-battle logs are written to `results/`.

### Updating gunlab

Save the downloaded update in the project root, either `files.zip` or a single `.py` / `.java` file, and run `sh update.sh`. The script moves every file into the repository layout and never touches `gunlab.toml`. Run `poetry run gunlab setup` afterwards when a `.java` file changed.

### Getting opponent JARs

Run these from the Robocode folder; they work in bash and fish.

1. Save the participant list (names and download links) and the current 1v1 ranking:
   ```bash
   cd ~/robocode
   curl -sL https://robowiki.net/wiki/RoboRumble/Participants \
     | sed 's/<[^>]*>//g' \
     | grep -E '^[A-Za-z0-9_.]+ [^,]+,https?://' \
     | sed 's/&amp;/\&/g' > participants.txt
   curl -sL 'https://literumble.appspot.com/Rankings?game=roborumble' \
     | grep -oE 'BotDetails\?game=roborumble&(amp;)?name=[^"&]+' \
     | sed -E 's/.*name=//; s/%20|\+/ /g' \
     | awk '!seen[$0]++' > ranking.txt
   wc -l participants.txt ranking.txt
   ```
   Both files should have about a thousand lines or more. `ranking.txt` lists the bots in rank order, one `package.Class version` per line.
2. Pick bots by rank. For example, every 60th bot from rank 30 for training:
   ```bash
   awk 'NR % 60 == 30' ranking.txt > picked.txt
   ```
   Other picks follow the same pattern, for example `awk 'NR == 160 || NR == 320' ranking.txt`.
3. Download the picked bots and keep only real JAR files:
   ```bash
   sed 's/$/,/' picked.txt > patterns.txt
   grep -F -f patterns.txt participants.txt \
     | awk -F'[ ,]' '{
         jar = $1 "_" $2 ".jar"; url = substr($0, index($0, ",") + 1)
         printf "curl -sL --max-time 60 -o \"robots/%s\" \"%s\"; if file \"robots/%s\" | grep -qE \"Java archive|Zip archive\"; then echo \"ok %s %s\"; else rm -f \"robots/%s\"; echo \"FAILED %s %s\"; fi\n", jar, url, jar, $1, $2, jar, $1, $2
       }' > download.sh
   sh download.sh | tee download.log
   ```
   - Each `ok` line gives a robot name to copy into a tier of `gunlab.toml`.
   - `FAILED` bots have dead links; pick a neighbor in the ranking instead.
   - Leave out rumble copies of the bundled sample bots, such as `sample.MyFirstRobot 1.0`; they clash with the originals.
4. Go back to the gunlab folder and run `setup`; it reports any tier name that does not match a JAR:
   ```bash
   cd path/to/gunlab
   poetry run gunlab setup
   ```

## Student workflow

1. Put `train.csv` and `validation.csv` in `data/`.
2. Train in NumPy. `examples/train_mlp.py` in the gunlab repository is a reference:
   - full-batch gradient descent with momentum;
   - one tanh hidden layer and a softmax over 31 GF bins;
   - soft targets over the hit window;
   - real-bullet waves weighted 4 times more than virtual ones;
   - the weights saved from the epoch with the best validation hit rate.
3. Export with `save_weights` (see below).
4. `poetry run gunlab score submissions/me.txt` scores the file on the real-bullet waves of `data/validation.csv`, with the same code used on the hidden test set. Compare the result with head-on aiming (always GF = 0) on the same tier; the reference trainer prints that number.
5. Optionally, with Robocode installed: `poetry run gunlab watch submissions/me.txt sample.Walls`.
6. Submit the `.txt` file.

## Dataset

| Column | Meaning |
|---|---|
| `distance` | distance to the enemy (pixels) |
| `bullet_power` | power of the bullet the robot would fire (0.1–3), a function of distance and both energies |
| `lateral_speed` | absolute enemy velocity perpendicular to the line between the robots |
| `advancing_velocity` | enemy velocity towards the robot (positive when approaching) |
| `enemy_speed` | absolute enemy velocity |
| `acceleration` | change of `enemy_speed` since the previous scan |
| `ticks_since_reverse` | ticks since the enemy's lateral direction last changed, capped at 200 |
| `wall_ahead` | fraction of the maximum escape angle reachable in the enemy's lateral direction before the wall |
| `wall_behind` | same, in the opposite direction |
| `dist_last_10` | distance the enemy covered over the last 10 scans |
| `last_gf` | GF of the most recent wave that reached the enemy in this battle (0 before the first one) |
| `mean_gf_10` | mean GF of the last 10 waves that reached the enemy |
| `hist_all_0` … `hist_all_4` | 5-bin histogram of the GFs of all past waves in this battle, decayed by 0.998 per wave, normalized |
| `hist_real_0` … `hist_real_4` | same over real-bullet waves only, decayed by 0.95 per wave |
| `gf` | label, GF ∈ [−1, 1] |
| `gf_width` | the enemy's half-width in GF units when the wave reached it (the hit window) |
| `gf_bin` | `round((gf + 1) / 2 · (bins − 1))` |
| `real` | 1 if the wave carried a real bullet |
| `round` | round number within the battle |
| `opponent` | opponent robot name |

The first 22 columns are the inputs, in this order (`gunlab.model.FEATURES`). `gunlab.dataset.load_csv(path)` returns them as `X`, together with `gf`, `width`, `real` and `opponent`.

The history features use only waves that reached the enemy before the row's own wave was launched, so the label never leaks into them. They start empty in each battle and carry over between its rounds.

Every real-bullet wave is kept; virtual waves are thinned by `[data] virtual_stride`.

## Model contract

- Input: the 22 features, standardized as z = (x − mean) / std with the mean and std stored in the file.
- Layers: dense, a ← f(W a + b), with W of shape (n_out, n_in) and f ∈ {identity, tanh, logistic, relu}.
- Output with K values:
  - K = 1: the GF itself, clipped to [−1, 1].
  - K > 1: one score per GF bin; the robot aims at the center of the highest-scoring bin, −1 + 2k / (K − 1).

K does not need to match `[board] bins`: the boards convert every predicted GF to board bins.

The offline board scores real-bullet waves (`--all-waves` adds the virtual ones) and is ranked by **hit**:

- **hit:** the fraction of waves where |predicted GF − GF| ≤ `gf_width`, the waves where the bullet would have hit.
- **accuracy:** the fraction of waves where predicted and true GF fall in the same board bin.
- **MSE:** the mean squared error between predicted and true GF.

Accuracy and hit reward the most probable bin. MSE rewards the conditional mean, which can fall between two likely positions and hit neither.

The arena board is ranked by **hit rate**, bullets that hit divided by bullets fired. Unlike bullet damage, the hit rate does not depend on how long the robot survives, so it varies much less between battles. The board also shows:

- **would hit:** the same hit-window check as offline, over every wave of the battle, virtual ones included;
- bullet damage.

Offline hit is an upper bound. In battle the gun fires along the aim computed one tick earlier, bullets can collide, and opponents may react to the gun.

Export:

```python
from gunlab.model import save_weights

save_weights(
    "submissions/alice.txt",
    weights=[W1, W2],  # W1: (hidden, 22), W2: (K, hidden)
    biases=[b1, b2],
    activations=["tanh", "identity"],
    mean=mean,
    std=std,
)
```

`mean` and `std` must be the standardization used in training. Zero values in `std` are stored as 1, so apply the same rule when training.

File format (plain text, `#` starts a comment):

```
gunlab-mlp 1
inputs 22
mean m1 ... m22
std s1 ... s22
layers L
dense n_in n_out activation
<n_out lines of n_in weights>
<one line of n_out biases>
...one dense block per layer
```

`gunlab.model.load_weights(path).guess_factor(X)` runs the same forward pass as the robot.

## Live arena on a projector

- Each battle opens the Robocode window and closes it at the end; the terminal shows the updated board between battles. Robocode remembers the window size, so maximize it once.
- Painting: click the robot's name on the right of the battle view and press **Paint** in its console. The robot then draws:
  - each bullet's wave, with the network's scores along it (larger, redder dots for higher scores);
  - the predicted (red ring) and actual (green dot) positions when the wave reaches the enemy;
  - a status line with hits per bullet fired, waves, and the live would-hit rate.
- 35 rounds at viewing speed take long. Use `[live] rounds` for the session and `gunlab arena --headless` for the official board.
- `gunlab arena --tier test` plays the hidden top bots, a good closing showcase. Expect low hit rates: these bots learn the gun and dodge it.
- `--tps 30` gives close viewing; higher values move faster.

## Commands

All commands accept `--config PATH` before the command name, for example `gunlab --config other.toml score`.

| Command | Purpose |
|---|---|
| `gunlab setup` | Compile the robots into Robocode, set the data quota, create the folders, list missing opponents |
| `gunlab generate TIER [--battles N] [--rounds N] [--live] [--tps N]` | Run LoggerBot against every robot of the tier and write `data/TIER.csv` |
| `gunlab score [WEIGHTS…] [--tier T \| --data CSV] [--all-waves] [--by hit\|accuracy\|mse]` | Offline board; default: every file in `submissions/`, real-bullet waves of `data/validation.csv`, ranked by hit |
| `gunlab arena [WEIGHTS…] [--headless] [--rounds N] [--tps N] [--tier T]` | Arena board ranked by hit rate, live by default; `--tier` plays that tier instead of `[arena] opponents` |
| `gunlab watch WEIGHTS OPPONENT [--rounds N] [--tps N]` | One live battle against a robot name or a second weights file |
| `gunlab bracket [WEIGHTS…] [--top N] [--seed-from CSV] [--headless] [--rounds N] [--tps N]` | Knockout duels between submissions, live by default; seeds from the given board CSV (default: the latest arena board) |

## Troubleshooting

- **`battle … failed`:** read `<home>/gunlab_tmp/last_run.log`. The usual cause is a robot name that does not match its JAR, or a missing JAR.
- **`NO WEIGHTS LOADED`, `(no weights)` on the board, or `no weights` on the status line:** NeuralBot fell back to head-on aiming. Its console prints the path where it looked for `weights.txt`. gunlab copies the file to `robots/arena/<Name>.data/` and `robots/.data/arena/<Name>.data/`.
- **`update.sh` prints `nothing to update`:** the download was not saved in the project root. Move `files.zip` or the downloaded file there and run it again.
- **A NeuralBot fails with an input-count error:** the weights file was trained for a different feature set. Retrain on the current CSVs, which have 22 inputs.
- **`generate` reports few or no rows:** raise `quota_bytes` and run `gunlab setup` again.
- **Robocode does not start:** check that the Robocode release supports the installed Java version, or set `[robocode] launcher`.
- **A live window stays open after its battle:** close it to continue.
