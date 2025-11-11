<p align="center">
    <img src="moose.png" alt="MOOSE Image" style="height:256px; width:auto;">
</p>

MOOSE
=====

Moose is a generalised planner that employs goal regression for synthesising programs represented as first-order decision lists. The programs can be executed as policies or for pruning the state space during search. The corresponding reference is

```
@inproceedings{chen-etal-aaai2026,
  author    = {Dillon Z. Chen and Till Hofmann and Toryn Q. Klassen and Sheila A. McIlraith},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  title     = {Satisficing and Optimal Generalised Planning via Goal Regression},
  year      = {2026},
}
```

# Setup

## Apptainer Installation
```
apptainer build moose.sif Apptainer
```

## Manual Installation
**[Optional]** If Moose is used for numeric planning, Python2 must be installed first. This is because Moose depends on Numeric Downward for generating training plans, which in turn depends on Python2. 
```
wget https://www.python.org/ftp/python/2.7.9/Python-2.7.9.tgz
tar xzf Python-2.7.9.tgz
cd Python-2.7.9
./configure --enable-optimizations
make altinstall
ln -s /usr/local/bin/python2.7 /usr/local/bin/python2
cd -
```

Create a virtual environment and then install requirements.
```
python3 -m venv .venv
source .venv/bin/activate
sh install.sh
```

# Usage
Policies are trained and used for planning as follows. 
```
# Training
./moose.sif train benchmarks/<DOMAIN>/domain.pddl

# Planning via policy execution
./moose.sif plan <DOMAIN>.model benchmarks/<DOMAIN>/domain.pddl benchmarks/<DOMAIN>/testing/<PROBLEM>.pddl

# Planning via optimal search
./moose.sif plan <DOMAIN>.model benchmarks/<DOMAIN>/domain.pddl benchmarks/<DOMAIN>/testing/<PROBLEM>.pddl --search symk
```

Call `./moose.sif <MODE> -h` for more instructions. If installation was done manually, replace `./moose.sif <MODE>` with `python3 <MODE>.py`.


## Example
```
# Training
./moose.sif train benchmarks/ferry/domain.pddl

# Planning via policy execution
./moose.sif plan ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p2_30.pddl

# Planning via optimal search
./moose.sif plan ferry.model benchmarks/ferry/domain.pddl benchmarks/ferry/testing/p0_30.pddl --search symk
```
