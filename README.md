# tum-adlr-ss21-05

# Installation
## Clone the repository
```
git clone --recurse-submodules -j8 git@github.com:DanielHettegger/tum-adlr-ss21-05.git
```

## Install requirements
```
cd tum-adlr-ss21-05
pip install -r requirements.txt
cd lib/stable-baselines3
pip install -e .
cd ../..
cd KUKA-iiwa-insertion
pip install -e .
```

## Start training
```
python3 train_td3.py
```

## Demonstrate learned policy
```
python3 play_td3.py
```