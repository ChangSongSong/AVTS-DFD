# AVM-FFD

### Training

Create a conda environment and install the dependencies
```sh
conda create -y -n avm-ffd python=3.8
conda activate avm-ffd
conda install -y pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
pip install -r requirements.txt
```

#### Phase 1 training
```sh
python src/main.py -c config/config/phase1/AVTS.yaml
```

### Development

#### Formatting code
```sh
pip install -r requirements.txt
pre-commit install
```
