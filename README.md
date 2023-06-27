# AVM-FFD

## How to run
### prepare dataset
- Phase 1
Download LRW dataset to `/Datasets/`

- Phase 2
Download FaceForensics++ dataset to `/Datasets/`

### Training
- Phase 1
Check the settings are correct in `/config/phase1/AVTS.yaml`,
then run the command:
`python3 src/main.py --config config/phase1/AVTS.yaml`

- Phase 2
Check the settings are correct in `/config/phase2/all.yaml`,
then run the command:
`python3 src/main.py --config config/phase2/all.yaml`

### Testing
- Phase 1
Change the settings in `config/phase1/AVTS.yaml`,
then run the same command:
`python3 src/main.py --config config/phase1/AVTS.yaml`

- Phase 2
Change the settings in `/config/phase2/all.yaml`,
then run the same command:
`python3 src/main.py --config config/phase2/all.yaml`