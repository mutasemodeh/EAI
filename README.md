# EAI
EAI codebase
## Description
This is the code BASE for all EAI related stuff

## Setup Instructions

```bash
git clone https://github.com/yourusername/your-repo-name.git
cd EIA_repo
conda env create -f environment.yml
conda activate your_env_name
python your_script.py
```


## Push/Pull your change 
If you have installed extra packates and would like to share them across the platform
```bash
git add myChange.py
git commit -m "Updated myChange.py and add xxx"
git push
```
To pull the latest changes:  

```bash
git pull
```

## Add Conda Environment Dependencies

If you have installed extra packages and would like to share them across the platform:

```bash
conda env export > environment.yml
git add environment.yml
git commit -m "Updated environment.yml with new dependencies"
git push
```
