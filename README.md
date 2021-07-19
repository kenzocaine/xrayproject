# Automated X-ray image diagnosis

- In collaboration with David McGady and Christian Hed
- This project is meant to generalize to all kinds of pathological lung diseases that can be detected in an X-ray image
- Segmentation is done using a modified U-net model
- Classification is done using a conventional CNN
- Data taken from https://www.kaggle.com/kmader/pulmonary-chest-xray-abnormalities


# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for xrayproject in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/xrayproject`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "xrayproject"
git remote add origin git@github.com:{group}/xrayproject.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
xrayproject-run
```

# Install

Go to `https://github.com/{group}/xrayproject` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/xrayproject.git
cd xrayproject
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
xrayproject-run
```
