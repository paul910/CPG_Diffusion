# CPG_Diffusion
 
## Setup

This guide will walk you through the process of setting up this project, installing required packages using pip, setting up config variables in a .ini file, and downloading data into a specific directory. Please follow the steps below:

### Step 1: Clone the repository
First, you need to clone the GitHub repository to your local machine. Open your terminal and type:
```bash
git clone https://github.com/paul910/CPG_Diffusion.git
```
After you run this command, a new directory will be created with the same name as your repository.

### Step 2: Navigate to the Repository

Move to the cloned repository's directory:

```bash
cd CPG_Diffusion
```

### Step 3: Install the required packages

The project should contain a ```requirements.txt``` file with a list of all the necessary Python packages. You can install these using pip by running:

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Config Variables

Open ```config.ini``` in a text editor of your choice, and set your config variables according to the project's requirements. This file should be documented with instructions on what each variable should be set to.

### Step 5: Download Data

Currently, Data only available on the AWS-Machine:

If NFS is not installed on your machine, install it by typing the following command in your terminal:
```bash
sudo apt-get install nfs-common
```
    
Then, mount the EFS file system by typing the following commands in your terminal and replacing ```<repository-path>``` with the path to your cloned repository (example: ```/home/ubuntu/CPG_Diffusion```): 
```bash
sudo mkdir /mnt/efs
sudo mount -t nfs4 -o nfsvers=4.1,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=2,noresvport fs-9b946fe0.efs.us-east-2.amazonaws.com:/ /mnt/efs
sudo mkdir <repository-path>/data
sudo cp -RT /mnt/efs/Paul/data/processed <repository-path>/data/reveal
```

### Final Step: Run the Project

You can run the project by typing the following command in your terminal:

```bash
python main.py
```

### Final Step: Run the Project

You can run the project by typing the following command in your terminal:

```bash
python main.py
```

#### Mode: train
With the log_wandb flag set to True, the training results can be observed on wandb.ai as shown in the console output. To use wandb you need to create an account and set the API key in the console when prompted. Otherwise, the validation loss will be printed to the console after each epoch.

#### Mode: sample, forward_diffusion
For the 'sample' and 'forward_diffusion' modes, visual results are produced using matplotlib. Each mode will display one instance (a total of 6 time steps over the diffusion process) either for the forward_process ('forward_diffusion') or backward_process ('sample') from the test set.
Mode 'sample' needs a already pre-trained model (either use the one in folder ```models/model.pth``` or train your own model) otherwise the output will not make any sense. Therefore specify the path to the model in the 'config.ini' file and it will be loaded automatically.