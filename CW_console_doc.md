# Specific command to login CW with Keisuke's account 

ssh -o IdentitiesOnly=yes kkamata+cwb607@sunk.cwb607-training.coreweave.app





# Welcome to the CoreWeave Training Cluster

The SA team has put together a training Slurm/SUNK (Slurm on Kubernetes) GPU cluster with some exercises on it that you can use to get hands-on with CoreWeave technologies. This is a guide to how to get access and what you can do there. Direct questions/complaints to Tara Madhyastha.

Before a live workshop, complete steps 1, 2 and 3.

Note: Consider the things you can try to be experimental and in flux until this note goes away. Feedback to Tara Madhyastha
Table of Contents
Table of Contents	0
Getting On	0
Step 1. Get an invitation	0
Step 2. Upload your SSH key	1
Step 3. Log in to the cluster	4
Things you can try	5
Things to install locally	5
UV	5
Copy the notebooks locally	5
Launching a Marimo notebook	5
Introduction to Slurm Commands	10
Running NCCL Jobs	10
Look at the Structure of a Tiny Pytorch Example in Marimo	10
How Do You Use Object Storage?	11
Logging in to WandB and Playing With Nanochat	11
Appendix A. Troubleshooting	11
I am having difficulty logging in.	11
I am having difficulty launching a Marimo notebook with the script above.	12
I get an error that my WandB key has too many characters.	12

Getting On
Step 1. Get an invitation
Ask SA to invite you to the CoreWeave Training Cluster Org (organization ID is cwb607). The email you will use should take the form of your normal CoreWeave email +cwb607 (e.g. tmadhyastha+cwb607@coreweave.com). You should ask to be added to group trainee.Then you will get an email from the CoreWeave Team (something like below) that you will need to click through to register and log in. This is how a new user accesses CoreWeave.


Step 2. Upload your SSH key

To access SUNK you need to upload a public SSH key to the cluster. This, combined with the private SSH key that you have on your computer, will allow you in. 

If you have a public/private SSH key combo, feel free to use the public key you already use. 

If you do not have one yet, generate a key using the following command (on a Mac):

ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 

WHEN PROMPTED FOR A PASSPHRASE, DO NOT GIVE ONE!

You should see output something like the following (with your username instead of tmadhyastha). You don’t need to save any of this information but make a note of where your public key is saved (highlighted here in yellow).

Your identification has been saved in /Users/tmadhyastha/.ssh/id_ed25519
Your public key has been saved in /Users/tmadhyastha/.ssh/id_ed25519.pub
The key fingerprint is:
SHA256:ek3l9mqvyZ1Lo6eH7O5ReEP16msnJPNN26Nq2xOYEXE tmadhyastha@CW-DC12RHQN7W-L
The key's randomart image is:
+--[ED25519 256]--+
|           ..E  .|
|           ..  ..|
|            o . .|
|           + o . |
|        S . B =  |
|       . o +o*...|
|      . . ...**oo|
|       .   o=O**+|
|          .B&XO+.|
+----[SHA256]-----+

View the contents of your public key on the terminal and copy it.

cat ~/.ssh/id_ed25519.pub

Now, assuming you are logged on to the Console, navigate to the User Settings icon in the bottom-left corner, and select Settings (Figure 1).


Figure 1.

Once you are on the settings page, scroll down to the box with SSH public keys in it, and paste in your SSH public key in the box that says SSH Public Keys (Figure 2). 


Figure 2. 
Step 3. Log in to the cluster
Start by adding your SSH private key to your keychain. This will ensure that you do not need to type the path every time you log in. Use the path to your private key in the command below. For example, if you generated a new key with the commands in the previous step, you would run:

ssh-add ~/.ssh/id_ed25519

You can then log in to the cluster with the following command.

ssh -o IdentitiesOnly=yes $(whoami)+cwb607@sunk.cwb607-training.coreweave.app
 
If this doesn’t work right away, you may need to wait a minute or two for your SSH public key to sync with the Slurm cluster.

The flag -o IdentitiesOnly=yes is necessary only if you encounter errors related to “too many authentication failures”, which can occur if you use Teleport a lot. 

If you are successful you will see something like the following. Ignore the warning and answer yes (see blue text)  if you are sure that you want to connect.

The authenticity of host 'sunk.cwb607-training.coreweave.app (166.19.16.193)' can't be established.
ED25519 key fingerprint is SHA256:xsEIy12RYbtZqqvBdZ1Rrl/Woj35epP0SLytEDJTk7Q.
This host key is known by the following other names/addresses:
    ~/.ssh/known_hosts:62: 166.19.16.193
Are you sure you want to continue connecting (yes/no/[fingerprint])? yes
Warning: Permanently added 'sunk.cwb607-training.coreweave.app' (ED25519) to the list of known hosts.
Creating directory '/mnt/home/tmadhyastha+cwb607'.
Welcome to a CoreWeave Slurm HPC Cluster

The cluster has the following tooling pre-installed on the login node
…

Things you can try
Now that you have logged on there are a variety of things you can try. We are creating a small playground of examples and activities here. Many of the examples are available as notebooks so that you can easily cut and paste commands, so one of the first things you may want to do is to set that up. 
Things to install locally
UV
uv is a fast tool for managing Python dependencies and packages. While logged in to the login node of the cluster, type the following commands.

curl -LsSf https://astral.sh/uv/install.sh | sh
exec -l $SHELL

These commands install the programs and restart your shell to add these commands to your PATH environment variable, without installing for all the users on the system.

Copy the notebooks locally
We have created a number of notebooks described below. You may want to copy them to your home directory so that you can edit them and save any changes you make as you work through them. Otherwise you will get permissions errors. To do that, type the following command from your slurm login node. 



cp -r /mnt/data/notebook-examples $HOME
chmod -R u+w $HOME/notebook-examples

Launching a Marimo notebook

For the purposes of these workshops, we will launch a marimo notebook that is executing on the login node of the SUNK cluster. Note that for actual work using a GPU, most users will ultimately want to set up a notebook that is executing on a compute node, following these instructions. 

The essential idea is that you need to launch the notebook on the node where you want it, and create a secure ssh tunnel to access it. This script below will make the setup a little easier. To use it, copy it and save it to a file called startmarimo-uvx in your home directory on your local laptop. 


#!/bin/bash

# Set NOTEBOOK to argument or default to notebook.py
if [ -z "$1" ]; then
   echo "No notebook specified, defaulting to notebook.py"
   NOTEBOOK="notebook.py"
else
   echo "Using notebook: $1 in your home directory"
   NOTEBOOK="$1"
fi

slurmlogin=sunk.cwb607-training.coreweave.app

# 1. Get a random open port
PORT=$(( ( RANDOM % 2000 )  + 8000 ))
USER=$(whoami)+cwb607


echo "Creating the ssh tunnel to access Marimo on port ${PORT}..."
ssh -N -L ${PORT}:localhost:${PORT} ${USER}@${slurmlogin} &
TUNNEL_PID=$!
echo "Starting Marimo server on slurm login node..."
ssh ${USER}@${slurmlogin} ". /mnt/home/${USER}/.local/bin/env; nohup uvx marimo edit ${NOTEBOOK} --headless --host 0.0.0.0 --port ${PORT} &"

echo "Terminating tunnel (PID $TUNNEL_PID)..."
kill $TUNNEL_PID


Then use it as follows:

Make it executable

chmod +x startmarimo-uvx

If you haven’t already, add your SSH private key to your keychain, for example:

ssh-add ~/.ssh/id_ed25519

Run the script:

./startmarimo-uvx


If you have successfully been able to launch a notebook and set up an ssh tunnel, you will see the following output. To connect to your notebook you need to connect to the URL as shown in Figure 3. 


Figure 3. 

You may be prompted by your browser to install a marimo app (Figure 4). Go ahead and do that if you are; it will streamline your cut and paste experience. 

Figure 4. 

When you are finished with your notebook, be sure to shut it down and clean up by clicking on the red X on the top right corner (Figure 5). You should see a message indicating that your ssh tunnel has been terminated. 

Figure 5.

Once you have opened a notebook, you can open a shell terminal onto the slurm login node. Do so by toggling the developer panel in the lower left corner of Marimo (purple arrow, Figure 6) and clicking on the terminal icon (red arrow, Figure 6). 


Figure 6. 
Introduction to Slurm Commands
Assuming you have created the startmarimo-uvx command as described in Section “Launching a Marimo Notebook”, you can start the slurm tutorial with the following command:


./startmarimo-uvx notebook-examples/slurm-intro.py

Note that because the commands we use to manage slurm are usually bash commands, this notebook is a markdown document and does not have executable cells. 
Running NCCL Jobs

If you are familiar with the basics of slurm commands, then get your hands on some NVIDIA Collective Communications Library tests to test out the network. 


./startmarimo-uvx notebook-examples/nccl-test.py

Look at the Structure of a Tiny Pytorch Example in Marimo

Feel the pulse and excitement of an actual Marimo notebook - everything we have done so far has been static and boring with markdown. Notebooks have a lot more power than this. To launch one of the examples ripped straight from Pytorch documentation and see what a notebook looks like (albeit, one written for Jupyter, so it’s not taking advantage of all the great Marimo features) you can do the following.


./startmarimo-uvx notebook-examples/pytorch_quickstart_tutorial.py

Nano
How Do You Use Object Storage?
This tutorial is a VERY basic introduction to how you configure and access object storage. It does not go into the performance benefits of LOTA. 

./startmarimo-uvx notebook-examples/storage_tutorial.py


Logging in to WandB and Playing With Nanochat

In this section, we will play with running Nanochat, a full out implementation of an LLM. This section assumes that you have a WandB account. To log to WandB you need (a) to have the wandb package installed in your environment, and (b) to create a key. 

Launch the tutorial as follows. 


./startmarimo-uvx notebook-examples/nanochat_tutorial.py


Click here to create a key. 
Appendix A. Troubleshooting
I am having difficulty logging in.
Make sure you used the right username:
Your username might not be the same as your coreweave username. Long story.
Make sure you used the right private key:
Double check the spreadsheet to make sure you have the correct private key and that it didn’t get garbled if you used an editor.
Incorrect permissions:
Only you should be able to read your private key. This will set permissions correctly. Replace cwkey with the name of your key. 
chmod 400 ~/.ssh/cwkey
Too many authentication failures: Try using only this private key and nothing else.
ssh -v -o IdentitiesOnly=yes -i ~/.ssh/cwkey [insert your username]@sunk.cwb607-training.coreweave.app
Format errors:
The private key should begin and end with the lines “-----BEGIN OPENSSH PRIVATE KEY-----” and “-----END OPENSSH PRIVATE KEY-----” - don’t delete those - and make sure there are no other spaces or characters before or after those lines. The newlines should be preserved.
General debugging:
provide the -v or -vvv flags to ssh to get more detail about the error. 
I am having difficulty launching a Marimo notebook with the script above. 
Make sure you used the right username:
This line below should evaluate to the correct login name for slurm, but only if your name on your mac is the same as your coreweave user name. If not, replace everything after the equal sign with your coreweave login (where you got the invitation email).
USER=$(whoami)+cwb607
Make sure you installed uv
Check instructions to make sure you installed uv. You should be able to log in to the login node and type 
which uvYou should see that it is in your path. 
I get an error that my WandB key has too many characters.
Make sure that you have upgraded to the most recent version of the wandb package. 
