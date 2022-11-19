# Pisces

This repository contains the evvaluation artifacts of our SoCC'22 paper
[Efficient Federated Learning via Guided Asynchronous Training](https://dl.acm.org/doi/10.1145/3542929.3563463).
It is implemented by extending an early version of [Plato](https://github.com/TL-System/plato), a new scalable federated learning research framework.

## 1. Overview

All the evaluations are done in **cluster deployment**.
In what follows, we assume you have an AWS account and can start a cluster of EC2 nodes. (Technically you can also run atop an existing cluster 
where you have sudo previliges with steps similar to what are shown below; however, this is not documented.)

### Table of Contents

* [Getting Started](#2-getting-started)
   * To run and manipulate a cluster of nodes in the AWS public cloud. 
- [Running Experiments](#3-running-experiments)
   * To replicate the experimental results presented in the paper.
- [Repo Structure](#4-repo-structure)
- [Notes](#notes)
- [Contact](#contact)

## 2. Getting Started


### 2.1 Prerequisites

#### 2.1.1 Anaconda Environment

In your host machine, you should be able to directly work in a **Python 3** Anaconda environment with some dependencies installed.
For ease of use, we provide a script for doing so:

```bash
# starting from [project folder]
cd experiments/dev
bash standalone_install.sh
# then you may need to exit your current shell and log in again for conda to take effect
```


#### 2.1.2 Install and Configure AWS CLI

One should have an **AWS account**. Also, at the coordinator node, 
one should have installed the latest **aws-cli** with credentials well configured (so that
we can manipulate all the nodes in the cluster remotely via command line tools.).

**Reference**

1. [Install AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html).
    * Example command for installing into Linux x86 (64-bit):
    ```bash
    # done anywhere, e.g., at your home directory
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    sudo apt install unzip
    unzip awscliv2.zip
    sudo ./aws/install
    ```
2. [Configure AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).
    * Example command for configuring one's AWS CLI:
    ```bash
    # can be done anywhere, e.g., at your home directory
    aws configure
    ```
   where one sequentially input AWS Access Key ID, AWS Secret Access Key, Default region name and default output format.

### 2.2 Launch a Cluster

Think of a name for the working folder and then create it by copying the
basic cluster management tools to the folder
`[project folder]/experiments`. 
Here is the example code where we create a folder named `socc22_ae` (where `ae` means artifact evalutation):

```bash
# starting from [project folder]
cd experiments
cp -r cluster_folder_template socc22_ae
cd socc22_ae
```

We will use this example folder name to demonstrate the rest of steps.

After creating the folder, you need to make necessary modifications to the following
cluster configuration files (see the comments for details):

```bash
# make some modifications that suit your need/budget
# 1. EC2-related
vim ec2_node_template.yml
# relevant key:
#    BlockDeviceMappings/Ebs/VolumeSize: how large is the storage of each node
#    KeyName: the path to the key file (relative to ~/.ssh/) you plan to use to 
#            log into each node of the cluster from your coordinator node
vim run.sh
# relevant variable:
#    LOCAL_PRIVATE_KEY: the path to the key file (should be absolute) you plan to use to 
#                      log into each node of the cluster from your coordinator node
#                      i.e., the same as the one you configure in ec2_node_template.yml

# 2. cluster-related
vim ec2_cluster_config.yml
# relevant key:
#     type and region for the server each client
# p.s. the provided images are specifically of Ubuntu 18.04

# 3. Github-related
vim setup.sh
# relevant variable:
#    GITHUB_REPO: please change it to your forked repo address
#    Otherwise you cannot apply your customization to the cluster
#    conveniently using our provided commands (later introduced in 2.3)
```

After modifying those configuration files, you can launch a cluster using commands:

```bash
# starting from [project folder]/experiments/socc22_ae
bash manage_cluster.sh launch
```

The sample stdout result looks like:

```
Launching 21 nodes ...
All 21 nodes are launched! Waiting for ready ...
All 21 nodes are ready. Collecting public IP addresses ...
```

**Remarks**

1. We provide you with `manage_cluster.sh` for a wide range of cluster-related control:

    ```bash
    # start an already launch cluster
    bash manage_cluster.sh start
    # stop a running cluster
    bash manage_cluster.sh stop
    # restart a running cluster
    bash manage_cluster.sh reboot
    # terminate an already launch cluster
    bash manage_cluster.sh terminate
    # show the public IP addresses for each node
    bash manage_cluster.sh show
    ```

### 2.3 Configure the Cluster

```bash
# starting from [project folder]/experiments/socc22_ae
bash setup.sh install
bash setup.sh deploy_cluster
```

The sample stdout results for the two commands look like:

```
Initialized node pisces-worker-1 (3.145.144.92).
...(similar lines)...
Updated the repo Pisces-private on pisces-worker-1 (3.145.144.92).
...(similar lines)...
Standalone installation finished on node pisces-worker-1 (3.145.144.92).
...(similar lines)...
```

and

```
Cluster server deployed on pisces-worker-1 (3.145.144.92).
...(similar lines)...
```

, respectively.

**Remarks** 
1. Make sure that your `~/.ssh/` has the correct key file that you specified in `../dev/ec2_node_template.yml`.
2. We provide you with `setup.sh` for a wide-range of application-related control:
    ```bash
    # update all running nodes' Github repo
    bash setup.sh update
    # add pip package to the used conda environment for all running nodes
    bash setup.sh add_dependency [package name (w/ or w/o =version)]
   ```

## 3. Running Experiments


### 3.1 Configuring Tasks

In the folder `[project folder]/experiments/exp_config_examples`, 
you can find the configuration files for running the experiments
mentioned in the paper.
For example, to replicate the Fig. 2 in the paper, 
you can run all the experiments configured in 
`[project folder]/experiments/exp_config_examples/fig2`.
Of course, you can also try your own experiments 
by customizing the configuration files.
The above-mentioned examples can serve as templates for you.

### 3.2 Running a Particular Task

Once you have prepared a legitimate configuration file for a task,
you can put it in your created experiment folder and then use 
the script `cluster_run.sh` to help you launch the task.

For example, if you want to run one of the experiment that is 
configured by `mnist-pisces.yml` to replicate Fig. 7, 
you can use the command

```bash
# starting from [project folder]/experiments/socc22_ae
cp -r ../exp_config_examples/fig7to9/ .
bash run.sh start_a_task fig7to9/mnist-pisces.yml
```

The sample stdout result looks like this

```
Task 20221031-194258 started on pisces-worker-1 (3.145.144.92).
...(similar lines)...

Use the command at local host if you want to stop it:
        bash run.sh kill_a_task fig7to9/20221031-194258
Use the command at local host if you want to retrieve the log:
        bash run.sh conclude_a_task fig7to9/20221031-194258
Use the command at a computing node if you want to watch its state:
        vim ~/Pisces/experiments/socc22_ae/fig7to9/20221031-194258/log.txt
```

We suggest you try this example first 
to see if you can get meaningful results.

**Remark**

1. We provide you with `cluster_run.sh` for a wide range of task-related control (no need to remember, 
because the prompt will inform you of them whenever you start a task, as mentioned above):
    ```bash
    # for killing the task halfway
    bash run.sh kill_a_task [target folder]/[some timestamp]
    # for fetching logs from all running nodes to the host machine
    # (i.e., the machine where you type this command)
    bash run.sh conclude_a_task [target folder]/[some timestamp]
    ```

## 4. Repo Structure

```
Repo Root
|---- plato      # Core implementation
|---- experiments                      # Evaluation
    |---- cluster_folder_template      # Basic tools for managing a cluster
    |---- dev                          # Cluster management backend
    |---- exp_config_examples          # Configuration files used in the paper
|---- packages   # External Python Package (e.g., YOLOv5)
```

## Notes

Please consider citing our paper if 
you use the code or data in your research project.

```bibtex
@inproceedings{pisces-socc22,
  author={Jiang, Zhifeng and Wang, Wei and Li, Baochun and Li, Bo},
  title={Pisces: Efficient Federated Learning via Guided Asynchronous Training},
  year={2022},
  isbn={9781450394147},
  publisher={Association for Computing Machinery},
  booktitle={Proceedings of the 13th Symposium on Cloud Computing},
  url={https://doi.org/10.1145/3542929.3563463},
  doi={10.1145/3542929.3563463},
  pages={370–385},
}
```

## Contact
Zhifeng Jiang (zjiangaj@cse.ust.hk).
