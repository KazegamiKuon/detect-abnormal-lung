# Deep-learning-and-Machine-Learning-Workflow

This is template repository. Structure and automated workflow for AI project.

Programming language is **python**. However for easy test, demo and debug, I usualy code in by [jupyter notebook](https://jupyter.org/install) on [anaconda](https://docs.anaconda.com/anaconda/install/)

You should install anconda before jupyter

Anyway for begin, we start with python using terminal and git

## Who are you?

If  you configed your git then pass.

Check your config:

```script
git config --list --show-origin
```

If your **email** and **name** already existed then you configed or you can change your configuration too

```script
git config --global user.name your_user_name
git config --global user.email your_email
```

## Create new repository

You create repository on github and then follow the directions on github.

**Remember**, you should create your folder with name same your repository name and cd to that folder before run script.

## Create workflow

For easy quality control and merge, I highly recomend you should follow one workflow. I follow this [workflow](https://nvie.com/posts/a-successful-git-branching-model/) also add in something. Check my workflow to see more.

You must create **develop** *branch* before starting. Highly recomend you create it on github. Then change your branch.

```script
git checkout develop
```

Every time when you want to create a branch, please create from develop
branch. After you have created, change branch again.

## Update (checkin) and push (checkout) your code

Well this step, you should do in your local branch or maybe it call as feature branch.

```script
git pull
git add .
git commit -m "message/ description when commit"
git push
```

When you done all your feature and wanna commit to dev, then easy do it at github.

* **Step 1**: Go to full request at github.

* **Step 2**: Create full request for your branch (should be feature branch if you not product own).

* **Step 3**: Chose your branch (should be feature branch) and develop branch where it commit to.

* **Step 4**: Review and if okie, click create full request.

* **Step 5**: Your code woulde be reviewed by mentor or quality controler. Check, fix if something happen, else do next task.

* **Step 6**: Create your feature branch and change to it. Return step 1 if you done it.

## Install/Uninstall/Export your environment

First, you could down some support library. You can modify what library could get in **setup-library.sh**

```script
source setup-library.sh
```

About environment, for easy control, I make file bash for easy do it at [git-actions-practice](https://github.com/KazegamiKuon/git-actions-practice). That reponsitory automatic download as default by run file bash uper.

To control environment, read README at **git-actions-pratice/conda-environment**

## (coming soon)
