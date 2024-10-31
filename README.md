Neighbourhood Density
=====================

**Neighbourhood Density is a Python tool for computing linguistic and sensorimotor neighbourhood densities.**

Downloading and installation
----------------------------

You should use Git to clone the package.  First make sure you have [Git intalled][git-download], and you have access to the 
repository on Github.  (Ask me for access if you don't have it.)  You may also have to 
[create and authorise an ssh key][github-ssh] for your Github account.  Then, in a command line, run:

```bash
git clone --recurse-submodules git@github.com:emcoglab/neighbourhood-density.git
```

(The `--recurse-submodules` option is important because LDM-Query is a command-line-interface wrapper around the main 
corpus analysis code, which is included as a Git submodule.  If you forgot to include `--recurse-submodules`, you can run
 `git submodule update --init` after your usual clone).

This will download all the code necessary to run the LDM-Query program.

Additional requirements are listed in `requirements.txt`.  You can automatically download and install these
dependencies using the `pip` tool, which is included with Python.  Once Python is installed, use `pip` to install the
dependencies like this:

```bash
pip install -r requirements.txt
```

If you use Python for more than just this, you may want to use [`virtualenv`][virtualenv] to isolate the packages you install, but 
this is not strictly necessary.

Finally, once all these modules are installed, you can run LDM-Query from the command line.  First go to the directory
you cloned to.  On Mac and Linux this is done with `cd`:

```bash
cd <path-to-cloned-neighbourhood-density>
```

Then invoke the program like:

```bash
python neighbourhood_density.py
```

You may first need to activate a virtual environment:

```bash
source venv/bin/activate
```




Getting updates
---------------

To see if there are updates, run the commands:

```bash
git fetch
git status
```

To pull updates into your local copy, use the following two commands:

```bash
git pull
git submodule update
```

Then everything should be up to date.  If you have made local changes you may need to [stash][git-stash] them first.

Configuration
-------------

Before Neighbourhood Density can be properly used, it must be configured so it knows where the files containing the corpora and LDMs 
are stored on your computer.

These are set in the file `config.yaml`, which is a text file in [YAML][yaml] format.  Comments in that file should explain
how to set your preferences.

When you have the required files downloaded and located where you want them, copy their *absolute* paths into the relevant 
places in `config.yaml`. Note that some of the files for the model must be located in specifically named directory hierarchies 
— these requirements are explained in comments in `config.yaml`.
