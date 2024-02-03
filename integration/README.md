This is the **Module Integration Project (MIP)**, which provides a means for running
all the TA1 modules from USC/ISI/Inferlink in a reproducible, parallelized
fashion:

* legend_segment
* map_crop
* text_spotting
* legend_item_segment
* legend_item_description
* line_extract
* polygon_extract
* point_extract
* georeference


# 1. Goals

The goals of this project include:

* The creation of docker containers for running all the modules
* A clear definition of the input files, output files, and inter-module 
  dependences for each module
* A tool that can run some or all of the modules, in the correct
  order, with a single, simple command line
* The ability to run (or re-run) a module from existing inputs


# 2. Conceptual Overview

Your host machine will have three directories:
* 
* `/ta1/input`: where the (static) input files live (this directory will
  contain the contents of the `ta1-data` repo)
* `/ta1/output`: where the results from each job (run) will live
* `/ta1/temp`: scratch space used when running each job

_Here we have used `/ta1` as the root of these three dirs, but in practice they
can each live anywhere._

These three directories are mounted as volumes when the modules' containers are run.

Each invocation of the MIP tool, to run one or more modules, is called a "job".
Each job is given a simple, short name, and the results from the job are stored
in a directory with that name. For example, if your job is named `alpha`, you
will find your results in `/ta1/output/alpha` (and `/ta1/temp/alpha`).

When you run the MIP tool, it will execute each of the known modules in the
proper order. The results from each module are in the job directory under the
name of the module. For example, if your job is named `alpha`, the results from
the map_crop module will be:

* `/ta1/output/alpha/map_crop.task.txt`: text file whose presence indicates to the
  system that the map_crop operation succeeded. The contents of the file have some
  basic information about the run, including elapsed time.
* `/ta1/output/alpha/map_crop.docker.txt`: the log file from the dockerized execution
  of the module, i.e. anything that gets printed out to stdout or stderr. If the
  module crashes, this is where to look. (At the top of this file are command
  lines you can use to manually start the docker container, including module options
  and volume mounts -- this is very handy for debugging.)
* `/ta1/output/alpha/map_crop/`: the directory containing the output from the module
  itself

Each module in the system has its own unique set of command line options. The `config.yml`
file is used to specify the various switch names and values. The syntax allows for the use
of a few variables, which are expanded at runtime to point to the proper directories. Here
is an example of a module that has three switches: one for the map input, one for the results
from a predecesssor module, and one for the output:

```yaml
fun_module:
    input_tif: $INPUT_DIR/$MAP_NAME/$MAP_NAME.tif
    json_feed: $OUTPUT_DIR/feed_module/$MAP_NAME.foo.json
    output_dir: $OUTPUT_DIR/fun_module
```

Finally, we need to run the MIP tool itself. There are just four switches:

* `--config-file`: path to the `config.yml` file (optional)
* `--job-name`: short string
* `--target-task-name`: name of the module to be run
* `--map-name`: name of the map image

The target task name can be any one of the known modules, or the special module `end`
which means "run everything". An example command line invocation would be:

```
$ ./mip.py --job-name alpha --target-task-name map_crop --map-name AK_Dillingham
```


# 2. MIP System Setup

_In the following, we will assume you are using `/ta1` for your setup._

1. Obtain a machine to run on: multiple cores, good GPUs, 128GB RAM (a 
   p2.xlarge EC2 instance seems to work well).
2. Make the three dirs: `mkdir /ta1/input /ta1/output /ta1/temp`
3. Check out the repo `ta1-data` and put its contents into `/ta1/input`
3. Check out the repo `usc-umn-inferlink-ta` and cd into its `integration` directory
4. Start the virtual environment: `poetry shell`
5. Pull all the prebuilt docker containers: `cd docker-tools ; ./build.sh --pull` ; cd ..`
6. Run `./mip.py` to your heart's content.


# 3. Adding a New Module

This section explains how to add a new module.

## 3.1. The Docker Containers

**TODO:** container design; dir layout; how to build the containers

## 3.2. The Tasking System

**TODO:** use of `luigi`; adding & registering a new Task class; adding to the config file;
adding a verification step


# 4. Advanced Topics

**TODO:**

* luigi configuration -- logging, scheduler
* known bugs
