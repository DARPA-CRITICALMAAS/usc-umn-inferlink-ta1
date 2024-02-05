This is the **Module Integration Project (MIP)**, which provides a means for
running all the TA1 modules from USC/ISI/InferLink in a reproducible,
parallelized fashion:

* legend_segment
* map_crop
* text_spotting
* legend_item_segment
* legend_item_description
* line_extract
* polygon_extract
* point_extract
* georeference

The goals of this project include:

* The creation of docker containers for running all the modules
* A clear definition of the input files, output files, and inter-module 
  dependencies for each module
* A tool (`mipper.py`) that can run some or all of the modules, in the correct
  order, with a single, simple command line
* The ability to run (or re-run) a module from existing inputs


# 1. Conceptual Overview

Your host machine will have three directories:

* `/ta1/input`: where the (static) input files live (this directory will contain
  the contents of the `ta1_integration_inputs` repo)
* `/ta1/output`: where the results from each job (run) will live
* `/ta1/temp`: scratch space used when running each job

_NOTE: Here we have used `/ta1` as the root of these three dirs, but in practice
they can each live anywhere._

These three directories are mounted as volumes when the modules' containers are
run.

Each invocation of the `mipper` tool, to run one or more modules, is called a
"job". Each job is given a (simple, short) name, and the results from the job
are stored in a directory with that name. For example, if your job is named
`alpha`, you will find your results in `/ta1/output/alpha` (and
`/ta1/temp/alpha`).

When you run `mipper`, it will execute each of the known modules in the proper
order. The results from each module are in the job directory under the name of
the module. For example, if your job is named `alpha`, the results from the
map_crop module will include:

* `/ta1/output/alpha/map_crop.task.txt`: a text file whose presence indicates to
  the system that the map_crop operation succeeded. The contents of the file
  have some basic information about the run, including elapsed time.
* `/ta1/output/alpha/map_crop.docker.txt`: the log file from the dockerized
  execution of the module, i.e. anything that gets printed out to stdout or
  stderr. If the module crashes, this is where to look. (At the top of this file
  are command lines you can use to manually start the docker container,
  including its module options and volume mounts: this is very handy for
  debugging.)
* `/ta1/output/alpha/map_crop/`: the directory containing the output from the
  module itself.

Each module in the system has its own unique set of command line options. The
`config.yml` file is used to specify the various switch names and values. The
syntax allows for the use of a few variables, which are expanded at runtime to
point to the proper directories. Here is an example of a portion of the
`config.yml` file for a module that has three switches: one for the map input,
one for the results from a predecessor module, and one for the output:

```yaml
extract_ore:
    input_tif: $INPUT_DIR/$MAP_NAME/$MAP_NAME.tif
    json_feed: $OUTPUT_DIR/feed_module/$MAP_NAME_foo.json
    output_dir: $OUTPUT_DIR/extract_ore
```

The  `mipper` tool (described below) takes command line switches to indicate
the config file location, the module(s) to be run, the map image to use, and the
name of the job.


# 2. MIP System Setup

_In the following, we will assume you are using `/ta1/...` as the root of the
three mipper directories. Feel free to change these paths._

1. Obtain a machine to run on: it requires multiple cores, good GPUs, and 128GB
   of RAM. (A p2.xlarge EC2 instance seems to work well.)
2. Make the three dirs: `mkdir /ta1/input /ta1/output /ta1/temp`
3. Check out the repo `ta1_integration_inputs` and into the directory 
   `/ta1/ta1_integration_inputs`.
4. Check out the repo `usc-umn-inferlink-ta` and cd into its `integration`
   directory.
5. Start the virtual environment: `poetry shell`
6. Pull all the prebuilt docker containers as follows:
   `cd docker-tools ; ./build.sh --pull ; cd ..`
7. Run `./mipper.py` to your heart's content.


# 3. Running the mipper Tool

The `mipper` tool runs one job, consisting of one or more module executions for
one map image.

The main command-line switches are:
* `--config-file / -c`: path to the config file, e.g. `./config.yml`
* `--map-name / -i`: name of the map image, e.g. `AK_Dillingham`
* `--job-name / -j`: name of the job, e.g. `dillingham-0203`
* `--module-name / -m`: name of the module to be run, e.g. `map_crop`; may be
  repeated

Example:
```
$ ./mip/apps/mipper.py --job-name alpha --module-name map_crop --map-name AK_Dillingham
```

The tool knows about the inter-module dependencies. It will run any run the
target module(s) given on the command line only after first running any required
predecessor modules, all in the proper order. The tool will skip the execution
of any module that has already successfully been run _for that job name_, as
indicated by the presence of a file named `MODULE.task.txt` in the output
directory. 

The module name can be any one of the known modules, or the special module named
`end` which means "run everything". (There is also a special module named
`start` which is the root of the dependency tree. Running mipper with
`--module-name start` is a good test to make sure the system is working
properly.)

Mipper supports a few other switches worth knowing:
* `--list-modules`: lists the names of the known modules and their predecessors
* `--list-deps`: displays a graph of the predecessors of the given target
  module, including information about which one have already been (successfully)
  run for the given job
* `--openai-key-file`: path to the file containing your OpenAI key
* `--force`: forces the execution of the given target module, even if it has
  already been run successfully


# 3. Adding a New Module

This section explains how to add a new module, in three steps:
1. Writing the module
2. Making the docker container
3. Adding the module to mipper
4. Adding the module to the config file

In our examples below, we will call the new module `extract_ore`, its container
`inferlink/ta1_extract_ore`, and its class `ExtractOreTask`.

_NOTE: Many of the existing module names have numbers in their names, e.g.
"5_map_crop" or "MapCrop5Task". This is just to alleviate my own confusion and
will soon go away._


## 3.1. Your New Module

Your module should follow all the conventions of a robust python app (described
elsewhere), but in particular must follow this rule:

_Any references to external files, such as input images or model weights or
output PNGs, **must** be specified on the command line, so that the files can be
located relative to the mipper system's required directory layout. Do not use
any hard-coded paths and do not assume anything lives at "`.`"._


## 3.2. Your New Docker Containers

Each module needs to be run in its own docker container.

1. Add new directory `integration/dockers/extract_ore`. (You can copy from one
   of the existing modules.)
2. Write a `Dockerfile` for your module and put it in the new dir. You should
   copy from one of the existing modules, but in general the file should contain
   these pieces, in order:
   1. The standard `FROM` line.
   2. The special line `# INCLUDEX base.txt`. This is an indicator to the mip
      docker build system to include the contents of another file into your
      file, to perform some functions common to all the docker containers.
   3. Whatever build requirements are needed for your module. 
   4. The special line `# INCLUDEX perms.txt`. This adds some user permission
      support and sets up the volumes used by all the docker containers.
   5. The two lines `CMD []` and `ENTRYPOINT ["python", "/path/to/module.py"]`.
3. Copy the file `build_docker.sh` from one of the existing modules into your
   new directory. In the simplest case, you'll need to only change the name of
   the docker image in the `docker build` step.

Your container will be invoked such that it takes your command line switches and
runs your python app from the `.../output/JOB` directory, (roughly) like this:
```
docker run \
    -v /home/ubuntu/dev/ta1_integration_inputs:/ta1/input \
    -v /home/ubuntu/dev/ta1_output:/ta1/outout 
    -v /home/ubuntu/dev/ta1_temp:/ta1/temp \
    --gpus all \
    --user cmaas \
    inferlink/ta1_extract_ore \
    --option1 value --option2 value...
```

To build your container, you need to first set the environment variable
`$REPO_ROOT` to point to where you have the `usc-umn-inferlink-ta1` repo checked
out. Then, you can just run your `./build_docker.sh`.

To build all the containers using mipper's build tools, first edit the file
`docker-tools/modules.sh` to add your new module (in three places, just follow
what's already there). Then, run the build script in that directory as
`./build.sh --build`: this will build each of the docker containers. (You can
then run `./build.sh --push` to push all the containers, if you have write
access to the inferlink DockerHub repository.)


## 3.3. Your New Mipper Module

To add your new module to the mipper system, you need to write a task class, add
some verification code, and register the module. Fortunately, this is easy:

1. In the `mip/module_tasks` directory, add a new file `extract_ore_task.py`.
   You can copy the file from one of the other `_task.py` files. Your new file
   will define a class `ExtractOreTask` with three easy parts:
   1. Set the class variable `NAME` to the sting `"extract_ore"`.
   2. Set the class variable `REQUIRES` to a list of task class names, for each
      of the classes (modules) that your new module requires in order to run.
   3. Optionally, implement the method `run_post()`. This function is executed
      by mipper after your module has successfully completed running inside its
      docker container, and so is the ideal place to add simple checks to make
      sure that any expected output files exist in the right places.
   4. If your new module is a "leaf node" in the module dependency graph, edit
      the file `all.py` to add `ExtractOreTask` to the `REQUIRES` class variable
      of `AllTask`.


## 3.4. Your New Config File Section

Finally, you need to add your new module to the `config.yml` file. You will do
this by adding a new section to the file and, for each command line switch your
python app uses, add a line for it.

For example, assume `extract_ore.py` has four swiches: one for the map input,
one for the results from a predecessor module, one for the output directory, and
one for the frobble parameter. Your new section of the config file would look
like this:

```yaml
extract_ore:
    input_tif: $INPUT_DIR/$MAP_NAME/$MAP_NAME.tif
    json_feed: $OUTPUT_DIR/feed_module/$MAP_NAME_foo.json
    output_dir: $OUTPUT_DIR/extract_ore
    frobble_value: 3.14
```

_NOTE: if your switch takes no parameter (such as a boolean flag), you can't
leave the "value" part of in the YAML line empty: set it to `""` instead. 


# 4. The Luigi Workflow Engine

The mipper system uses the python `luigi` package to orchestrate the execution
of its tasks. The motivated reader is referred to the luigi docs for more
information: https://luigi.readthedocs.io/.

A few details about our use of luigi:
* Luigi is configured using the `luigi.cfg` file.
* Luigi's logger is configured using the `luigi_logging.conf` file.
* We are using the "local" scheduler, with only one process, until we're sure
  everything is stable. This means that modules will not be executed in
  parallel for a given map, even if the dependecy graph allows for it. (You can
  run multiple instances of mipper, however, using different job ids, to
  different maps in parallel.) 
