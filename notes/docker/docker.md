# [Docker](https://docs.docker.com/reference/)



## Basic Usage: Daily Use with [Docker Command Line Interface (CLI)](https://docs.docker.com/reference/cli/docker/)

### Help

- `docker --help`
- `docker COMMAND --help`
- `docker COMMAND SUBCOMMAND --help`

### Browse Exisinging Images and Containers

- [List images](https://docs.docker.com/reference/cli/docker/image/ls/):
  - `docker image ls [OPTIONS] [REPOSITORY[:TAG]]`
  - Aliases:
    - `docker image ls`
    - `docker image list`
    - **docker images**
- [List containers](https://docs.docker.com/reference/cli/docker/container/ls/):
  - `docker container ls [OPTIONS]`
  - Aliases:
    - `docker container ls`
    - `docker container list`
    - `docker container ps`
    - **docker ps**
  - By default, only show containers that are actively running.
    - To see all containers, append OPTION `-a` or `--all`.

### Clear Cache

Docker loves to cache. This is generally great, but it can sometimes lead to undesired effects.
```bash
# Remove unused images (not associated with a container or a tag)
docker image prune
# Remove the docker build cache ("docker build" is used to build a per-user image)
docker buildx prune
# Remove all stopped containers
docker container prune
```
For more details: [Docker Clear Cache](https://depot.dev/blog/docker-clear-cache).

### Create Docker Container from Images

- Create a new container from images]((https://docs.docker.com/reference/cli/docker/container/create/)):
  - `docker container create [OPTIONS] IMAGE [COMMAND] [ARG...]`
  - Containers are created from images.
    - If the image does not locally exist, if will be downloaded from online register (by default, Docker Hub).
  - Aliases:
    - `docker container create`
    - **docker create**
  - E.g., `docker container create hello-world:latest`
    - The default container for testing docker installation.
    - Will print out a welcome message upon execution.
    - Won't see this message until manually started.
- At this step, the created container is not yet started.
  - Does not automatically execute the image's entry point.

### Start a Container

- [Run one or more stopped containers]((https://docs.docker.com/reference/cli/docker/container/start/):
  - `docker container start [OPTIONS] CONTAINER [CONTAINERS...]`
  - The `CONTAINER` need not to be the full ID.
    - Container tags are also allowed. 
    - The first three characters of its ID should be sufficient in most cases.
  - The same container could be started multiple times.
    - Containers do **not** get deleted by default. 
    - No need to create a new one form an image. 
  - Aliases:
    - `docker container start`
    - **docker start**
  - E.g.:
    - Start the container:
      - `docker container start CONTAINER`
    - Start the container with STDOUT/STDERR attached and signals forwarded:
      - `docker container start --attach CONTAINER`
      - `docker container start --a CONTAINER`
    - Start the container interactively (with STDIN attached):
      - `docker container start --interactive CONTAINER`
      - `docker container start --i CONTAINER`
```
$ docker container create hello-world:latest

$ docker ps
CONTAINER ID   IMAGE                COMMAND    CREATED        STATUS                    PORTS     NAMES

$ docker ps --all
CONTAINER ID   IMAGE                COMMAND    CREATED        STATUS                    PORTS     NAMES
5e245a0c63f2   hello-world:latest   "/hello"   24 hours ago   Created                             keen_ramanujan

$ docker container start 5e2
5e245a0c63f2

$ docker ps --all
CONTAINER ID   IMAGE                COMMAND    CREATED        STATUS                      PORTS     NAMES
5e245a0c63f2   hello-world:latest   "/hello"   24 hours ago   Exited (0) 11 seconds ago             keen_ramanujan

# A zero return code shows the execution was successful.
# Other return codes might show problems or failures.
# To see the container's output, either attach STDOUT/STDERR with -a OPTION,
# Or use the logs command:

$ docker logs 5e2

Hello from Docker!
This message shows that your installation appears to be working correctly.

$
```

### Running a Container

- [Create and run a new container from an image ("the short way")](https://docs.docker.com/reference/cli/docker/container/run/):
  - `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`
  - Automatically create, start, and attach the container.
    - STDIN/STDERR/STDOUT and signals.
    - **Running a container occupies the terminal window!**
  - OPTIONS
    - [Detached mode (`-d`, `--detach`)](https://docs.docker.com/reference/cli/docker/container/run/#detach)
      - Starts a container as a background process that doesn't occupy your terminal window.
      - `docker run --detach our-server`
      - `docker run -d our-server`
    - [Publish or expose port (`-p`, `--expose`)](https://docs.docker.com/reference/cli/docker/container/run/#publish)
      - **Ports in the container must be mapped to ports on the host machine before we could access them.**
      - This is called _publish_ or _exposure_.
      - Formula: _outside, colon, inside_
      - E.g., `docker run -p 127.0.0.1:80:8080/tcp nginx:alpine`.
        - This binds port 8080 of the container to TCP port 80 on 127.0.0.1 of the host.
        - You can also specify UDP and SCTP ports.
        - Note: If you don't specify an IP address (i.e., -p 80:80 instead of -p 127.0.0.1:80:80):
          - Docker publishes the port on all interfaces (address 0.0.0.0) by default.
          - These ports are externally accessible.
    - [Publish all exposed ports (`-P`, `--publish-all`)](https://docs.docker.com/reference/cli/docker/container/run/#publish-all)
      - Publish all exposed ports to random ports.
    - [Mount volume (`-v`)](https://docs.docker.com/reference/cli/docker/container/run/#volume)
      - **Everything created in the container stays inside the container (unless with mountedc volumes)**.
      - `docker  run  -v $(pwd):$(pwd) -w $(pwd) -i -t  ubuntu pwd`
        - The example above mounts the current directory into the container at the same path using the `-v` flag,
        - sets it as the working directory, and then
        - runs the pwd command inside the container.
      - As of Docker Engine version 23, you can use relative paths on the host:
        - `docker  run  -v ./content:/content -w /content -i -t  ubuntu pwd`
    - [Mount volume read-only (--read-only)](https://docs.docker.com/reference/cli/docker/container/run/#read-only)
      - `docker run --read-only -v /icanwrite busybox touch /icanwrite/here`
  - Aliases:
    - `docker container run`
    - **docker run**
```
$ docker run hello-world:latest`

Hello from Docker!
This message shows that your installation appears to be working correctly.

$
```

### Attach to a Running Container

- `docker attach`
  - Conflicts with existing sessions.
  - All sessions will see the same input/output.
- `docker exec -it <container_name_or_id> /bin/bash`
  - Do not conflict with existing sessions. 

### Save and Load an Image

- [docker image save](https://docs.docker.com/reference/cli/docker/image/save/)
  - Save one or more images to a tar archive (streamed to STDOUT by default). 
  - Save built images for further use. Good on shared devices where old Docker images are frequently removed. 
  - Syntax:
    - USAGE: `docker save [OPTIONS] IMAGE [IMAGE...]`
    - OPTIONS:
      - `-o, --output string`: Write to a file, instead of STDOUT
    - Aliases:
      - docker image save
      - **docker save**
  - Example:
    - `docker save -o /path/to/your/image.tar IMAGE` 
- [docker image load](https://docs.docker.com/reference/cli/docker/image/load/)
  - Load an image from a tar archive or STDIN.
  - Load prebuilt images into Docker management. Good on shared devices where old Docker images are frequently removed.
  - Syntax:
    - USAGE: `docker load OPTIONS`
    - OPTIONS:
      - `-i, --input string`: Read from tar archive file, instead of STDIN
    - Aliases:
      - docker image load
      - **docker load**
  - Example:
    - `docker load -i /path/to/your/image.tar`

### Stop or Kill a Running Container

- [Stop one or more running containers](https://docs.docker.com/reference/cli/docker/container/stop/)
  - `docker container stop [OPTIONS] CONTAINER [CONTAINER...]`
  - The main process inside the container will receive **SIGTERM**, and after a grace period, **SIGKILL**. 
  - OPTIONS
    - [`-s, --signal`](https://docs.docker.com/reference/cli/docker/container/stop/#signal): Signal to send to the container.
    - [`-t, --timeout`](https://docs.docker.com/reference/cli/docker/container/stop/#timeout): Seconds to wait before killing the container.
      - E.g., `docker stop -t 0 our-server`:
        - Immediately stop the running container.
        - Be careful, could lead to data loss. 
  - Aliases
    - `docker container stop`
    - **docker stop**
- [Kills one or more containers](https://docs.docker.com/reference/cli/docker/container/kill/):
  - `docker container kill [OPTION] CONTAINER [CONTAINER...]`
  - The main process inside the container is sent:
    - **SIGKILL** signal (default), or the OPTION.
  - OPTION [`-s, --signal`](https://docs.docker.com/reference/cli/docker/container/kill/#signal).
    - The signal can be a name or a number.
    - The SIG prefix is optional.
    - E.g., the following are equivalent:
      - `docker kill --signal=SIGHUP our-server`
      - `docker kill --signal=HUP our-server`
      - `docker kill --signal=1 our-server`
  - CONTAINER can be an ID, an ID-prefix, or a name.
  - Aliases:
    - `docker container kill`
    - **docker kill**

### Remove Containers or Images

- [Remove one or more containers](https://docs.docker.com/reference/cli/docker/container/rm/)
  - `docker container rm [OPTIONS] CONTAINER [CONTAINER...]`
  - Batch removal: `docker ps -aq | xargs docker rm`
    - `-q`: Tell `docker ps` to only output IDs.
    - [xargs](https://man7.org/linux/man-pages/man1/xargs.1.html)
      - A built-in Linux tool that builds and executes command lines from STDIN. 
      - `xargs [options] [command [initial-arguments]]`
    - Take Each ID from `docker ps -aq` and feed it to `docker rm`'s arguments. 
  - Aliases:
    - `docker container remove`
    - `docker container rm`
    - **docker rm**
- [Remove (and un-tag) one or more images](https://docs.docker.com/reference/cli/docker/image/rm/)
  - `docker image rm [OPTIONS] IMAGE [IMAGE...]`
  - If a container uses this image, the container should be removed first. 
    - Unless we use the [`-f`, `--force`](https://docs.docker.com/reference/cli/docker/image/rm/) OPTION.
    - But this could lead to undefined behaviors. 
  - Aliases:
    - `docker image remove`
    - `docker image rm`
    - **docker rmi**



## Advanced Usage: Docker Image Creation

### [Dockerfile](https://docs.docker.com/reference/dockerfile/)

- Files needed to create a local Docker image:
  - [Dockerfile](https://docs.docker.com/reference/dockerfile/)
  - Associated scripts (if any)
```
$ vi Dockerfile 

# Which existing image to pull from, either local or Internet.
# By default, from Docker Hub.
FROM ubuntu

LABEL maintainer="Xi Han <xihan1997@gmail.com>"

# Switch user to run the following commands.
# By default, root.
USER root

# Copy to the specified directory in the container image.
COPY ./entrypoint.bash /

RUN apt -y update
RUN apt -y install curl bash
RUN chmod 755 /entrypoint.bash

USER nobody

# CMD command can be used as well.
ENTRYPOINT [ "/entrypoint.bash" ]

$
```
```
$ vi entrypoint.bash

#!/usr/bin/env bash

echo "Hello from our-first-image :-)"

$
```

### Build a Docker Image from Dockerfiles

- [Build a Docker image](https://docs.docker.com/reference/cli/docker/buildx/build/):
  - `docker buildx build [OPTIONS] PATH | URL | -`
  - PATH: Dockerfile defaults to `PATH/Dockerfile`.
  - [Specify a Dockerfile with OPTIONS `-f` or `--file`](https://docs.docker.com/reference/cli/docker/buildx/build/#file):
    - `--file <filepath>` or `-f <filepath>` specifies the filepath of the Dockerfile to use.
      - Useful when Dockerfile is not named exactly `PATH/Dockerfile`.
      - E.g., `docker build --file server.Dockerfile --tag our-first-server .`
    - To read a Dockerfile from stdin, use `-` as the argument for `--file`:
      - E.g., `cat Dockerfile | docker buildx build -f - .`
  - Aliases:
    - **docker build**
    - `docker builder build`
    - `docker image build`
    - `docker buildx b`
  - E.g.:
```
$ docker build --tag our-first-image .

$ docker images

IMAGE                                                                           ID             DISK USAGE   CONTENT SIZE   EXTRA
hello-world:latest                                                              d4aaab6242e0       25.9kB         9.52kB    U   
our-first-image:latest                                                          0ed09d259d6b        229MB         73.4MB        

$ docker run our-first-image
Hello from our-first-image :-)

$
```

