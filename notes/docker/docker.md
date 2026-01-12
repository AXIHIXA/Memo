# [Docker](https://docs.docker.com/reference/)

## [Docker Command Line Interface (CLI)](https://docs.docker.com/reference/cli/docker/)

### Help

- `docker --help`
- `docker COMMAND --help`
- `docker COMMAND SUBCOMMAND --help`

### Browse Exisinging Images and Containers

- List images: `docker images`
  - Aliases:
    - `docker image ls`
    - `docker image list`
    - `docker images`
- List containers: `docker ps`
  - By default, only show containers that are actively running.
    - To see all containers, append OPTION `-a` or `--all`.
  - Aliases:
    - `docker container ls`
    - `docker container list`
    - `docker container ps`
    - `docker ps`

### [Create Docker Container from Images](https://docs.docker.com/reference/cli/docker/container/create/)

- Create a new container:
  - `docker container create [OPTIONS] IMAGE [COMMAND] [ARG...]`
  - Containers are created from images.
    - If the image does not locally exist, if will be downloaded from online register (by default, Docker Hub).
  - Aliases:
    - `docker container create`
    - `docker create`
  - E.g., `docker container create hello-world:latest`
    - The default container for testing docker installation.
    - Will print out a welcome message upon execution.
    - Won't see this message until manually started.
- At this step, the created container is not yet started.
  - Does not automatically execute the image's entry point.

### [Start a Container](https://docs.docker.com/reference/cli/docker/container/start/)

- Run one or more stopped containers:
  - `docker container start [OPTIONS] CONTAINER [CONTAINERS...]`
  - The `CONTAINER` need not to be the full ID.
    - Container tags are also allowed. 
    - The first three characters of its ID should be sufficient in most cases.
  - The same container could be started multiple times.
    - Containers do **not** get deleted by default. 
    - No need to create a new one form an image. 
  - Aliases:
    - `docker container start`
    - `docker start`
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

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

$
```

### [Running a Container](https://docs.docker.com/reference/cli/docker/container/run/)

- Create and run a new container from an image ("the short way"):
  - `docker run [OPTIONS] IMAGE [COMMAND] [ARG...]`
  - Automatically create, start, and attach the container.
    - `docker run == docker container create + docker container start + docker container attach`
  - Does **not** show the container ID (like for `docker container create` command).
  - Aliases:
    - `docker container run`
    - `docker run`
```
$ docker run hello-world:latest`

Hello from Docker!
This message shows that your installation appears to be working correctly.

To generate this message, Docker took the following steps:
 1. The Docker client contacted the Docker daemon.
 2. The Docker daemon pulled the "hello-world" image from the Docker Hub.
    (amd64)
 3. The Docker daemon created a new container from that image which runs the
    executable that produces the output you are currently reading.
 4. The Docker daemon streamed that output to the Docker client, which sent it
    to your terminal.

To try something more ambitious, you can run an Ubuntu container with:
 $ docker run -it ubuntu bash

Share images, automate workflows, and more with a free Docker ID:
 https://hub.docker.com/

For more examples and ideas, visit:
 https://docs.docker.com/get-started/

$
```

### Create a Docker Container from [Dockerfiles](https://docs.docker.com/reference/dockerfile/)

- Two files needed to create a local Docker image:
  - Dockerfile
  - entrypoint.bash
- Dockerfile
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
- `entrypoint.bash`
```
$ vi entrypoint.bash

#!/usr/bin/env bash

echo "Hello from our-first-image :-)"
```
- [Build a Docker image](https://docs.docker.com/reference/cli/docker/buildx/build/):
  - `docker build [OPTIONS] PATH | URL | -`
  - Aliases:
    - `docker build`
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

### [Remove a Container](https://docs.docker.com/reference/cli/docker/container/rm/)

- Remove one or more containers:
  - `docker container rm [OPTIONS] CONTAINER [CONTAINER...]`
  - Aliases:
    - `docker container remove`
    - `docker container rm`
    - `docker rm`

### [Remove an Image](https://docs.docker.com/reference/cli/docker/image/rm/)

- Removes (and un-tags) one or more images:
  - `docker image rm [OPTIONS] IMAGE [IMAGE...]`
  - If a container uses this image, the container should be removed first
    - Unless we use the `-f` OPTION.
  - Aliases:
    - `docker image remove`
    - `docker image rm`
    - `docker rmi`

