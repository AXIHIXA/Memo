# GIT

## GIT LFS

- Git LFS
```bash
sudo apt update
sudo apt install -y git-lfs
```
- Clone Repo with LFS
```bash
git clone ssh://xxx.git
```
- Pull LFS content of an existing local repo that is cloned before LFS is installed
```bash
git lfs install
git lfs fetch
git lfs pull
```

## Repository

- Check out to an existing branch
```bash
# First check out the dev branch
git checkout dev
# Update dev to match the upstream version
# (this will discard any changes in your local workspace that are not committed or stashed)
git fetch origin; git reset --hard origin/dev
```
- Confirm you're on the newly created branch
```bash
git branch
```

## BH

- bh Docker tool:
```bash
cd cudnn
chmod +x ./scripts/bh
./scripts/bh run --dry-run
```
- Copy and save the commands for Docker image build and container run. 
```bash
# Check whether this aligns with the latest output!
echo "
FROM urm.nvidia.com/hw-cudnn-docker/dev
RUN groupadd -g 30 hardware ; \
useradd -l -p '' -g hardware --uid 151841 xihan -s /bin/bash ; \
passwd -d xihan ; \
passwd -d root ; \
echo xihan 'ALL=(ALL) NOPASSWD: ALL' >> /etc/sudoers
USER xihan
" | docker build -f - -t urm.nvidia.com/hw-cudnn-docker/dev:xihan-local .
```
```bash
#!/bin/bash
docker \
    run \
    -it \
    --rm \
    --user=151841:30 \
    --volume=/home/xihan:/home/xihan \
    --volume=/home/scratch.xihan_coreai:/home/scratch.xihan_coreai \
    --workdir=/home/xihan \
    --hostname=docker-computelab-304.nvidia.com \
    --name=xihan \
    --gpus=all \
    urm.nvidia.com/hw-cudnn-docker/dev:xihan-local
```
- [Docker Clear Cache](https://github.com/AXIHIXA/Memo/blob/master/notes/docker/docker.md#clear-cache):
```bash
# Remove unused images (not associated with a container or a tag)
docker image prune
# Remove the docker build cache ("docker build" is used to build a per-user image)
docker buildx prune
# Remove all stopped containers
docker container prune
```
