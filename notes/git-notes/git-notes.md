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
```
- [Docker Clear Cache](https://github.com/AXIHIXA/Memo/blob/master/notes/docker/docker.md#clear-cache):
```bash
# Remove unused images (not associated with a container or a tag)
docker image prune
# Remove the docker build cache ("docker build" is used to build a per-user image)
docker buildx prune
# Remove a stopped container
docker rm <container_name>
# Remove all stopped containers
docker container prune
```
