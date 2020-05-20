if [ -z ${1+x} ]
then
    echo "-m message NOT set, using \"Update\""
    git add -A &&
    git commit -m "Update" &&
    git push
else
    echo "-m \"{$1}\""
    git add -A &&
    git commit -m "\"{$1}\"" &&
    git push
fi



