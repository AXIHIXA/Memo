# --download-directory is not really working for ubuntu release.
# --download-directory "/Hfhf/CG-AI/KK_pixiv" invoked at /mnt/f/ results in /mnt/f/Hfhf/CG-AI/KK_pixiv.
# Invoke at root of F:/!

/mnt/e/opt/PatreonDownloader-AlexCSDev/PatreonDownloader.App.exe \
--url "https://www.patreon.com/c/KK_pixiv/posts" \
--download-directory "/Hfhf/CG-AI/KK_pixiv" \
--file-exists-action "ReplaceIfDifferent" \
--use-sub-directories

/mnt/e/opt/PatreonDownloader-AlexCSDev/PatreonDownloader.App.exe \
--url "https://www.patreon.com/c/KK_pixiv/posts" \
--download-directory "/Hfhf/CG-AI/KK_pixiv_exp" \
--file-exists-action "ReplaceIfDifferent"
