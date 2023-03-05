flag_file=".gcloud.lock"
if [ ! -e "$flag_file" ]; then
    gcloud init
    gcloud auth application-default login
    echo "Command executed"
    touch "$flag_file"
fi
# cavern.sh should end with a newline character
# since oh-my-zsh's zshrc file is appended after it.
