if [ "$1" == "-md" ]; then
    python md_tree.py
    echo "Markdown tree generated"
elif [ "$1" == "-g" ] && [ "$2" != "" ]; then
    git add .
    git commit -m "$2"
    git push
    echo "Changes pushed to remote repository"
elif [ "$1" == "-g" ]; then
    git add .
    git commit -m "Update"
    git push
    echo "Changes pushed to remote repository"
else
    echo "Usage: ./run.sh [-md] [-g] [commit message]"
    echo "  -md: Generate Markdown tree"
    echo "  -g: Commit and push changes to remote repository (optional commit message)"
fi