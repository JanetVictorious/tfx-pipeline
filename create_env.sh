echo ""
echo "+----------------------+"
echo "|  Create Python env   |"
echo "+----------------------+"

if [[ -d .venv ]]
then
    echo ".venv exists in your directory"
    echo "Will not create a virtual environment"
else
    echo ".venv does not exists"
    echo "Creating virtual environment..."
    python -m venv .venv
fi

echo ""
echo "+------------------------+"
echo "|  Compile requirements  |"
echo "+------------------------+"

if [[ ! -f src/requirements.txt ]]
then
    echo "Compiled requirements not found"
    echo "Complie requirements..."
    pip-compile src/requirements.in
fi

echo ""
echo "+-----------------------+"
echo "|  Installing packages  |"
echo "+-----------------------+"

.venv/bin/pip install -r src/requirements.txt

echo ""
echo "+------------------------+"
echo "|  Activate environment  |"
echo "+------------------------+"

source .venv/bin/activate