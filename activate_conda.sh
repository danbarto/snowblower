export TWHOME=$PWD
export PYTHONPATH=${PYTHONPATH}:$PWD

if [ $USER == "dspitzba" ]; then
	echo "Hello Daniel!"
	conda activate snowblower
fi
if [ $USER == "ewallace" ]; then
	echo "Hello Jackson!"
	conda activate coffeadev
fi

#( conda activate daskanalysisenv && jupyter notebook --no-browser )
