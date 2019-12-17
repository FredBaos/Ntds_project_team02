# pwd
# export PYTHONPATH='pwd'

echo acquistion
python3 acquisition/acquisition.py
echo exploitation
python3 exploitation/exploitation.py
echo visualization
python3 visualization/create_visu.py #takes time

# On the server
# pwd
# sudo PYTHONPATH='pwd' python3 visualization/app.py

# TODO readme wiki