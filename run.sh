path=pwd
export PYTHONPATH=$path
python3 acquisition/acquisition.py
python3 exploitation/exploitation.py
python3 visualization/create_visu.py #takes time

#pip3 install notebook ipywidgets plotly
#jupyter nbextension enable --py widgetsnbextension
#jupyter nbextension enable --py plotlywidget