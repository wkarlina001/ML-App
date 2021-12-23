from flask import Flask, render_template, request
import numpy as np
from joblib import dump, load
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import uuid

app = Flask(__name__)

@app.route("/", methods =['GET','POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":

        return render_template('index.html', href="static/base.svg")
    else:
        text = request.form['text']
        random_str = uuid.uuid4().hex
        path = 'app/static/'+random_str+'.svg'
        model_in = load('app/model.joblib')
        make_picture('app/AgesAndHeights.pkl', model_in, floats_string_to_np_arr(text), path)

        return render_template('index.html', href=path[4:])

def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(",") if is_float(x)])
    return floats.reshape(len(floats),1)

def make_picture(train_data_filename, model, new_inp_np_arr, output_file):
    data = pd.read_pickle(train_data_filename)
    ages = data['Age']
    
    data = data[ages>0]
    ages = data['Age']
    heights = data['Height']
    x_new = np.arange(18).reshape((18, 1))
    preds  = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs Age", labels={'x': 'Age (Years)',
                                                                    'y': 'Height (Inches)'})
    fig.add_trace(
        go.Scatter(x=x_new.reshape(x_new.shape[0]), y=preds, mode='lines', name='Model'))

    new_preds  = model.predict(new_inp_np_arr)
    fig.add_trace(
        go.Scatter(x=new_inp_np_arr.reshape(len(new_inp_np_arr)), y=new_preds, 
                    mode='markers', marker =dict(color='purple',size=10),
                    name='New Outputs'))

    fig.write_image(output_file, width = 800, engine='kaleido')
    fig.show()