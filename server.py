from flask import Flask,request
import sys
import os
import transcribe

#Crucial for saving and handling post form data and files
from werkzeug.datastructures import ImmutableMultiDict

from flask import jsonify
import json

app = Flask(__name__)

@app.route("/", methods = ['POST'])
def index():
    #print('Request-form',list(request.form.keys()),file=sys.stderr)
    #print('Request-form-name',request.form['name'],file=sys.stderr)
    #print('Request-files:',request.files,file=sys.stderr)
    #print('Requestfiletype:',type(request.files),file=sys.stderr)

    #data = request.files.to_dict()
    #print(data,file=sys.stderr)   
    data = request.files.to_dict()
   
    #print('data',data,file=sys.stderr)
   
    #to-do Input file validation... (ensure input file is valid jpg or png)
    file = data['upload']
    #print('File name:',file.filename,file=sys.stderr)
    
    file_path = os.path.join("User_Input_Images",file.filename)
    
    #print("Current_Directory:",os.getcwd())
    
    os.chdir("/home/ubuntu/hcr-ann/")
    file.save(file_path)
    
    #print('File saved with name:',file.filename,file=sys.stderr)
    
    #output_text = "Server side work in progress"
    output_text = transcribe.extract(file.filename)
    #output_text = ""
    # if output_text == None:
    #     output_text = "Server Side Failure"
    
    response = json.dumps({'transcription':output_text})
    
    return response

if(__name__ == "__main__"):
    app.run(host = '0.0.0.0',port = 5000)

