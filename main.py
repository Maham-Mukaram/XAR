from config import app
import os
import re
from src.DB_MODEL import save_image, get_images,getpath,get_usertype, get_users, delete_user, get_username, create_user
from main_model import predict
from report import report_g
from segment_image import segment
from heapmaps import heap
from routes import (
    authRoutes
)
from flask_login import current_user, login_required,logout_user
from flask import Flask, request, render_template, logging, session, redirect, url_for, flash, jsonify
APP_ROOT= os.path.dirname(os.path.abspath(__file__))

#LOGIN PAGE

@app.route('/login')
def login():
    if request.method == 'POST':
        session['email'] = request.form['email']
        User_Type = get_usertype(request.form['email'])
        if str(User_Type) == 'administrator':
            return redirect(url_for('AdminHome'))
        else:
            return redirect(url_for('home'))
    return render_template('login.html')

# SIGN-UP PAGE

@app.route('/signup')
def signup():
    return render_template('signup.html')

#HOME PAGE
@app.route('/')
@app.route('/home')
@login_required
def home():

    print(current_user)
    user_name=get_username(int(current_user.get_id()))
    any_uploads = ''
    any_uploads = get_images(int(current_user.get_id()))
    return render_template('home.html',  user_name=user_name, any_uploads=any_uploads)


@app.route('/AdminHome')
@login_required
def AdminHome():
    user_name = get_username(int(current_user.get_id()))
    print(current_user)
    any_uploads = ''
    any_uploads = get_users(int(current_user.get_id()))
    return render_template('AdminHome.html', user_name=user_name, any_uploads=any_uploads,)


@app.route("/upload", methods=['POST'])
def upload():
    user_id = int(current_user.get_id())
    if not os.path.exists('static/Patient_images/User'+str(user_id)):
        os.makedirs('static/Patient_images/User'+str(user_id))
    target = os.path.join(APP_ROOT, 'static/Patient_images/User'+str(user_id))
    patient_name = request.form.get("name")
    any_uploads = get_images(int(current_user.get_id()))
    if not os.path.isdir(target):
        os.mkdir(target)
    filedata = request.files["file"]
    destination = "/".join([target, patient_name + '.jpg'])
    filedata.save(destination)
    if any_uploads:
        if(patient_name not in any_uploads):
            save_image(int(current_user.get_id()), str(patient_name), str(destination))
            return "success"
        else:
            return "updated"
    else:
        save_image(int(current_user.get_id()), str(patient_name), str(destination))
        return "success"


@app.route("/reg_doctor", methods=['GET', 'POST'])
def reg_doctor():
    fname = request.form.get("fname")
    lname = request.form.get("lname")
    email = request.form.get("email")
    password = request.form.get("password")
    create_user(fname, lname, email, password, "docter")

    return 'success'


@app.route('/patient_name', methods=['GET', 'POST'])
def get_id():
    p_name = request.form.get("patient_name")
    print(p_name)
    if p_name != 'False':
        return redirect(url_for("n1"))
    return render_template('home.html')


@app.route('/getdat')
def getdata():
    user_id = str(int(current_user.get_id()))
    p_name= request.args.get("p_name")
    path = getpath(p_name)
    path = ('User'+user_id+'/'+path)
    caption = report_g(path, p_name)
    finding, classification = predict(path, p_name,user_id)
    heap(path, p_name, user_id)
    print(path)
    print(caption)
    return jsonify({"image_name": p_name, "path" : path, "caption" : caption, "finding" : finding})

@app.route('/getsegment')
def get_segment():
    user_id = str(int(current_user.get_id()))
    p_name = request.args.get("p_name")
    path = getpath(p_name)
    path = ('User' + user_id + '/' + path)
    segment(path, p_name, user_id)
    return jsonify({"image_name": p_name, "path" : path})

@app.route('/get_email')
def get_email():
    u_email = request.args.get("u_email")
    delete_user(u_email)
    return 'ok'



@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('login'))

# MAIN
if __name__ == '__main__':
    app.run(debug=True)