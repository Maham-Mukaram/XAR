import os
from src.DB_MODEL import (
    db, User
)
from flask import Flask
from flask_restful import Api
from flask_login import LoginManager,login_user,logout_user,login_required,current_user
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.secret_key="mmmz1234"
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

app.config['SQLALCHEMY_DATABASE_URI']="postgresql://mmmz@mmmz:1Yasmeen!@mmmz.postgres.database.azure.com:5432/postgres"
app.config['SQLALCHEMY_POOL_RECYCLE']=499
app.config['SQLALCHEMY_POOL_TIMEOUT']=20
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False


api = Api(app)

db.init_app(app)
db.create_all(app = app)

app.app_context().push()
login_manager = LoginManager()
login_manager.login_view='login'
login_manager.init_app(app)


@login_manager.user_loader
def load_user(email):
    user = User.query.get(email)
    return user


