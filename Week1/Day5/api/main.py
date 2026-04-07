from flask import Flask
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)


# creating a database

# defining api end points

@app.route("/")
def home():
    return "Home Page"

if __name__=="__main__":
    app.run(debug=True)