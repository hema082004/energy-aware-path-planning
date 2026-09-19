from flask import Flask

app = Flask(__name__)

@app.route("/")
def home():
    return """
    <html>
        <head>
            <title>Energy-Aware Path Planning</title>
        </head>
        <body>
            <h1>Energy-Aware Path Planning</h1>
            <p>Vercel deployment is working successfully.</p>
            <p>2D and 3D robot path planning project.</p>
        </body>
    </html>
    """
