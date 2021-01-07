from tensorflow import keras
from keras.models import load_model
import pickle
from flask import Flask,render_template,request
import pandas as pd
import numpy as np

model = keras.models.load_model('model.h5')
ohe = pickle.load(open('ohe.pkl','rb'))
venue_encoder = pickle.load(open('venue_encoder.pkl','rb'))
team_encoder = pickle.load(open('team_encoder.pkl','rb'))
batting_avg = pickle.load(open('batting_avg.pkl','rb'))
bowling_avg = pickle.load(open('bowling_avg.pkl','rb'))
avg_score_stadium = pickle.load(open('avg_score_stadium.pkl','rb'))
scaler = pickle.load(open('scaler.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    venue_list = avg_score_stadium['venue'].values.tolist()
    team_list = batting_avg['bat_team'].values.tolist()
    return render_template("index.html",venue_list = venue_list,team_list=team_list);

@app.route('/predict',methods=['POST'])
def predict():
    Venue = request.form.get('Venue')
    batting_team = request.form.get('batting_team')
    bowling_team = request.form.get('bowling_team')
    runs = request.form.get('runs')
    wickets = request.form.get('wickets')
    overs = request.form.get('overs')
    runsl = request.form.get('runsl')
    wicketsl = request.form.get('wicketsl')

    v_score = int(avg_score_stadium[avg_score_stadium['venue']==Venue]['avg_score_stadium'].values[0])
    b_score = int(batting_avg[batting_avg['bat_team']==batting_team]['total'].values[0])
    bo_score = int(bowling_avg[bowling_avg['bowl_team']==bowling_team]['total'].values[0])

    df_input = [[Venue,batting_team,bowling_team,runs,wickets,overs,runsl,wicketsl,v_score,b_score,bo_score]]

    innings = pd.DataFrame(df_input, columns=['venue','batting_team','bowling_team','runs','wickets','overs','run_last5','wickets_last5','avg_score_stadium','batting_avg','bowling_avg'])

    innings['venue']=venue_encoder.transform(innings['venue'])
    innings['batting_team'] = team_encoder.transform(innings['batting_team'])
    innings['bowling_team'] = team_encoder.transform(innings['bowling_team'])

    X = innings.drop(['venue','batting_team','bowling_team'],axis=1)
    X_trans = ohe.transform(innings[['venue','batting_team','bowling_team']]).toarray()

    X = np.hstack((X,X_trans))
    X = scaler.transform(X)
    y_pred = model.predict(X)
    score = int(y_pred[0][0])
    return render_template("index.html",score=score)

@app.route('/aboutus')
def about():
    return render_template("AboutUs.html");

if __name__=="__main__":
    app.run(debug=True)