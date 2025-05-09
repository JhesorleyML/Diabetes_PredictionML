import pickle
import pandas as pd
import os
from rest_framework.decorators import api_view
from rest_framework.response import Response

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
with open(MODEL_PATH, 'rb') as f:
    model = pickle.load(f)

@api_view(['POST'])
def predict(request):
    try:
        #get the data from request
        data = request.data
        features = pd.DataFrame([{
            'gender':data['gender'], 
            'age':data['age'],
            'hypertension':data['hypertension'],
            'heart_disease':data['heart_disease'],
            'smoking_history':data['smoking_history'],
            'bmi':data['bmi'],
            'HbA1c_level':data['HbA1c_level'],
            'blood_glucose_level':data['blood_glucose_level']}])
        prediction = model.predict(features)
        probabilities = model.predict_proba(features)
        return Response({'prediction': prediction.tolist(),'probabilities': probabilities.tolist()})
    except Exception as e:
        return Response({'error':str(e)})


# Create your views here.
# def main(request):
#     return HttpResponse('Hello')
